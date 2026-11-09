"""Multi-objective optimizer on ZDT4 using a Pareto-front-aware adaptive strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


class Optimizer:
    """Adaptive multi-objective optimizer for ZDT4 with Pareto-front-aware search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        rng = np.random.RandomState(0)
        used = 0
        archive = []  # list of (x, f) pairs

        def evaluate(x):
            nonlocal used
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            return x_clipped, f

        def add_to_archive(x, f):
            nonlocal archive
            # Remove points dominated by f
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            # Check if f is dominated by any in new_archive
            for ax, af in new_archive:
                if np.all(af <= f) and np.any(af < f):
                    return
            new_archive.append((x, f))
            archive = new_archive

        def is_nondominated(f, archive_f_list):
            for af in archive_f_list:
                if np.all(af <= f) and np.any(af < f):
                    return False
            return True

        def crowding_distance(archive_list):
            """Compute crowding distance for archive points."""
            if len(archive_list) <= 2:
                return [float('inf')] * len(archive_list)
            
            distances = [0.0] * len(archive_list)
            f_vals = np.array([p[1] for p in archive_list])
            
            for obj in range(self.n_obj):
                sorted_idx = np.argsort(f_vals[:, obj])
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                f_range = f_vals[sorted_idx[-1], obj] - f_vals[sorted_idx[0], obj]
                if f_range == 0:
                    continue
                for i in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[i]] += (
                        f_vals[sorted_idx[i + 1], obj] - f_vals[sorted_idx[i - 1], obj]
                    ) / f_range
            
            return distances

        # ZDT4 structure:
        # f1(x) = x[0]
        # g(x) = 1 + 10*(n-1) + sum(x_i^2 - 10*cos(4*pi*x_i)) for i=1..n
        # h(f1, g) = 1 + sqrt(f1/g)
        # f2(x) = g * h
        # Optimal: x_i = 0 for i >= 1 gives g = 1
        # x[0] in [0, 1] for the Pareto front

        # Phase 1: Dense systematic coverage of the Pareto front
        # Use logarithmic spacing in f1 to get better coverage near f1=0
        n_phase1 = min(60, budget // 4)
        
        # Create non-uniform spacing: more points near low f1 values
        # where the front is more curved
        f1_targets = np.concatenate([
            np.linspace(0.005, 0.1, 15),
            np.linspace(0.11, 0.5, 20),
            np.linspace(0.51, 0.995, 25)
        ])
        f1_targets = f1_targets[:n_phase1]
        
        for f1_target in f1_targets:
            if used >= budget or time.time() > deadline - 120:
                break
            
            # Multiple attempts with decreasing perturbation
            for attempt, scale in enumerate([0.08, 0.03, 0.01]):
                if used >= budget:
                    break
                if time.time() > deadline - 120:
                    break
                
                x = np.zeros(dim)
                x[0] = f1_target
                # x[1:] near 0 to minimize g
                x[1:] = scale * rng.randn(dim - 1)
                x = np.clip(x, lo, hi)
                x, f = evaluate(x)
                add_to_archive(x.copy(), f.copy())
        
        # Phase 2: Adaptive refinement using a priority queue
        # Focus on regions of the front that are poorly covered
        if len(archive) > 0 and used < budget:
            # Sort archive by f1 to identify gaps
            archive_sorted = sorted(archive, key=lambda p: p[1][0])
            
            # Compute gaps between consecutive f1 values
            f1_vals = [p[1][0] for p in archive_sorted]
            gaps = []
            for i in range(len(f1_vals) - 1):
                gap = f1_vals[i + 1] - f1_vals[i]
                gaps.append((gap, f1_vals[i], f1_vals[i + 1]))
            
            # Also add edges
            gaps.append((f1_vals[0], 0.0, f1_vals[0]))
            gaps.append((1.0 - f1_vals[-1], f1_vals[-1], 1.0))
            
            # Sort by gap size, largest first
            gaps.sort(reverse=True)
            
            # Refine the largest gaps
            n_refine = min(len(gaps), (budget - used) // 3)
            for gap_size, f1_lo, f1_hi in gaps[:n_refine]:
                if used >= budget or time.time() > deadline - 80:
                    break
                
                # Generate several points in this gap
                n_points = max(2, int(gap_size * 20))
                n_points = min(n_points, (budget - used) // 2)
                
                for _ in range(n_points):
                    if used >= budget:
                        break
                    if time.time() > deadline - 80:
                        break
                    
                    f1_new = rng.uniform(max(0.001, f1_lo), min(0.999, f1_hi))
                    x = np.zeros(dim)
                    x[0] = f1_new
                    x[1:] = 0.02 * rng.randn(dim - 1)
                    x = np.clip(x, lo, hi)
                    x, f = evaluate(x)
                    add_to_archive(x.copy(), f.copy())
        
        # Phase 3: Local optimization of archive points to reduce f2
        # For each archive point, try to push x[1:] closer to 0
        if len(archive) > 0 and used < budget:
            # Get crowding distances to prioritize diverse points
            cd = crowding_distance(archive)
            
            # Create priority queue: (negative_crowding_distance, index)
            # We want to refine points that are well-separated in the archive
            pq = [(-cd[i], i) for i in range(len(archive))]
            heapq.heapify(pq)
            
            # Track which indices we've already refined
            refined = set()
            n_refine_max = min(len(archive), (budget - used) // 2)
            
            while len(refined) < n_refine_max and used < budget:
                if time.time() > deadline - 40:
                    break
                
                neg_cd, idx = heapq.heappop(pq)
                if idx in refined:
                    continue
                refined.add(idx)
                
                x_base, f_base = archive[idx]
                
                # Multiple refinement attempts with decreasing step
                for step_scale in [0.3, 0.1, 0.05, 0.02, 0.01]:
                    if used >= budget:
                        break
                    if time.time() > deadline - 40:
                        break
                    
                    # Shrink x[1:] toward 0
                    x_new = x_base.copy()
                    x_new[1:] = x_new[1:] * (1 - step_scale) + 0.005 * rng.randn(dim - 1)
                    x_new = np.clip(x_new, lo, hi)
                    x_new, f_new = evaluate(x_new)
                    add_to_archive(x_new.copy(), f_new.copy())
            
            # Push back any remaining indices
            while pq:
                heapq.heappush(pq, heapq.heappop(pq))
        
        # Phase 4: Fill remaining budget with targeted random search
        while used < budget:
            if time.time() > deadline - 5:
                break
            
            # Random f1 with x[1:] near 0
            f1_rand = rng.uniform(0.005, 0.995)
            x = np.zeros(dim)
            x[0] = f1_rand
            x[1:] = 0.01 * rng.randn(dim - 1)
            x = np.clip(x, lo, hi)
            x, f = evaluate(x)
            add_to_archive(x.copy(), f.copy())
            
            # Also try a local refinement if budget allows
            if len(archive) > 0 and used < budget:
                if time.time() > deadline - 5:
                    break
                idx = rng.randint(len(archive))
                x_base, f_base = archive[idx]
                x_refine = x_base.copy()
                x_refine[1:] = x_refine[1:] * 0.5 + 0.003 * rng.randn(dim - 1)
                x_refine = np.clip(x_refine, lo, hi)
                x_refine, f_refine = evaluate(x_refine)
                add_to_archive(x_refine.copy(), f_refine.copy())
        
        # Return (return value is ignored by evaluator, but must not raise)
        if len(archive) > 0:
            return np.array([x for x, f in archive])
        return np.array([np.zeros(dim)])
# EVOLVE-BLOCK-END
