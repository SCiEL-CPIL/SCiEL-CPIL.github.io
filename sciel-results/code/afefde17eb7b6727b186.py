import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 325.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_sequence(n, dim, start=1):
            seq = np.zeros((n, dim))
            for i in range(n):
                for d in range(dim):
                    seq[i, d] = halton(start + i, d + 2)
            return seq

        # 1. Initialize with Halton sequence for uniform coverage
        n_init = 50
        if n_init > budget:
            n_init = budget

        init_seq = halton_sequence(n_init, dim)
        init_X = lo + init_seq * (hi - lo)
        
        archive_X = []
        archive_F = []
        
        for i in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = init_X[i]
            f = evaluate(x)
            archive_X.append(x)
            archive_F.append(f)

        # Helper to check if a point is dominated by any in the archive
        def is_dominated(f_new, archive_F):
            for f_arch in archive_F:
                if np.all(f_arch <= f_new) and np.any(f_arch < f_new):
                    return True
            return False

        # Helper to add to archive if non-dominated
        def add_to_archive(x, f):
            if is_dominated(f, archive_F):
                return False
            # Remove dominated points from archive
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            # Update global lists
            archive_X[:] = new_archive_X
            archive_F[:] = new_archive_F
            return True

        # Update archive with initial points (some might be dominated)
        temp_archive_X = []
        temp_archive_F = []
        for x, f in zip(archive_X, archive_F):
            if not is_dominated(f, temp_archive_F):
                # Remove dominated from temp
                new_x = []
                new_f = []
                for ax, af in zip(temp_archive_X, temp_archive_F):
                    if not (np.all(f <= af) and np.any(f < af)):
                        new_x.append(ax)
                        new_f.append(af)
                temp_archive_X = new_x
                temp_archive_F = new_f
                temp_archive_X.append(x)
                temp_archive_F.append(f)
        archive_X = temp_archive_X
        archive_F = temp_archive_F

        # 2. Local Search Phase
        # Strategy: Pick a point from archive, perform local search to improve one objective
        # while constraining the other, or just do a generic local search and keep if non-dominated.
        
        # We will use a simple adaptive step size local search.
        # For each step:
        # 1. Pick a random point from the current archive.
        # 2. Generate a perturbation.
        # 3. Evaluate.
        # 4. If non-dominated, add to archive.
        
        # To be more effective, we can bias the search towards the "knee" or spread.
        # But simple random selection from Pareto front + local search is robust.
        
        # Let's use a mix of:
        # A) Pure local search on random archive point
        # B) "Epsilon-constraint" style: Pick a target f2, minimize f1.
        #    This is complex to implement strictly with black-box.
        
        # Simpler and effective: 
        # Maintain a set of "active" search centers (the archive).
        # For each new eval:
        # - Select a center from archive (random or weighted by hypervolume contribution).
        # - Step out from center.
        
        # Let's implement Hypervolume Contribution selection for parent choice.
        # Ref point: [3000.0, 0.06] as per task description for scoring, 
        # but for internal selection we can use the current bounds of the archive.
        
        ref_point = np.array([3000.0, 0.06])
        
        def hypervolume_2d(points, ref):
            # points: list of (f1, f2)
            # Sort by f1
            if len(points) == 0:
                return 0.0
            pts = sorted(points, key=lambda x: x[0])
            hv = 0.0
            prev_f2 = ref[1]
            for f1, f2 in pts:
                if f2 < prev_f2:
                    hv += (ref[0] - f1) * (prev_f2 - f2)
                    prev_f2 = f2
                else:
                    # If f2 is not better than previous, it doesn't add HV in this sorted order?
                    # Wait, standard 2D HV calculation:
                    # Sort by f1 ascending.
                    # For each point, if f2 is lower than all previous f2s, it contributes.
                    # Contribution = (ref[0] - f1) * (prev_min_f2 - f2)
                    pass
            
            # Correct 2D HV:
            pts = sorted(points, key=lambda x: x[0])
            hv = 0.0
            max_f2_so_far = -np.inf
            for f1, f2 in pts:
                if f2 < ref[1] and f1 < ref[0]:
                    if f2 < max_f2_so_far:
                        hv += (ref[0] - f1) * (max_f2_so_far - f2)
                        max_f2_so_far = f2
                    else:
                        # This point is dominated in HV sense by previous points? 
                        # No, if f2 is higher, it doesn't add new area below the curve.
                        # But we need to handle the first point.
                        if max_f2_so_far == -np.inf:
                            hv += (ref[0] - f1) * (ref[1] - f2)
                            max_f2_so_far = f2
            # The above logic is slightly flawed for the first point.
            # Let's use a robust O(N log N) or just O(N^2) since N is small.
            
            # O(N^2) Hypervolume for 2D
            # Filter points better than ref
            valid_pts = [(f1, f2) for f1, f2 in points if f1 < ref[0] and f2 < ref[1]]
            if not valid_pts:
                return 0.0
            
            # Sort by f1
            valid_pts.sort(key=lambda x: x[0])
            
            hv = 0.0
            prev_f2 = ref[1]
            for f1, f2 in valid_pts:
                if f2 < prev_f2:
                    hv += (ref[0] - f1) * (prev_f2 - f2)
                    prev_f2 = f2
            return hv

        def calc_hv_contributions(archive_F, ref):
            if not archive_F:
                return []
            
            # Total HV
            total_hv = hypervolume_2d(archive_F, ref)
            
            # Contribution of each point i is Total HV - HV of archive without i
            contributions = []
            n = len(archive_F)
            for i in range(n):
                # Create list without i
                subset = [f for j, f in enumerate(archive_F) if j != i]
                hv_subset = hypervolume_2d(subset, ref)
                contributions.append(total_hv - hv_subset)
            
            return contributions

        # Main Loop
        step_size_base = 0.15
        min_step_size = 0.01
        
        while used < budget and time.time() < deadline:
            if not archive_X:
                # Fallback if archive is empty (shouldn't happen)
                x = lo + (hi - lo) * np.random.random(dim)
                f = evaluate(x)
                archive_X.append(x)
                archive_F.append(f)
                continue

            # Calculate HV contributions for selection
            # This is O(N^2) per step. N is at most 200. 200^2 = 40000 ops. 
            # 200 steps * 40000 = 8M ops. This might be slow in Python.
            # Optimization: Only recalculate contributions every K steps or use a simpler metric.
            # Simpler metric: Distance to ideal point or just random selection.
            # Let's stick to random selection for speed, but maybe weight by "age" or just random.
            # Actually, let's just pick a random point from the archive.
            
            idx = np.random.randint(len(archive_X))
            parent = archive_X[idx]
            
            # Adaptive step size: start moderate, shrink if no improvements?
            # For now, use a fixed moderate step size with some randomness.
            step_size = step_size_base * (hi - lo)
            
            # Generate offspring
            # Gaussian perturbation
            child = parent + np.random.randn(dim) * step_size
            child = np.clip(child, lo, hi)
            
            f = evaluate(child)
            
            # Try to add to archive
            if not is_dominated(f, archive_F):
                # Remove dominated points
                new_archive_X = []
                new_archive_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not (np.all(f <= af) and np.any(f < af)):
                        new_archive_X.append(ax)
                        new_archive_F.append(af)
                
                new_archive_X.append(child)
                new_archive_F.append(f)
                
                archive_X = new_archive_X
                archive_F = new_archive_F
                
                # If we added a point, maybe we should do a bit more local search around it?
                # No, just continue.
                
            # Optional: If we didn't improve, maybe try a larger step or different parent?
            # For simplicity, just continue.

        return all_X
