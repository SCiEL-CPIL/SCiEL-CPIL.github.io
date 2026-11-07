"""Optimized multi-objective optimizer for RE21 with hypervolume-guided adaptive DE."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.stats.qmc as qmc


class Optimizer:
    """Hypervolume-guided adaptive differential evolution for 2-objective RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([3000.0, 0.06])

        def hypervolume_2d(points):
            """Exact hypervolume for 2D minimization with given reference point.
            Only points strictly better than ref in both objectives contribute."""
            if len(points) == 0:
                return 0.0
            
            # Filter points strictly better than ref in both objectives
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective (ascending)
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            hv = 0.0
            min_y = ref[1]
            for i in range(len(pts)):
                x_i, y_i = pts[i]
                if y_i < min_y:
                    next_x = pts[i + 1, 0] if i + 1 < len(pts) else ref[0]
                    hv += (next_x - x_i) * (min_y - y_i)
                    min_y = y_i
            return hv

        def hv_contributions(points):
            """Calculate marginal hypervolume contribution of each point."""
            n = len(points)
            if n == 0:
                return np.array([])
            
            total_hv = hypervolume_2d(points)
            contribs = np.zeros(n)
            for i in range(n):
                # Create a copy of points without point i
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                reduced_points = points[mask]
                reduced_hv = hypervolume_2d(reduced_points)
                contribs[i] = total_hv - reduced_hv
            return contribs

        # Deterministic initialization using Sobol sequence
        # We want to cover the space well.
        n_init = min(60, budget // 3)
        
        # Use Sobol for better coverage than random
        # scipy.stats.qmc.Sobol
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Generate n_init points in [0, 1]^d
        # Note: Sobol sequence requires power of 2 for best properties, but works for others
        # We will generate a power of 2 >= n_init and take the first n_init
        m = int(2 ** np.ceil(np.log2(n_init)))
        samples = sobol.random(m)
        X = lo + (hi - lo) * samples[:n_init]
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Main loop
        pop_size = n_init
        rng = np.random.RandomState(42)
        
        while used < budget:
            # Calculate contributions to identify worst point
            contribs = hv_contributions(F)
            
            # Get non-dominated points for DE parents
            # Simple non-domination sort
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            nd_idx = np.where(~dominated)[0]
            
            if len(nd_idx) < 3:
                # If not enough non-dominated, use all
                nd_idx = np.arange(n)
            
            # Adaptive mutation factor
            progress = min(1.0, used / budget)
            F_mut = 0.8 - 0.6 * progress
            
            # Generate one child (or more if budget allows, but let's stick to 1 for control)
            # Actually, let's generate as many as we can fit in the budget in this step?
            # The loop structure is "while used < budget".
            # Let's generate 1 child per iteration to allow fine-grained contribution updates.
            # But that might be slow due to HV calculation.
            # Let's generate a batch.
            
            batch_size = min(10, budget - used)
            
            for _ in range(batch_size):
                if used >= budget:
                    break
                
                # Pick 3 distinct parents from non-dominated set
                if len(nd_idx) >= 3:
                    p1_idx, p2_idx, p3_idx = rng.choice(len(nd_idx), size=3, replace=False)
                    p1, p2, p3 = X[nd_idx[p1_idx]], X[nd_idx[p2_idx]], X[nd_idx[p3_idx]]
                    mutant = p1 + F_mut * (p2 - p3)
                elif len(nd_idx) == 2:
                    p1, p2 = X[nd_idx[0]], X[nd_idx[1]]
                    mutant = p1 + F_mut * (p2 - p1) * rng.random()
                else:
                    mutant = X[nd_idx[0]].copy()
                
                # Add perturbation
                mutant = mutant + 0.05 * (hi - lo) * rng.standard_normal(dim)
                mutant = np.clip(mutant, lo, hi)
                
                f_mut = problem(mutant)
                used += 1
                
                # Find the worst point in current population based on HV contribution
                # We need to update contributions after adding the new point?
                # Or just replace the current worst?
                # To be efficient, we can just replace the point with the lowest contribution in the *current* set.
                # The new point will be added, and then we remove the worst.
                
                # Add new point to population
                X = np.vstack([X, mutant])
                F = np.vstack([F, [f_mut]])
                
                # Now population size is pop_size + 1
                # Calculate contributions for the new set
                contribs = hv_contributions(F)
                
                # Find the index of the point with the minimum contribution
                worst_idx = np.argmin(contribs)
                
                # Remove the worst point
                mask = np.ones(len(F), dtype=bool)
                mask[worst_idx] = False
                X = X[mask]
                F = F[mask]
                
                # Update non-dominated indices for next iteration?
                # We can just recalculate nd_idx in the next while loop iteration.
                # But we are inside the batch loop.
                # It's safer to break the batch and let the while loop handle it,
                # or recalculate nd_idx here.
                # Given the small size, recalculating is fine.
                n = len(F)
                dominated = np.zeros(n, dtype=bool)
                for i in range(n):
                    if dominated[i]:
                        continue
                    for j in range(n):
                        if i == j or dominated[j]:
                            continue
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
                nd_idx = np.where(~dominated)[0]
                if len(nd_idx) < 3:
                    nd_idx = np.arange(n)

        return X
# EVOLVE-BLOCK-END
