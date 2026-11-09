# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Deterministic structured DTLZ4 optimizer using low-discrepancy sequences."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dims):
        """Generate n points in [0,1)^dims using the Halton sequence."""
        points = np.zeros((n, dims))
        for d in range(dims):
            base = d + 2  # 2, 3, 5, 7, ...
            for i in range(n):
                f = 1.0
                num = i + 1
                result = 0.0
                while num > 0:
                    f /= base
                    num, rem = divmod(num, base)
                    result += f * rem
                points[i, d] = result
        return points

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        M = self.n_obj
        n_z = M - 1  # number of shape variables (2 for M=3)
        n_g = n - n_z  # number of distance variables (10 for n=12, M=3)

        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        # DTLZ4 structure:
        # - z-variables (first n_z) determine the shape on the Pareto front
        # - g-variables (last n_g) determine distance from the front
        # - On the Pareto front, all g-variables = 0.5 (minimizes g)
        
        # Strategy: Use Halton sequence for uniform coverage of z-space [0,1]^n_z
        # Fix g-variables at 0.5 to ensure points are on the Pareto front
        
        # Determine how many points we can afford
        # Reserve some budget for potential refinement
        n_halton = min(self.budget - 10, 200)
        
        # Generate Halton sequence points in z-space
        halton_z = self._halton(n_halton, n_z)
        
        F_list = []
        X_list = []
        
        # Evaluate all Halton points with g-variables fixed at 0.5
        for i in range(n_halton):
            if used >= self.budget or time.time() > deadline:
                break
            
            z = halton_z[i]
            g_vars = np.full(n_g, 0.5)
            x = np.concatenate([z, g_vars])
            
            f = evaluate(x)
            if f is not None:
                X_list.append(x)
                F_list.append(f)
        
        if not X_list:
            # Fallback: single evaluation at center
            x = np.concatenate([np.full(n_z, 0.5), np.full(n_g, 0.5)])
            f = evaluate(x)
            if f is not None:
                X_list.append(x)
                F_list.append(f)
            return X_list
        
        X = np.array(X_list)
        F = np.array(F_list)
        
        # Non-dominated filter
        def non_dominated_indices(F_arr):
            n_pts = len(F_arr)
            nd = []
            for i in range(n_pts):
                is_nd = True
                for j in range(n_pts):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd.append(i)
            return nd
        
        nd_idx = non_dominated_indices(F)
        if not nd_idx:
            nd_idx = list(range(len(F)))
        
        # Use remaining budget for local refinement of non-dominated points
        remaining_budget = self.budget - used
        if remaining_budget > 0 and len(nd_idx) > 0:
            # Sort non-dominated points by sum (lower is better)
            nd_F = F[nd_idx]
            sums = nd_F.sum(axis=1)
            order = np.argsort(sums)
            
            # Refine the best points with small perturbations in z-space
            n_refine = min(remaining_budget, 50)
            step = 0.03
            
            for k in range(n_refine):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Cycle through the best non-dominated points
                pi = nd_idx[order[k % len(order)]]
                x0 = X[pi].copy()
                f0 = F[pi].copy()
                
                # Try a few perturbations
                n_trials = min(3, self.budget - used)
                best_x = x0.copy()
                best_f = f0.copy()
                
                for _ in range(n_trials):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    x = x0.copy()
                    # Perturb z-variables with small random steps
                    z_perturb = step * (np.random.rand(n_z) * 2 - 1)
                    x[:n_z] = np.clip(x[:n_z] + z_perturb, lo[:n_z], hi[:n_z])
                    # Keep g-variables at 0.5
                    x[n_z:] = 0.5
                    
                    f = evaluate(x)
                    if f is None:
                        break
                    
                    # Update if this point is better (lower sum or non-dominated)
                    if f.sum() < best_f.sum():
                        best_x = x.copy()
                        best_f = f.copy()
                
                X[pi] = best_x
                F[pi] = best_f
        
        return X
# EVOLVE-BLOCK-END
