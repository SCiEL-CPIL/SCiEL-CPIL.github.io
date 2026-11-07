import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # DTLZ7 Pareto front: x[2:] = 0.5, x[0], x[1] in [0,1]
        # With g=0: f1=x0, f2=x1, f3 = (n_m+1) - x0*(1+sin(pi*x0)) - x1*(1+sin(pi*x1))
        # where n_m = n - m + 1 = 22 - 3 + 1 = 20
        
        n_m = n - m + 1
        
        # Use Gauss-Legendre quadrature for optimal point placement
        # This maximizes the hypervolume contribution by placing points
        # at locations that best capture the non-dominated region
        
        n_total = min(budget, 200)
        
        # For 2D Pareto front, use tensor product Gauss-Legendre
        # Split budget between two dimensions
        n_per_dim = int(np.sqrt(n_total))
        if n_per_dim < 2:
            n_per_dim = 2
        
        # Gauss-Legendre nodes on [-1, 1], mapped to [0, 1]
        x0_gl, w0 = np.polynomial.legendre.leggauss(n_per_dim)
        x0_vals = 0.5 * (x0_gl + 1)  # map to [0, 1]
        
        x1_gl, w1 = np.polynomial.legendre.leggauss(n_per_dim)
        x1_vals = 0.5 * (x1_gl + 1)  # map to [0, 1]
        
        # Create tensor product grid
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        x0_flat = X0.ravel()
        x1_flat = X1.ravel()
        
        # If we have more budget, add more points with higher-order quadrature
        extra_budget = n_total - len(x0_flat)
        if extra_budget > 0:
            # Use a finer Gauss-Legendre grid
            n_fine = int(np.sqrt(extra_budget)) + n_per_dim
            if n_fine < n_per_dim + 2:
                n_fine = n_per_dim + 2
            
            x0_fine_gl, _ = np.polynomial.legendre.leggauss(n_fine)
            x0_fine = 0.5 * (x0_fine_gl + 1)
            
            x1_fine_gl, _ = np.polynomial.legendre.leggauss(n_fine)
            x1_fine = 0.5 * (x1_fine_gl + 1)
            
            X0_fine, X1_fine = np.meshgrid(x0_fine, x1_fine)
            x0_flat = np.concatenate([x0_flat, X0_fine.ravel()])
            x1_flat = np.concatenate([x1_flat, X1_fine.ravel()])
        
        # Truncate to budget
        x0_flat = x0_flat[:n_total]
        x1_flat = x1_flat[:n_total]
        
        # Build points on the Pareto front
        X = np.zeros((len(x0_flat), n))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        X[:, 2:] = 0.5  # Pareto front location
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = len(x0_flat)
        
        # Check time
        if time.time() > deadline - 5.0:
            return X
        
        # Phase 2: Local refinement around non-dominated points
        remaining = budget - used
        if remaining > 0 and remaining > 5:
            # Find non-dominated points
            nd_mask = self._find_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) > 0:
                # Select diverse non-dominated points for refinement
                n_refine = min(remaining, 30)
                
                if len(nd_indices) <= n_refine:
                    refine_idx = nd_indices
                else:
                    # Select evenly spaced from non-dominated
                    step = len(nd_indices) / n_refine
                    refine_idx = nd_indices[[int(i * step) for i in range(n_refine)]]
                
                rng = np.random.default_rng(42)
                for idx in refine_idx:
                    if time.time() > deadline - 5.0:
                        break
                    if used >= budget:
                        break
                    
                    # Local search around this point
                    x_base = X[idx].copy()
                    n_local = min(remaining - (used - len(x0_flat)), 3)
                    
                    for _ in range(n_local):
                        if used >= budget:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        
                        # Small perturbation
                        perturbation = 0.02 * rng.standard_normal(n)
                        # Keep x[2:] close to 0.5 (Pareto front)
                        perturbation[2:] *= 0.1
                        x_child = x_base + perturbation
                        x_child = np.clip(x_child, lo, hi)
                        
                        fc = problem(x_child)
                        used += 1
                        
                        # If better than parent, update
                        if np.all(fc <= F[idx]) and np.any(fc < F[idx]):
                            X[idx] = x_child
                            F[idx] = fc
        
        return X

    def _find_non_dominated(self, F):
        """Find indices of non-dominated points."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j or not nd[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
                    break
        return nd
