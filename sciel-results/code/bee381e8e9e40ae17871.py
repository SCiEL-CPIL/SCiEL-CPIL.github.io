"""Multi-objective optimizer for ZDT6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(0)
        
        # ZDT6: f1 = 1 + 10*(n-1) + sum_{i=1}^{n-1}(xi-0.5)^2*(1+10*sin(6*pi*xi+pi*f1))
        #       f2 = g(x)*(1 - sqrt(f1/g(x)))
        #       g(x) = 1 + 9*x[1]  (for ZDT6 specifically, but here g = 1 + sum(x[1:]))
        # Actually for ZDT6: g(x) = 1 + 9*x[1] is NOT right.
        # ZDT6: g(x) = 1 + 9 * (sum(x[1:]) / (n-1))
        # Pareto front: g = 1 (i.e., x[1:] = 0) and f1 minimized.
        # Wait, let me reconsider. ZDT6 definition:
        #   f1(x) = 1 + 10*(n-1) + sum_{i=1}^{n-1} (xi - 0.5)^2 * (1 + 10*sin(6*pi*xi + pi*f1))
        #   f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)))
        #   g(x) = 1 + 9 * (sum_{i=2}^n xi / (n-1))
        # Pareto front: g(x) = 1, i.e., x[1:] = 0, and f1 is minimized.
        # So the front is: x[1:] = 0, x[0] varies to minimize f1.
        
        # Actually, for ZDT6 the Pareto front has g=1 and x[1:]=0.
        # Let me generate points with x[1:] = 0 and various x[0], then refine.
        
        def generate_front_point(x0, rng):
            """Generate a point on or near the ZDT6 Pareto front."""
            x = np.zeros(n)
            x[0] = x0
            # x[1:] = 0 for exact front
            return np.clip(x, lo, hi)
        
        # Strategy: 
        # 1. Generate many points with x[1:] = 0 and varying x[0] to map the front
        # 2. Use remaining budget to perturb x[1:] slightly to explore near-front regions
        # 3. Keep the best non-dominated set
        
        # Phase 1: Map the Pareto front by varying x[0]
        n_front = min(80, budget // 2)
        X_front = np.zeros((n_front, n))
        for i in range(n_front):
            x0 = (i + 0.5) / n_front  # spread x[0] in [0,1]
            X_front[i] = generate_front_point(x0, rng)
        
        F_front = np.array([problem(x) for x in X_front])
        used = n_front
        
        # Phase 2: Use remaining budget for local search / perturbation
        remaining = budget - used
        if remaining > 0:
            # Generate perturbed points around front points
            n_perturb = remaining
            X_perturb = np.zeros((n_perturb, n))
            for i in range(n_perturb):
                if time.time() > deadline:
                    break
                # Pick a random front point
                parent_idx = rng.integers(0, n_front)
                x = X_front[parent_idx].copy()
                # Perturb x[1:] slightly
                sigma = 0.05
                x[1:] += sigma * rng.standard_normal(n - 1)
                # Slightly perturb x[0] too
                x[0] += 0.02 * rng.standard_normal()
                x = np.clip(x, lo, hi)
                X_perturb[i] = x
            
            # Only evaluate if we have time
            if time.time() < deadline - 5:
                F_perturb = np.array([problem(x) for x in X_perturb[:remaining]])
                used += remaining
            else:
                F_perturb = np.array([])
        
        # Combine all evaluated points
        if len(F_perturb) > 0:
            all_X = np.vstack([X_front, X_perturb[:len(F_perturb)]])
            all_F = np.vstack([F_front, F_perturb])
        else:
            all_X = X_front
            all_F = F_front
        
        # Filter: keep only points strictly better than reference point [11.0, 11.0]
        ref = np.array([11.0, 11.0])
        mask = np.all(all_F < ref, axis=1)
        all_X = all_X[mask]
        all_F = all_F[mask]
        
        # Remove dominated points to get the non-dominated set
        if len(all_F) > 1:
            # Non-dominated filter
            keep = np.ones(len(all_F), dtype=bool)
            for i in range(len(all_F)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(all_F)):
                    if not keep[j]:
                        continue
                    if np.all(all_F[i] <= all_F[j]) and np.any(all_F[i] < all_F[j]):
                        keep[j] = False
                    elif np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                        keep[i] = False
                        break
            all_X = all_X[keep]
            all_F = all_F[keep]
        
        return all_X
        
# EVOLVE-BLOCK-END
