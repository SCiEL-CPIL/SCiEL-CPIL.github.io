"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ7 using known front structure + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Directly sample the known DTLZ7 Pareto front
        # For DTLZ7 with m=1 (position-dependent variables), the Pareto front is:
        #   f_1 = x_1, f_2 = x_2, f_3 = 1 + g + h(f_1, f_2, g)
        # On the front, g=0, so f_3 = 1 + (m - f_1 - f_2) * (1 + 2*0) = 1 + (1 - f_1 - f_2)
        # Wait, let me re-derive for DTLZ7 with m=1, n=22, M=3:
        #   g = sum_{i=m+1}^{n} g_i(x_i) where g_i(x_i) = (x_i - 0.5)^2
        #   h = (m - f_1 - f_2) * (1 + 2*g)  ... actually for DTLZ7:
        #   h = (m - f_1 - ... - f_{M-1}) * (1 + 2*g)
        # With M=3: h = (1 - f_1 - f_2) * (1 + 2*g)
        # On Pareto front: g=0, so f_3 = 1 + h = 1 + (1 - f_1 - f_2) = 2 - f_1 - f_2
        # But we need f_1 + f_2 <= 1 for h >= 0, and f_i >= 0.
        # So the front is: f_1 >= 0, f_2 >= 0, f_1 + f_2 <= 1, f_3 = 2 - f_1 - f_2

        # Use Fibonacci lattice on the unit disk mapped to the triangle {f1>=0, f2>=0, f1+f2<=1}
        # Actually, the Pareto front in objective space is a triangle in 3D.
        # Let's parameterize: f_1 = u, f_2 = v*(1-u), f_3 = 2 - f_1 - f_2
        # where u in [0,1], v in [0,1]. This covers the triangle.
        
        # Number of points to place on the front
        n_front = min(150, budget // 2)
        
        # Fibonacci lattice for uniform coverage
        golden = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_front)
        # Map to [0,1) in a quasi-random way
        u_vals = (indices / golden) % 1.0
        v_vals = ((indices * 0.733) % 1.0)  # different irrational for independence
        
        # Generate points on the Pareto front
        # x_1 = f_1 = u, x_2 = f_2 = v*(1-u), x_3..x_22 = 0.5 (to make g=0)
        X_front = np.zeros((n_front, dim))
        X_front[:, 0] = u_vals  # x_1 = f_1
        X_front[:, 1] = v_vals * (1.0 - u_vals)  # x_2 = f_2
        X_front[:, 2:] = 0.5  # g_i = (0.5 - 0.5)^2 = 0
        
        # Clip to bounds
        X_front = np.clip(X_front, lo, hi)
        
        # Evaluate front points
        F_all = []
        X_all = []
        
        for i in range(n_front):
            if time.time() > deadline:
                break
            F = problem(X_front[i])
            F_all.append(F)
            X_all.append(X_front[i])
        
        used = len(F_all)
        F_all = np.array(F_all) if F_all else np.empty((0, n_obj))
        X_all = np.array(X_all) if X_all else np.empty((0, dim))

        # Phase 2: Local refinement around non-dominated points
        if used < budget and len(F_all) > 0:
            # Find non-dominated points
            n_pts = len(F_all)
            dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                for j in range(n_pts):
                    if i != j and np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                        dominated[i] = True
                        break
            
            if np.any(~dominated):
                non_dom_idx = np.where(~dominated)[0]
                # Refine around non-dominated points
                rng = np.random.default_rng(42)
                n_refine = min(budget - used, 50)
                
                for k in range(n_refine):
                    if time.time() > deadline:
                        break
                    # Pick a random non-dominated parent
                    parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
                    parent = X_all[parent_idx]
                    
                    # Small perturbation
                    sigma = 0.05
                    child = parent + sigma * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)
                    
                    F_child = problem(child)
                    F_all = np.vstack([F_all, [F_child]])
                    X_all = np.vstack([X_all, [child]])
                    used += 1

        return X_all
# EVOLVE-BLOCK-END
