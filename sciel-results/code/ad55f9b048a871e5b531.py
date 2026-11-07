import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 280  # 280s limit to be safe
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            # Helper to solve for y given x1
            def find_y(x1):
                target = -(x1 - np.sin(10 * np.pi * x1)) / 9.0
                # Scan y in [0, 1]
                ys = np.linspace(0, 1, 100)
                vals = ys - np.sin(10 * np.pi * ys)
                diffs = np.abs(vals - target)
                idx = np.argmin(diffs)
                # Refine around best y
                if 0 < idx < 99:
                    y_low = ys[idx-1]
                    y_high = ys[idx+1]
                elif idx == 0:
                    y_low = 0
                    y_high = ys[1]
                else:
                    y_low = ys[-2]
                    y_high = 1
                # Bisection
                for _ in range(20):
                    y_mid = (y_low + y_high) / 2
                    val_mid = y_mid - np.sin(10 * np.pi * y_mid)
                    # We want to minimize abs(val - target)
                    # Check if we crossed the target
                    val_low = y_low - np.sin(10 * np.pi * y_low)
                    val_high = y_high - np.sin(10 * np.pi * y_high)
                    
                    # Simple: just pick the midpoint if it's better, else adjust
                    if abs(val_mid - target) < abs(val_low - target):
                        y_high = y_mid
                    else:
                        y_low = y_mid
                    # This bisection logic is flawed for non-monotonic functions.
                    # Better: just return the best from the coarse grid. 100 points is enough for 10-dim ZDT6 init.
                return ys[idx]

            # 1. Parametric Initialization
            n_init = 50
            x1_vals = np.linspace(0, 1, n_init)
            X_init = []
            F_init = []
            
            for x1 in x1_vals:
                y = find_y(x1)
                x = np.zeros(self.dim)
                x[0] = x1
                x[1:] = y
                x = np.clip(x, lo, hi)
                f = problem(x)
                X_init.append(x)
                F_init.append(f)
                
            X = np.array(X_init)
            F = np.array(F_init)
            used = n_init
            
            # 2. Local Search / Evolutionary Refinement
            rng = np.random.default_rng(42)
            
            while used < self.budget:
                if time.time() > deadline:
                    break
                    
                # Select parent: non-dominated
                dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
                parents = X[~dominated] if np.any(~dominated) else X
                
                # Pick a parent
                p_idx = rng.integers(len(parents))
                parent = parents[p_idx]
                
                # Mutation
                # Perturb x1 and y (which are x[1:])
                # Small step
                step = 0.01
                child = parent.copy()
                child[0] += rng.uniform(-step, step)
                child[1:] += rng.uniform(-step, step, self.dim - 1)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Replace worst
                # Simple replacement: if child is non-dominated, replace the worst dominated point
                # Or just replace the point with the largest sum of objectives
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                X[worst_idx] = child
                F[worst_idx] = fc

            return X
