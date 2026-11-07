import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 280 # Safety margin

    def _check_time(self):
        return time.time() > self.deadline

    def _evaluate(self, x, problem):
        if self.eval_count >= self.budget:
            return None
        self.eval_count += 1
        try:
            return problem(x)
        except Exception:
            return None

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        X_list = []
        F_list = []
        
        def add_point(x, f):
            if f is None or len(f) != self.n_obj or not np.all(np.isfinite(f)):
                return
            x_clipped = np.clip(x, lo, hi)
            # Simple duplicate check
            for i in range(len(X_list)):
                if np.allclose(X_list[i], x_clipped, atol=1e-6):
                    return
            X_list.append(x_clipped)
            F_list.append(f)

        # Phase 1: Initialization
        # Use a mix of grid and random to ensure coverage
        # 16 corners
        for i in range(2**self.dim):
            x = np.array([(hi[d] if (i >> d) & 1 else lo[d]) for d in range(self.dim)])
            f = self._evaluate(x, problem)
            if f is not None:
                add_point(x, f)
        
        # Center
        x_center = (lo + hi) / 2
        f_center = self._evaluate(x_center, problem)
        if f_center is not None:
            add_point(x_center, f_center)

        # Random points to fill gaps
        # Budget remaining: 200 - 17 = 183
        # Let's add 63 random points to make 80 total
        n_rand = 63
        for _ in range(n_rand):
            if self._check_time() or self.eval_count >= self.budget:
                break
            x = np.random.uniform(lo, hi)
            f = self._evaluate(x, problem)
            if f is not None:
                add_point(x, f)

        if not F_list:
             # Fallback
             x0 = (lo + hi) / 2
             f0 = self._evaluate(x0, problem)
             if f0 is not None: add_point(x0, f0)
             return X_list

        # Phase 2: Refinement
        # Weight vectors
        weight_vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1.0/3, 1.0/3, 1.0/3],
            [0.7, 0.15, 0.15],
            [0.15, 0.7, 0.15],
            [0.15, 0.15, 0.7],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.4, 0.4, 0.2],
            [0.4, 0.2, 0.4],
            [0.2, 0.4, 0.4],
        ]

        # Sort weight vectors to prioritize extremes? 
        # Actually, for hypervolume, we want the "knees" of the front too.
        # The list above is decent.

        # Loop through rounds
        max_rounds = 10
        for r in range(max_rounds):
            if self.eval_count >= self.budget or self._check_time():
                break
            
            F = np.array(F_list)
            X = np.array(X_list)
            
            # Determine how many evaluations we can spend per optimization run
            # We want to finish all weight vectors in this round
            # Let's say we do 1 optimization per weight vector per round
            # If budget is tight, reduce maxfev
            
            evals_left = self.budget - self.eval_count
            if evals_left <= 0: break
            
            # If we have very few evals left, do a single high-precision run on the best point
            if evals_left < 20:
                # Pick the best point (lowest sum)
                sums = np.sum(F, axis=1)
                best_idx = np.argmin(sums)
                x_start = X[best_idx].copy()
                
                def obj_sum(x):
                    f = self._evaluate(x, problem)
                    if f is None: return 1e6
                    return np.sum(f)
                
                try:
                    minimize(obj_sum, x_start, method='Nelder-Mead', bounds=list(zip(lo, hi)), options={'maxfev': evals_left, 'xatol': 1e-8, 'fatol': 1e-8})
                except: pass
                break

            # Normal round
            # Shuffle weight vectors to ensure diversity across seeds/runs?
            # No, keep deterministic order for reproducibility within a seed, 
            # but the random init provides the variation.
            
            # Select starting points
            # For each weight vector, find the point in the current archive that is best for that weight
            # This is a greedy approach
            
            for w in weight_vectors:
                if self.eval_count >= self.budget or self._check_time():
                    break
                
                w = np.array(w)
                
                # Find best start point for this weight
                # scalarize current F
                scalarized_F = F @ w
                best_start_idx = np.argmin(scalarized_F)
                x_start = X[best_start_idx].copy()
                
                best_f = None
                best_x = x_start.copy()
                
                def objective(x, w=w):
                    nonlocal best_f, best_x
                    f = self._evaluate(x, problem)
                    if f is None:
                        return 1e6
                    val = np.dot(w, f)
                    if best_f is None or val < np.dot(w, best_f):
                        best_f = f
                        best_x = x.copy()
                    return val
                
                # Set maxfev based on remaining budget
                # We have ~16 weight vectors. If we have 100 evals left, ~6 per run.
                # If we have 50 evals left, ~3 per run.
                maxfev = max(5, (self.budget - self.eval_count) // len(weight_vectors))
                
                try:
                    minimize(
                        objective, 
                        x_start, 
                        method='Nelder-Mead', 
                        bounds=list(zip(lo, hi)),
                        options={'maxfev': maxfev, 'xatol': 1e-7, 'fatol': 1e-7}
                    )
                    if best_f is not None:
                        add_point(best_x, best_f)
                except Exception:
                    pass
        
        return X_list
