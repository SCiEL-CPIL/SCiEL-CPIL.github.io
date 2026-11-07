import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 315.0
        self.eval_count = 0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        dim = self.dim
        n_obj = self.n_obj
        
        self.X = []
        self.F = []
        
        def eval_real(x):
            if self.eval_count >= self.budget:
                raise RuntimeError("budget exceeded")
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != dim:
                x = np.resize(x, dim)
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).reshape(-1)
            self.eval_count += 1
            return f

        def add_eval(x, f):
            x = np.clip(np.asarray(x, dtype=float).reshape(-1), lo, hi)
            f = np.asarray(f, dtype=float).reshape(-1)
            self.X.append(x.copy())
            self.F.append(f.copy())

        def generate_weights(n_weights, n_obj):
            weights = []
            rng = np.random.default_rng(42)
            for _ in range(n_weights):
                w = rng.dirichlet(np.ones(n_obj))
                weights.append(w)
            return np.array(weights)

        n_weights = 10
        weights = generate_weights(n_weights, n_obj)
        
        z_star = np.full(n_obj, 1e6)
        
        # Initial random search
        n_init = 10
        rng = np.random.default_rng(0)
        for _ in range(n_init):
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(lo, hi)
            try:
                f = eval_real(x)
                add_eval(x, f)
                z_star = np.minimum(z_star, f)
            except Exception:
                break
        
        if len(self.X) == 0:
            return None

        best_x_for_weight = {}
        
        # Optimization loop
        for round_idx in range(2):
            for k, w in enumerate(weights):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                
                def obj_func(x):
                    try:
                        f = eval_real(x)
                        diffs = w * (f - z_star)
                        return np.max(diffs)
                    except Exception:
                        return 1e6

                if k in best_x_for_weight:
                    x0 = best_x_for_weight[k]
                else:
                    x0 = rng.uniform(lo, hi)
                
                try:
                    # maxiter=3, dim=4 -> ~15 evals
                    res = minimize(obj_func, x0, method='Nelder-Mead', 
                                   options={'maxiter': 3, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
                    
                    x_best = res.x
                    # Ensure we record the final best
                    f_best = eval_real(x_best)
                    add_eval(x_best, f_best)
                    best_x_for_weight[k] = x_best
                    z_star = np.minimum(z_star, f_best)
                    
                except Exception:
                    pass

        # Final refinement
        if self.eval_count < self.budget and time.time() < deadline:
            F_arr = np.array(self.F)
            mask = np.ones(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                if not mask[i]:
                    continue
                for j in range(len(F_arr)):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        mask[j] = False
            
            front_indices = np.where(mask)[0]
            
            for idx in front_indices:
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                x0 = self.X[idx]
                for _ in range(5):
                    if self.eval_count >= self.budget or time.time() > deadline:
                        break
                    x_new = x0 + rng.normal(0, 0.05, dim)
                    x_new = np.clip(x_new, lo, hi)
                    try:
                        f_new = eval_real(x_new)
                        add_eval(x_new, f_new)
                    except Exception:
                        break

        return None
