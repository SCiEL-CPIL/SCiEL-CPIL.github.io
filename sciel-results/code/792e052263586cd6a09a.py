"""Model-and-optimize for RE37 multi-objective task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    """Sobolev Initialization + Nelder-Mead Local Search:
    1) Sobolev sequence for high-quality space-filling DOE.
    2) Evaluate initial points.
    3) For a set of weight vectors, run Nelder-Mead local search from best starting points.
    4) This avoids the high computational cost and stochasticity of DE/GP fitting.
    """

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

        rng = np.random.default_rng(0)

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

        # --- 1) Sobolev Sequence DOE ---
        # Use Sobolev for better space filling than Halton in low dims
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        # 2^k points. 2^5 = 32 points is a good start for 4 dims.
        n_sobolev = 32
        sobolev_pts = sampler.random(n_sobolev)
        
        # Add corners and center
        specials = [lo, hi, (lo+hi)/2.0]
        # Add a few random points to ensure diversity if Sobolev is too structured
        random_pts = rng.uniform(lo, hi, size=(8, dim))
        
        initial_grid = np.vstack([specials, sobolev_pts, random_pts])
        rng.shuffle(initial_grid)

        model_X = []
        model_Y = []
        
        def add_point(x, f):
            model_X.append(np.asarray(x, dtype=float).copy())
            model_Y.append(np.asarray(f, dtype=float).copy())

        # Evaluate initial grid
        for x in initial_grid:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            try:
                f = eval_real(x)
                add_point(x, f)
            except Exception:
                break

        if len(model_X) == 0:
            return None

        Xtr = np.vstack(model_X)
        Ytr = np.vstack(model_Y)

        # --- 2) Define Weight Vectors ---
        # We want to cover the Pareto front. 
        # Use a grid of weights on the simplex.
        weights = []
        # Extreme points
        for k in range(n_obj):
            w = np.zeros(n_obj); w[k] = 1.0; weights.append(w)
        # Uniform weights
        w = np.ones(n_obj) / n_obj
        weights.append(w)
        # Additional random weights to fill gaps
        for _ in range(5):
            w = rng.dirichlet(np.ones(n_obj))
            weights.append(w)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        # --- 3) Local Search with Nelder-Mead ---
        # For each weight, find the best starting point from Xtr
        # and run Nelder-Mead to minimize the weighted sum.
        
        # We have used ~40 evaluations. Remaining: ~160.
        # We have ~8 weights. We can afford ~20 evals per local search?
        # Nelder-Mead in 4D takes ~5 evals/step. 20 evals = 4 steps. Might be too few.
        # Let's use maxfev=30 per local search.
        
        for w in weights:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            
            # Select best starting point for this weight
            # Score each existing point by weighted sum
            scores = Ytr @ w
            best_idx = np.argmin(scores)
            x_start = Xtr[best_idx].copy()
            
            # Define objective function for Nelder-Mead
            def obj_func(x):
                if self.eval_count >= self.budget:
                    return 1e9
                x = np.clip(x, lo, hi)
                f = eval_real(x)
                return np.dot(f, w)
            
            # Run Nelder-Mead
            # maxiter and maxfev to control budget
            # Simplex size: 0.1 * (hi-lo)
            x0 = x_start
            try:
                res = minimize(obj_func, x0, method='Nelder-Mead', 
                               options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 50, 'maxfev': 20})
                # res.fun is the weighted sum, but we need the actual f vector
                # We must evaluate the final x to get the vector and record it properly
                # Note: minimize might have evaluated x0 again or not. 
                # To be safe, we evaluate res.x explicitly.
                x_opt = res.x
                f_opt = eval_real(x_opt)
                add_point(x_opt, f_opt)
            except Exception:
                pass

        # --- 4) Remaining Budget: Random Local Search ---
        # If we have budget left, do more local searches from random points
        while self.eval_count < self.budget:
            if time.time() > deadline:
                break
            x_start = lo + (hi - lo) * rng.random(dim)
            w = rng.dirichlet(np.ones(n_obj))
            
            def obj_func(x):
                if self.eval_count >= self.budget:
                    return 1e9
                x = np.clip(x, lo, hi)
                f = eval_real(x)
                return np.dot(f, w)
            
            try:
                res = minimize(obj_func, x_start, method='Nelder-Mead', 
                               options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 30, 'maxfev': 15})
                x_opt = res.x
                f_opt = eval_real(x_opt)
                add_point(x_opt, f_opt)
            except Exception:
                break

        return None
# EVOLVE-BLOCK-END
