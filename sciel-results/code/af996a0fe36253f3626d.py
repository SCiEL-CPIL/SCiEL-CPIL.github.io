import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        d = self.dim
        budget = self.budget
        
        # Track calls strictly
        calls = [0]
        
        def evaluate(x):
            if calls[0] >= budget:
                return None
            calls[0] += 1
            val = func(x)
            return val

        rng = np.random.default_rng()
        
        # Phase 1: Random sampling to get a good starting point
        n_random = min(30, budget)
        best_x = None
        best_f = float('inf')
        
        for _ in range(n_random):
            if time.time() > deadline or calls[0] >= budget:
                break
            x = rng.uniform(-5, 5, d)
            f = evaluate(x)
            if f is not None and f < best_f:
                best_f = f
                best_x = x.copy()
                
        if best_x is None:
            best_x = np.zeros(d)
            f = evaluate(best_x)
            if f is not None:
                best_f = f
            else:
                best_f = 0.0

        # Phase 2: CMA-ES
        # CMA-ES is robust for non-linear, non-convex functions and handles
        # anisotropic landscapes (like Bent Cigar) well by adapting the covariance matrix.
        
        # Determine initial sigma
        # The search space is [-5, 5]^d.
        # A reasonable initial step size is about 10-20% of the box width.
        # Box width is 10. So sigma ~ 1.0 or 2.0.
        # Let's start with 1.0.
        
        x0 = best_x.copy()
        sigma0 = 1.0
        
        # Max iterations for CMA-ES
        # Each iteration uses ~ (4 + floor(3*ln(n))) evaluations on average, but can vary.
        # With budget 500 and dim 5, we have plenty of room for many iterations.
        # maxiter should be high enough to allow convergence.
        
        maxiter = 1000
        
        # We need to wrap the function to respect the budget and deadline.
        # scipy.optimize.minimize with method='CMA' (via cma library) is not standard in all scipy versions.
        # Let's check if we can use scipy.optimize.minimize with method='Powell' or 'Nelder-Mead' as a fallback?
        # CMA-ES is superior for this task. Is 'cma' available?
        # The prompt says "Python standard library, NumPy and SciPy are available."
        # It does NOT explicitly say 'cma' package is available.
        # However, scipy.optimize.minimize does NOT have a 'CMA' method built-in in standard recent versions (it used to have experimental ones or require the cma package).
        # Wait, scipy.optimize.minimize has methods: Nelder-Mead, Powell, CG, BFGS, Newton-CG, L-BFGS-B, TNC, COBYLA, SLSQP, trust-*, dogleg, trust-*, trust-constr.
        # None of these are CMA-ES.
        
        # So I must implement a simple CMA-ES or use a different robust method.
        # Implementing CMA-ES from scratch is complex but doable for a simple case.
        # Alternatively, use a (1+1)-ES with adaptive step size?
        # Or use Powell's method which is derivative-free and robust?
        
        # Let's try to implement a simplified CMA-ES or use a robust derivative-free method available in SciPy.
        # Powell's method is often good for box-constrained problems and doesn't require gradients.
        # However, CMA-ES is specifically designed for this type of BBOB problem.
        
        # Let's try a custom implementation of a simple (1+1) Evolution Strategy with step size adaptation,
        # or better yet, a small population CMA-ES.
        
        # Given the constraints, let's implement a basic CMA-ES.
        
        class CMAES:
            def __init__(self, x0, sigma0, bounds, evaluate_func, dim, max_evals, deadline):
                self.x = x0
                self.sigma = sigma0
                self.dim = dim
                self.bounds = bounds
                self.evaluate = evaluate_func
                self.max_evals = max_evals
                self.deadline = deadline
                
                self.C = np.eye(dim)
                self.pc = np.zeros(dim)
                self.ps = np.zeros(dim)
                self.chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(20*dim**2))
                
                self.cc = (4 + dim/3) / (dim + 4 + 2*dim/dim)
                self.cs = (mu + 1/2) / (dim + mu + 5)
                self.c1 = 2 / ((dim+1.3)**2 + mu)
                self.cmua = min(1 - self.c1, 2 / ((dim+2)**2 + mu))
                
                self.mu = dim // 2
                self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                self.weights /= np.sum(self.weights)
                self.mueff = 1 / np.sum(self.weights**2)
                
                # Update constants
                self.cc = (4 + self.dim/3) / (self.dim + 4 + 2*self.dim/self.dim)
                self.cs = (self.mueff + 1/2) / (self.dim + self.mueff + 5)
                self.c1 = 2 / ((self.dim+1.3)**2 + self.mueff)
                self.cmua = min(1 - self.c1, 2 / ((self.dim+2)**2 + self.mueff))

            def _step(self):
                n = self.dim
                lam = 2 * self.mu
                y = np.zeros((lam, n))
                x_pop = np.zeros((lam, n))
                f_pop = np.zeros(lam)
                
                for i in range(lam):
                    z = rng.standard_normal(n)
                    # Cholesky decomposition for C
                    L = np.linalg.cholesky(self.C)
                    x_i = self.x + self.sigma * L @ z
                    # Clip to bounds
                    x_i = np.clip(x_i, self.bounds[0], self.bounds[1])
                    
                    f_i = self.evaluate(x_i)
                    if f_i is None:
                        # Budget or time exceeded
                        return False
                    f_pop[i] = f_i
                    x_pop[i] = x_i
                    y[i] = (x_i - self.x) / self.sigma
                
                # Sort by fitness
                idx = np.argsort(f_pop)
                y_sorted = y[idx]
                x_sorted = x_pop[idx]
                f_sorted = f_pop[idx]
                
                # Update mean
                w = self.weights
                self.x = self.x + self.sigma * np.sum(w[:, None] * y_sorted[:self.mu], axis=0)
                
                # Update path evolution strategies
                y_sigma = (self.x - self.x_prev) / (self.sigma * self.chi_n) if hasattr(self, 'x_prev') else np.zeros(n)
                # Wait, standard CMA-ES updates ps and pc.
                # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (1/sigma) * (x_new - x_old)
                # We need to store x_old.
                
                # This implementation is getting complicated and error-prone without testing.
                # Let's fall back to a simpler, robust method: **Nelder-Mead** with restarts, or **Powell**.
                # Powell is derivative-free and handles bounds (though scipy's Powell doesn't natively handle bounds well, we can clip).
                # Actually, let's use **Nelder-Mead** with a large number of iterations and careful initialization.
                # Or better, let's use **COBYLA** which handles inequalities (bounds).
                
                return True

        # Let's try SciPy's Powell method first, as it's robust and doesn't need gradients.
        # We will clip the input to ensure bounds are respected.
        
        def bounded_func(x):
            x_clipped = np.clip(x, -5.0, 5.0)
            return evaluate(x_clipped)
        
        # Powell method
        try:
            from scipy.optimize import minimize
            res = minimize(
                bounded_func,
                best_x,
                method='Powell',
                options={'maxiter': 5000, 'maxfev': budget - calls[0]}
            )
            if res.x is not None:
                best_x = np.clip(res.x, -5.0, 5.0)
                best_f = bounded_func(best_x)
                if best_f is None:
                    best_f = float('inf')
        except Exception:
            pass

        # If Powell didn't use all budget, try L-BFGS-B from the best point found
        if calls[0] < budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                res2 = minimize(
                    bounded_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0) for _ in range(d)],
                    options={'maxiter': 1000, 'maxfev': budget - calls[0]}
                )
                if res2.x is not None:
                    best_x = np.clip(res2.x, -5.0, 5.0)
                    best_f = bounded_func(best_x)
                    if best_f is None:
                        best_f = float('inf')
            except Exception:
                pass

        return best_x, best_f
