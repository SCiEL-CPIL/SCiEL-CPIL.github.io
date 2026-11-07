import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_fx = np.inf

        def evaluate(x):
            nonlocal best_x, best_fx, n_calls
            x = np.clip(x, -5, 5)
            fx = func(x)
            n_calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse global search to find a good basin
        # f7 (Step Ellipsoidal) has a flat region. We need to get into the "basin" first.
        # Random sampling is robust here.
        n_coarse = 30
        for _ in range(n_coarse):
            if n_calls >= budget or time.time() > deadline - 10:
                break
            x = np.random.uniform(-5, 5, dim)
            evaluate(x)

        # Phase 2: CMA-ES for global convergence
        # CMA-ES is robust to non-smoothness and can handle the step if sigma is managed well.
        # We use a smaller population to save budget for polishing.
        lam = 6
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        dsigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c1
        
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        DC = np.linalg.cholesky(C)
        chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        x = best_x.copy()
        # Start with a larger sigma to explore the step region
        sigma = 1.5 

        stagnation_counter = 0
        best_fx_prev = best_fx
        cma_iterations = 0
        max_cma_iter = 40 # Limit CMA-ES iterations to leave room for polishing

        while n_calls < budget and cma_iterations < max_cma_iter:
            if time.time() > deadline - 10:
                break
                
            # Check stagnation
            if best_fx == best_fx_prev:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_fx_prev = best_fx
                
            # If stuck, try a local random restart or larger steps
            if stagnation_counter > 10:
                # Randomize direction or increase sigma
                if stagnation_counter % 5 == 0:
                    sigma *= 1.5
                    C = np.eye(dim)
                    DC = np.linalg.cholesky(C)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    x = best_x + np.random.randn(dim) * sigma
                    x = np.clip(x, -5, 5)
                if stagnation_counter > 20:
                    # Full restart from a random point near best
                    x = best_x + np.random.randn(dim) * 2.0
                    x = np.clip(x, -5, 5)
                    sigma = 1.0
                    C = np.eye(dim)
                    DC = np.linalg.cholesky(C)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    stagnation_counter = 0

            # Generate population
            Z = np.random.randn(lam, dim)
            Y = Z @ DC.T
            
            F = np.empty(lam)
            for i in range(lam):
                if n_calls >= budget:
                    break
                xi = x + sigma * Y[i]
                xi = np.clip(xi, -5, 5)
                F[i] = evaluate(xi)
            
            if n_calls >= budget:
                break
                
            indices = np.argsort(F)
            Z_sorted = Z[indices]
            Y_sorted = Y[indices]
            
            z_bar = np.dot(weights, Z_sorted[:mu])
            mean_step = DC @ z_bar
            x = x + sigma * mean_step
            x = np.clip(x, -5, 5)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_bar
            sigma = sigma * np.exp(cs / dsigma * (np.linalg.norm(ps) / chi_N - 1))
            sigma = np.clip(sigma, 1e-6, 5.0)
            
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * mean_step
            
            Y_sel = Y_sorted[:mu]
            M = np.zeros((dim, dim))
            for i in range(mu):
                M += weights[i] * np.outer(Y_sel[i], Y_sel[i])
                
            h_sig = 1.0 if np.linalg.norm(ps) / chi_N < (1.4 + 2/(dim+1)) else 0.0
            
            C = (1 - c1 - cmu * h_sig) * C + c1 * np.outer(pc, pc) + cmu * M
            C = (C + C.T) / 2
            
            try:
                DC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 0.01 * np.eye(dim)
                try:
                    DC = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    DC = np.eye(dim)
                    C = np.eye(dim)
            
            cma_iterations += 1

        # Phase 3: Intensive Polishing with Nelder-Mead and Coordinate Descent
        # The step function has a flat bottom. Nelder-Mead might struggle to pinpoint the exact center
        # if it's on the flat part, but it can refine the boundary.
        # However, f7 is defined such that the minimum is at the center of the ellipsoid.
        # The "step" means f(x) = 0.5 + sum(...) if norm < 1, else sum(...).
        # Actually, BBOB f7 is: 
        # if ||x|| < 1: 0.5 + 0.5 * sum(x_i^2 / alpha_i^2)
        # else: sum(x_i^2 / alpha_i^2)
        # The minimum is at 0.5 (before shift/rotation/offset).
        
        remaining = budget - n_calls
        if remaining > 20:
            # Use Nelder-Mead with tight tolerances
            try:
                # Use a smaller initial simplex to stay local
                x0 = best_x.copy()
                # Initialize simplex manually for better control? 
                # scipy handles it, but let's ensure it doesn't wander too far.
                res = minimize(func, x0, method='Nelder-Mead', 
                               options={'maxfev': remaining, 'maxiter': remaining, 
                                        'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Final check: if we still have budget, do a few random perturbations around best_x
        # to see if we can jump out of a local flat region (though unlikely for f7 once in basin)
        remaining = budget - n_calls
        if remaining > 5:
            for _ in range(min(10, remaining)):
                if n_calls >= budget:
                    break
                # Small perturbation
                x = best_x + np.random.randn(dim) * 0.1
                x = np.clip(x, -5, 5)
                evaluate(x)

        return best_x, best_fx
