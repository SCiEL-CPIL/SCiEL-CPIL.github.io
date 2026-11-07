"""Rewritten optimizer for BBOB f14 (Different powers) using CMA-ES with restarts."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        rng = np.random.default_rng(12345)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Diverse Initialization
        # Sample a variety of points to find a good starting center
        init_points = [np.zeros(d)]
        for i in range(d):
            for val in [-4, 4]:
                pt = np.zeros(d)
                pt[i] = val
                init_points.append(pt)
        for _ in range(15):
            init_points.append(rng.uniform(-5, 5, d))

        for p in init_points:
            p_clipped = np.clip(p, -5, 5)
            fx = try_eval(p_clipped)
            if fx is None:
                break

        if calls_used >= max_calls or time.time() > deadline - 10:
            return best_x, best_f

        # Phase 2: CMA-ES with Restarts
        # CMA-ES Parameters
        lambda_ = 10
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))

        # Main restart loop
        while calls_used < max_calls:
            if time.time() > deadline - 5:
                break
            
            # Initialize CMA-ES state
            m = best_x.copy()
            sigma = 1.0
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            
            stagnation_counter = 0
            max_stagnation = 8 # Generations without improvement
            
            # Inner loop: CMA-ES generations
            while calls_used < max_calls and time.time() < deadline - 2:
                # Sample population
                Y = rng.multivariate_normal(np.zeros(d), C, size=lambda_)
                X = m + sigma * Y
                
                # Evaluate population
                F = np.full(lambda_, float('inf'))
                for i in range(lambda_):
                    fx = try_eval(np.clip(X[i], -5, 5))
                    if fx is None:
                        break
                    F[i] = fx
                
                if calls_used >= max_calls:
                    break
                    
                # Sort by fitness
                indices = np.argsort(F)
                Y_sorted = Y[indices]
                X_sorted = X[indices]
                F_sorted = F[indices]
                
                # Update global best
                if F_sorted[0] < best_f - 1e-10:
                    best_f = F_sorted[0]
                    best_x = X_sorted[0].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                if stagnation_counter >= max_stagnation:
                    break # Restart CMA-ES instance
                    
                # Update Mean
                m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
                
                # Update Step Size and Covariance
                y_w = (m_new - m) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
                
                hsig = (np.sqrt(1 - (1 - c_sigma)**2) / (1 - c_sigma)) * chi_d
                norm_ps = np.linalg.norm(ps)
                h = 1.0 if norm_ps / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(d+1)) * hsig else 0.0
                
                pc = (1 - c_c) * pc + h * c_c * y_w
                
                C = (1 + c_1 * (h * (2 - c_c) - 1) - c_mu) * C \
                    + c_1 * np.outer(pc, pc) \
                    + c_mu * np.sum(weights[:, None] * np.outer(Y_sorted[:mu] - y_w, Y_sorted[:mu] - y_w), axis=0)
                    
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Update Sigma
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_d - 1))
                
                m = m_new
                
                # Check for convergence
                if sigma < 1e-10:
                    break

            # After CMA-ES instance finishes (stagnation or convergence),
            # we loop back to restart with a fresh CMA-ES instance centered at the current best.
            # To encourage exploration if stuck, we could perturb m, but best_x is a good anchor.
