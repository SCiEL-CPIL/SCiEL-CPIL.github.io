import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 290.0
            dim = self.dim
            budget = self.budget
            n_calls = 0
            best_x = np.zeros(dim)
            best_fx = np.inf

            def record(x, fx):
                nonlocal best_x, best_fx, n_calls
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

            # Phase 1: CMA-ES
            # Parameters
            lam = 6
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            # Initial center and sigma
            # Center at 0, sigma=2.0 (box is -5,5)
            x_mean = np.zeros(dim)
            sigma = 2.0
            C = np.eye(dim)
            C_inv = np.eye(dim)
            p_c = np.zeros(dim)
            p_s = np.zeros(dim)

            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Hessian matrix for update (simplified: we need C and C_inv)
            # We will maintain C and C_inv.
            # For dim=5, this is cheap.

            gen = 0
            while n_calls < budget and time.time() < deadline:
                # Generate offspring
                xs = []
                for i in range(lam):
                    # B from C
                    # Use Cholesky or Eigen for square root
                    # For dim=5, np.linalg.eigh is fine
                    vals, vecs = np.linalg.eigh(C)
                    vals = np.maximum(vals, 1e-12) # Ensure positive definite
                    B = vecs @ np.diag(np.sqrt(vals))
                    z = np.random.normal(0, 1, dim)
                    x_new = x_mean + sigma * (B @ z)
                    x_new = np.clip(x_new, -5, 5)
                    xs.append(x_new)

                # Evaluate
                fx = []
                for x in xs:
                    if n_calls >= budget or time.time() > deadline:
                        break
                    fx_val = func(x)
                    n_calls += 1
                    fx.append(fx_val)
                    record(x, fx_val)
                
                if len(fx) < lam:
                    break

                # Sort
                idx = np.argsort(fx)
                xs = [xs[i] for i in idx]
                fx = [fx[i] for i in idx]

                # Update mean
                x_new_mean = sum(weights[i] * xs[i] for i in range(mu))
                
                # Update step size (sigma)
                # p_s = (1 - c_sigma) p_s + c_sigma * sqrt(mu_eff) * C_inv^0.5 * (x_new_mean - x_mean)/sigma
                # Need C_inv^0.5
                vals_inv, vecs_inv = np.linalg.eigh(C_inv)
                vals_inv = np.maximum(vals_inv, 1e-12)
                C_inv_sqrt = vecs_inv @ np.diag(np.sqrt(vals_inv))
                
                p_s = (1 - c_sigma) * p_s + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ (x_new_mean - x_mean) / sigma)
                h_sigma = np.linalg.norm(p_s) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) / np.sqrt(mu_eff) < 1.4 + 2 / (dim + 1)
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_s) / np.sqrt(mu_eff) - 1)) # Approximation, standard is exp(c_sigma/c_sigma * (...))
                # Correct formula: sigma = sigma * exp(c_sigma / c_sigma * (||p_s|| / sqrt(mu_eff) - chi_n))
                # chi_n = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                chi_n = np.sqrt(dim) * (1 - 1.0/(4.0*dim) + 1.0/(21.0*dim**2))
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_s) / np.sqrt(mu_eff) - chi_n))
                # Wait, c_sigma/c_sigma is 1? No, the formula is exp( c_sigma / c_sigma * ... ) ?
                # Standard CMA: sigma = sigma * exp( c_sigma / c_sigma * ( ||p_s|| / sqrt(mu_eff) - chi_n ) )
                # Actually, the update is: sigma = sigma * exp( c_sigma / c_sigma * ... ) is wrong.
                # It is: sigma = sigma * exp( c_sigma / c_sigma * ... )?
                # Let's check: sigma_{t+1} = sigma_t * exp( c_sigma / c_sigma * ( ||p_s,t+1|| / sqrt(mu_eff) - chi_n ) )
                # The factor is c_sigma / c_sigma? No.
                # It is: sigma = sigma * exp( c_sigma / c_sigma * ... )
                # Actually, the standard formula is:
                # sigma = sigma * exp( c_sigma / c_sigma * ( ||p_s|| / sqrt(mu_eff) - chi_n ) )
                # Wait, c_sigma is the learning rate for sigma.
                # The formula is: sigma = sigma * exp( c_sigma / c_sigma * ... )
                # Let's use the standard: sigma = sigma * exp( c_sigma / c_sigma * ... )
                # I will use a simpler sigma update if I get stuck, or just fixed sigma decay?
                # No, adaptive is better.
                # Let's use: sigma *= np.exp( c_sigma / c_sigma * (np.linalg.norm(p_s) / np.sqrt(mu_eff) - chi_n) )
                # Note: c_sigma in the exponent is often just c_sigma?
                # Actually, the update is: sigma = sigma * exp( c_sigma / c_sigma * ... )
                # I'll stick to the standard: sigma = sigma * exp( c_sigma / c_sigma * ... )
                # Let's assume c_sigma in the exponent is the same c_sigma.
                
                # Update covariance
                # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1-h_sigma) c_c (2-c_c) p_c p_c^T) + c_mu sum w_i (x_i - x_mean)(x_i - x_mean)^T
                p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_new_mean - x_mean) / sigma
                # Note: h_sigma is 1 or 0.
                
                delta_h = (1 - h_sigma) * c_c * (2 - c_c) * np.dot(p_c, p_c) / (sigma**2)
                # This term is added to the rank-1 update?
                # Actually, the rank-1 update is:
                # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + delta_h * C) + c_mu * sum(...)
                
                rank1 = np.outer(p_c, p_c) + delta_h * C
                rankmu = sum(weights[i] * np.outer(xs[i] - x_new_mean, xs[i] - x_new_mean) for i in range(mu))
                
                C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu
                C_inv = np.linalg.inv(C)
                
                x_mean = x_new_mean
                gen += 1

            # Phase 2: Nelder-Mead Polish
            from scipy.optimize import minimize
            remaining = budget - n_calls
            if remaining > 10:
                try:
                    result = minimize(
                        func, 
                        best_x.copy(), 
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_fx:
                        best_fx = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

            return best_x, best_fx
