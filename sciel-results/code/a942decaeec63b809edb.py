import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, leave 10s for overhead/cleanup
        deadline = start_time + 320.0
        calls = 0
        max_calls = self.budget
        dim = self.dim
        
        # CMA-ES Parameters
        # Population size: 4 + 3*ln(n) is standard, for n=5 this is ~9. 
        # We use a slightly larger population for better exploration in the first phase.
        lam = 9 
        # Weights for selection
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        
        # Initial state
        # Start at origin, but sigma is large enough to cover the box [-5, 5]
        # Box width is 10, so sigma=2.0 covers ~4 standard deviations.
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        inv_sqrt_C = np.eye(dim)
        
        # Precompute inverse square root of C (initially I, so inv_sqrt is I)
        # We will update this periodically
        
        best_x = None
        best_f = np.inf
        
        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= max_calls:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time Limit")
            
            calls += 1
            x = np.asarray(x, dtype=float)
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        try:
            while calls < max_calls and time.time() < deadline:
                # Generate population
                # Sample from N(mean, sigma^2 * C)
                # We need to handle the square root of C. 
                # For efficiency, we update inv_sqrt_C every few generations.
                # Initially C=I, so inv_sqrt_C=I.
                
                # Optimization: Only recompute inv_sqrt_C every 5 generations to save time
                # But for simplicity and robustness in a short budget, let's do it every time
                # if dim is small (5 is very small).
                
                # Compute square root of C via eigen decomposition
                # C is symmetric positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues due to numerical errors
                eigvals = np.clip(eigvals, 1e-12, None)
                inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                # Generate lam samples
                # z ~ N(0, I)
                zs = np.random.randn(lam, dim)
                # Transform to N(0, C)
                # x = mean + sigma * (inv_sqrt_C @ z.T).T
                # Note: inv_sqrt_C @ z.T gives N(0, C) if z is N(0, I)?
                # Let z ~ N(0, I). y = A z. Cov(y) = A A^T.
                # We want Cov = C. So A A^T = C.
                # If A = C^{1/2}, then A A^T = C.
                # My inv_sqrt_C is C^{-1/2}.
                # So I should use C^{1/2} = (inv_sqrt_C)^{-1}.
                # Let's just compute sqrt_C directly.
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                
                # Generate points
                # zs is (lam, dim). We need sqrt_C @ z.T for each z?
                # No, z is row vector. y = z @ sqrt_C^T = z @ sqrt_C (since symmetric).
                # Let's verify: Var(z @ S) = S Var(z) S^T = S I S^T = S S^T = C. Correct.
                ys = zs @ sqrt_C
                xs = mean + sigma * ys
                
                # Evaluate
                fs = np.zeros(lam)
                for i in range(lam):
                    try:
                        fs[i] = wrapper(xs[i])
                    except (ValueError, TimeoutError):
                        # If we hit limit, break out
                        break
                    except Exception:
                        fs[i] = np.inf
                
                if calls >= max_calls or time.time() > deadline:
                    break
                
                # Sort by fitness
                order = np.argsort(fs)
                xs_sorted = xs[order]
                fs_sorted = fs[order]
                
                # Selection: top mu
                selected_xs = xs_sorted[:mu]
                selected_fs = fs_sorted[:mu]
                
                # Update mean
                new_mean = np.sum(selected_xs * weights[:, np.newaxis], axis=0)
                
                # Update step size control vectors
                # pc = (1 - c_c) pc + c_c * sqrt(mu_eff) * sigma * C^{-1/2} (x_new - x_old)
                # Note: The standard update uses the weighted sum of the selected points relative to the OLD mean,
                # transformed by C^{-1/2}.
                # diff = new_mean - mean
                # pc = (1 - c_c) * pc + c_c * sqrt(mu_eff) * sigma * (inv_sqrt_C @ diff)
                
                diff = new_mean - mean
                pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * sigma * (inv_sqrt_C @ diff)
                
                # ps = (1 - c_sigma) ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * sqrt_C @ (sigma * diff)
                # Note: The standard formula is:
                # ps = (1 - c_sigma) ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^{1/2} @ (sigma * (x_{new} - x_{old}))
                # Wait, the term is usually scaled by sigma?
                # Hansen's book: ps = (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff) * C^{1/2} * (sigma * (x_{w:1} - x_{old}))
                # Actually, the vector is the difference in the search space, scaled by sigma?
                # Let's check standard implementation.
                # Usually: ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (sqrt_C @ (sigma * diff))
                
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (sqrt_C @ (sigma * diff))
                
                # Update sigma
                # sigma = sigma * exp( c_sigma / d_sigma * ( ||ps|| / chi_n - 1 ) )
                chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update Covariance Matrix C
                # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - h_pc) c_c (2 - c_c) C) + c_mu sum(w_i (y_i - y_mean)(y_i - y_mean)^T)
                # Where y_i are the selected points in the normalized space (C^{-1/2} (x_i - mean_old))
                
                # Calculate h_pc
                h_pc = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(2 * (calls + 1)))) / (chi_n * np.sqrt(1 - (1 - c_c)**(2 * (calls + 1))))
                # Simplified h_pc check:
                # h_pc = 1 if ||pc|| / sqrt(1-(1-c_c)^(2t)) < (1.4 + 2/(dim+1)) chi_n else 0
                # Let's use the simpler threshold check
                threshold = (1.4 + 2.0 / (dim + 1.0)) * chi_n
                pc_norm = np.linalg.norm(pc)
                # Approximation for the denominator term is 1 for large t, but let's be safe
                # For small t, 1-(1-c_c)^(2t) is small, so ratio is large?
                # Actually, standard code uses:
                # hpc = (np.linalg.norm(pc) / np.sqrt(1.0 - (1.0 - cc)**(2.0 * t))) / (chi_n * np.sqrt(1.0 - (1.0 - cc)**(2.0 * t)))
                # This is effectively ||pc|| / (chi_n * (1 - (1-cc)^(2t)))
                # Let's just use the threshold on ||pc|| relative to expected norm
                # Expected norm of pc is roughly chi_n.
                # If ||pc|| > threshold, h_pc = 0, else 1.
                h_pc = 1.0 if pc_norm < threshold else 0.0
                
                # Term 1: (1 - c_1 - c_mu) * C
                C_new = (1 - c_1 - c_mu) * C
                
                # Term 2: c_1 * (pc pc^T + (1 - h_pc) * c_c * (2 - c_c) * C)
                C_new += c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
                
                # Term 3: c_mu * sum(w_i * (y_i - y_mean)(y_i - y_mean)^T)
                # y_i = C^{-1/2} (x_i - mean_old)
                # y_mean = C^{-1/2} (new_mean - mean_old) = C^{-1/2} diff
                y_mean = inv_sqrt_C @ diff
                for i in range(mu):
                    y_i = inv_sqrt_C @ (selected_xs[i] - mean)
                    dy = y_i - y_mean
                    C_new += c_mu * weights[i] * np.outer(dy, dy)
                
                C = C_new
                
                # Ensure symmetry and positive definiteness (numerical stability)
                C = (C + C.T) / 2.0
                # Eigenvalue clipping
                eigvals_C, eigvecs_C = np.linalg.eigh(C)
                eigvals_C = np.clip(eigvals_C, 1e-12, None)
                C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T
                
                mean = new_mean
                
        except (ValueError, TimeoutError):
            pass
        except Exception:
            # Fallback if CMA-ES crashes
            if best_x is None:
                # Try a simple random search fallback
                rng = np.random.default_rng(42)
                for _ in range(10):
                    if calls >= max_calls: break
                    x = rng.uniform(-5, 5, dim)
                    try:
                        wrapper(x)
                    except:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            # Ensure at least one call is made if none were
            if calls == 0:
                try:
                    wrapper(best_x)
                except:
                    pass

        return best_x, best_f
