"""CMA-ES based optimizer for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation with budget and time constraints."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lambda_ = int(4 + 3 * np.log(dim))
        # Step size
        sigma = 2.0
        # Initial mean: start at center (0, 0, 0, 0, 0) or random
        # For Schaffers F7, the optimum is often near the center but shifted
        # Let's start with a random point in [-5, 5]^d, but maybe center is better?
        # Actually, the task says minimizer lies in [-4, 4]^5.
        # Let's initialize mean to 0.
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        # Eigenvalues of C (initialized to 1)
        ev = np.ones(dim)
        # Eigenvectors of C (initialized to identity)
        evx = np.eye(dim)

        # Step size control parameters
        mu = lambda_ // 2  # Number of weighted individuals
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_x = mean.copy()
        best_f = func(best_x)
        eval_count = 1

        # Pre-compute inverse of eigenvectors
        # We will update C = evx @ diag(ev) @ evx.T
        # And need to update ev and evx using rank-one and rank-mu updates

        def update_cma():
            nonlocal C, ev, evx, p_sigma, p_c, sigma, mean, c_sigma, c_c, c_1, c_mu, mu_eff, weights, mu, lambda_, dim
            
            # Check for termination or time limit
            if eval_count >= budget:
                return False
            if time.time() > deadline:
                return False

            # Generate population
            # y_i = z_i * sigma, z_i ~ N(0, I)
            # x_i = mean + sigma * B @ D @ z_i
            # Where B = evx, D = sqrt(ev)
            
            # We need to generate lambda_ samples
            # z = rng.standard_normal((lambda_, dim))
            # But we must be careful with RNG state. Let's use a single rng.
            
            # Actually, standard CMA-ES uses:
            # y_i = sigma * B @ D @ z_i
            # x_i = mean + y_i
            
            # Let's generate z
            z = rng.standard_normal((lambda_, dim))
            
            # Transform z
            # B @ D @ z.T -> (dim, lambda_)
            # z is (lambda_, dim)
            # B is (dim, dim), D is (dim,)
            # y = (B * D) @ z.T  -> (dim, lambda_)
            # y.T -> (lambda_, dim)
            
            # Efficient computation:
            # D is vector, B is matrix.
            # B @ (D[:, None] * z.T) ? No.
            # y_i = B @ (D * z_i)
            
            # Let's compute Y = (B * D) @ z.T
            # B * D scales columns of B by D
            BD = B * D  # (dim, dim)
            Y = BD @ z.T  # (dim, lambda_)
            Y = Y.T  # (lambda_, dim)
            
            X = mean[None, :] + Y  # (lambda_, dim)
            
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            eval_count += lambda_
            
            # Check time again
            if time.time() > deadline:
                return False

            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()

            # Sort by fitness
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            # Take top mu
            Y_mu = Y[:mu]
            
            # Weighted mean
            y_w = weights @ Y_mu  # (dim,)
            
            # Update step size
            # h_sigma = sqrt((1 - exp(-2 * min(1, chi_norm / sigma) * mu_eff)) / (dim * (1 - exp(-min(1, chi_norm / sigma) * mu_eff))))
            # Simplified: if ||y_w|| / (sigma * sqrt(mu_eff)) < 1.4 + 2/(dim+1), then h_sigma = 1
            # chi_norm = sqrt(dim)
            chi_norm = np.sqrt(dim)
            norm_yw = np.linalg.norm(y_w)
            if norm_yw > 0:
                ratio = norm_yw / (sigma * np.sqrt(mu_eff))
            else:
                ratio = 0
            h_sigma = 1.0 if ratio < 1.4 + 2.0 / (dim + 1) else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w / sigma
            
            # Update mean
            mean = mean + sigma * y_w
            
            # Update C
            # h_c = sqrt((1 - exp(-2 * min(1, ||p_c|| / sqrt(c_c) * mu_eff)) / (dim * (1 - exp(-min(1, ||p_c|| / sqrt(c_c) * mu_eff) * mu_eff))))
            # p_c = (1 - c_c) * p_c + h_c * sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # First update p_c
            # We need ||p_c|| before update? No, standard is:
            # p_c = (1-c_c)p_c + h_c * sqrt(c_c(2-c_c)mu_eff) * y_w
            # h_c depends on ||p_c|| / sqrt(c_c)
            
            norm_pc = np.linalg.norm(p_c)
            if norm_pc > 0:
                ratio_pc = norm_pc / np.sqrt(c_c)
            else:
                ratio_pc = 0
            h_c = 1.0 if ratio_pc < np.sqrt((2 + 2/(dim+1)) * c_c) else 0.0
            
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Rank-one update
            # C = (1 + c_1 * h_c * (2 - c_c) * mu_eff * (p_c p_c^T / dim - I) + c_mu * sum w_i (y_i y_i^T) / dim) * C
            # Wait, the standard update is:
            # C <- (1 + c_1 * (h_c * (2-c_c)*mu_eff * (p_c p_c^T / dim - I) + c_mu * sum w_i (y_i y_i^T))) C
            # Actually, it's usually applied to the eigenvalues/eigenvectors directly or via matrix multiplication.
            # Matrix multiplication is O(n^3). For dim=5, it's fine.
            
            # Let's do matrix update
            # delta_C = c_1 * (h_c * (2 - c_c) * mu_eff * (np.outer(p_c, p_c) / dim - np.eye(dim)) + c_mu * sum(weights[i] * np.outer(y_i, y_i) for i in range(mu)))
            
            # Compute rank-mu part
            # Y_mu is (mu, dim)
            # W is (mu,)
            # Rank_mu = sum(w_i * y_i y_i^T)
            # This is (Y_mu.T * w) @ Y_mu ? No.
            # Rank_mu = (Y_mu.T * weights) @ Y_mu  # (dim, dim)
            # Check: (Y_mu.T * w) is (dim, mu), Y_mu is (mu, dim). Product is (dim, dim).
            # Element (i,j) = sum_k Y_mu[k,i] * w[k] * Y_mu[k,j]. Correct.
            
            Rank_mu = (Y_mu.T * weights) @ Y_mu
            
            # Rank-one part
            Rank_one = np.outer(p_c, p_c) / dim - np.eye(dim)
            
            # Total delta
            delta_C = c_1 * (h_c * (2 - c_c) * mu_eff * Rank_one) + c_mu * Rank_mu
            
            # Update C
            C = C + delta_C
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_norm - 1))
            norm_ps = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / chi_norm - 1))
            
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update eigenvalues and eigenvectors
            # We need to decompose C again?
            # Standard CMA-ES maintains B and D (eigenvectors and sqrt eigenvalues).
            # After updating C, we need to recompute B and D.
            # This is O(n^3). For dim=5, it's fast.
            ev, evx = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            ev = np.clip(ev, 1e-12, None)
            D = np.sqrt(ev)
            
            # Update B and D in outer scope?
            # Since we are in a nested function, we need to nonlocal them or return them.
            # Let's return the updated state or use nonlocal.
            
            return True

        # We need B and D in the loop. Let's initialize them.
        B = np.eye(dim)
        D = np.ones(dim)
        
        # RNG
        rng = np.random.default_rng(0)
        
        # Main loop
        while True:
            # Check termination
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break
                
            # Generate population
            z = rng.standard_normal((lambda_, dim))
            
            # Transform
            BD = B * D
            Y = BD @ z.T
            Y = Y.T
            X = mean[None, :] + Y
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            eval_count += lambda_
            
            if time.time() > deadline:
                break

            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()

            # Sort
            order = np.argsort(F)
            Y = Y[order]
            F = F[order]
            
            Y_mu = Y[:mu]
            
            # Weighted mean
            y_w = weights @ Y_mu
            
            # Update step size
            chi_norm = np.sqrt(dim)
            norm_yw = np.linalg.norm(y_w)
            if norm_yw > 0:
                ratio = norm_yw / (sigma * np.sqrt(mu_eff))
            else:
                ratio = 0
            h_sigma = 1.0 if ratio < 1.4 + 2.0 / (dim + 1) else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w / sigma
            
            # Update mean
            mean = mean + sigma * y_w
            
            # Update p_c
            norm_pc = np.linalg.norm(p_c)
            if norm_pc > 0:
                ratio_pc = norm_pc / np.sqrt(c_c)
            else:
                ratio_pc = 0
            h_c = 1.0 if ratio_pc < np.sqrt((2 + 2/(dim+1)) * c_c) else 0.0
            
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            # Update C
            Rank_mu = (Y_mu.T * weights) @ Y_mu
            Rank_one = np.outer(p_c, p_c) / dim - np.eye(dim)
            
            delta_C = c_1 * (h_c * (2 - c_c) * mu_eff * Rank_one) + c_mu * Rank_mu
            
            C = C + delta_C
            C = (C + C.T) / 2
            
            # Update sigma
            norm_ps = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / chi_norm - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update eigenvalues and eigenvectors
            ev, evx = np.linalg.eigh(C)
            ev = np.clip(ev, 1e-12, None)
            D = np.sqrt(ev)
            B = evx
            
            # Termination criteria
            if sigma < 1e-12:
                break
            if np.max(ev) / np.min(ev) > 1e14:
                break
            if np.linalg.norm(y_w) / sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
