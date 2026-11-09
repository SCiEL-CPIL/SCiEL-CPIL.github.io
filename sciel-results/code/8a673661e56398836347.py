"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s margin within 300s limit

        n = self.dim
        x_mean = np.zeros(n)
        sigma = 1.0
        
        # CMA-ES parameters
        lambda_ = int(4 + 3 * np.log(n))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Initial evaluation
        x = np.clip(x_mean, -5, 5)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        
        calls_made = 1
        old_x_mean = x_mean.copy()

        while calls_made + lambda_ <= self.budget:
            if time.time() > deadline:
                break
            
            # Sample offspring
            z = np.random.randn(lambda_, n)
            
            # Apply Broyden-Fisher decomposition of C
            eig_vals, eig_vecs = np.linalg.eigh(C)
            eig_vals = np.maximum(eig_vals, 0)
            sqrt_eig_vals = np.sqrt(eig_vals)
            
            # y = x_mean + sigma * (z @ (eig_vecs * sqrt_eig_vals))
            y = x_mean + sigma * (z @ (eig_vecs * sqrt_eig_vals))
            y = np.clip(y, -5, 5)
            
            # Evaluate
            f_y = np.array([func(yi) for yi in y])
            calls_made += lambda_
            
            # Update best
            min_idx = np.argmin(f_y)
            if f_y[min_idx] < best_f:
                best_f = f_y[min_idx]
                best_x = y[min_idx].copy()
                
            # Sort offspring
            idx = np.argsort(f_y)
            y_sorted = y[idx]
            
            # Update mean
            new_x_mean = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # Update step size control
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * D_inv * B * (x_mean_new - x_mean_old) / sigma
            # D_inv * B = (B.T * D_inv).T ? No.
            # y = B @ D @ z. So z = D_inv @ B_inv @ y.
            # The update rule uses the normalized difference.
            # h_sigma = 1 if ||ps|| / sqrt(1 - (1-c_sigma)^{2(g+1)}) < (1.4 + 2/(n+1)) * chi_n else 0
            # Simplified: check if ||ps|| is too large
            
            # Calculate D_inv @ B @ (diff)
            # B is eig_vecs (columns are eigenvectors)
            # D is diag(sqrt_eig_vals)
            # So B @ D is the matrix that transforms z to x-space step.
            # We need (B @ D)^{-1} @ (new_x_mean - old_x_mean)
            # (B @ D)^{-1} = D_inv @ B.T (since B is orthogonal)
            
            diff = (new_x_mean - old_x_mean) / sigma
            # Apply inverse transform
            # proj = B.T @ diff
            proj = eig_vecs.T @ diff
            # Apply D_inv
            inv_diff = proj / np.maximum(sqrt_eig_vals, 1e-10)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_diff
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update C
            # pc = (1 - c_c) * pc + h_c * sqrt(c_c * (2 - c_c) * mu_eff) * (new_x_mean - old_x_mean) / sigma
            # h_c is usually 1 if ||pc||/sqrt(1-(1-c_c)^(2(g+1))) < (1.4 + 2/(n+1)) * chi_n
            # We approximate h_c = 1 for simplicity, as it's rarely 0 in early stages
            
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
            
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c)**2) * np.eye(n)) 
            #    + c_mu * sum(weights_i * (y_i - old_x_mean)/sigma @ (y_i - old_x_mean)/sigma.T)
            
            delta_c = (1 - (1 - c_c) ** 2) * np.eye(n)
            
            # Rank-1 update
            c_update = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_c)
            
            # Rank-mu update
            # We need the normalized differences for the top mu offspring
            y_diffs = (y_sorted[:mu] - old_x_mean) / sigma
            rank_mu = np.sum(weights[:, None, None] * y_diffs[:, :, None] * y_diffs[:, None, :], axis=0)
            
            C = c_update + c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            # eigvals = np.linalg.eigvalsh(C)
            # if min(eigvals) < 1e-10:
            #     C += (1e-10 - min(eigvals)) * np.eye(n)
            
            old_x_mean = new_x_mean
            x_mean = new_x_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
