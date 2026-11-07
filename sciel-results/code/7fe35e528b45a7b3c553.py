"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Black-box continuous optimizer for BBOB f13 (Sharp ridge).
    Uses a simplified CMA-ES variant with adaptive step size and covariance.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin

        rng = np.random.default_rng(42)

        # --- Phase 1: Initial Sampling ---
        n_init = 20
        xs = rng.uniform(-5, 5, (n_init, self.dim))
        fvals = np.array([func(x) for x in xs])
        
        best_idx = np.argmin(fvals)
        x_best = xs[best_idx].copy()
        f_best = fvals[best_idx]
        
        # Estimate initial covariance from best half
        k = n_init // 2
        best_indices = np.argsort(fvals)[:k]
        x_samples = xs[best_indices]
        mean_x = np.mean(x_samples, axis=0)
        centered = x_samples - mean_x
        C = np.cov(centered, rowvar=False)
        C += 1e-6 * np.eye(self.dim)  # Regularize
        
        # Initial step size
        sigma = np.linalg.norm(centered, axis=1).mean() * 0.5
        sigma = np.clip(sigma, 0.1, 2.0)
        
        calls_made = n_init
        lambda_pop = 10
        mu = 5  # Top 5 for update
        
        # Weights
        weights = np.log(lambda_pop + 1) - np.log(np.arange(1, lambda_pop + 1))
        weights /= weights.sum()
        w = weights[:mu]
        w = w / w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # CMA-ES Parameters
        c1 = 2 / ((self.dim + 1.3) ** 2 + 10)
        cmu = 2 / ((self.dim + 1) ** 2 + 5)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 2) / (self.dim + mu_eff - 2)) + c_sigma
        
        # Evolution Paths
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        # Cholesky of C
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(self.dim)
            
        # Inverse C
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            C_inv = np.eye(self.dim)
            
        # Sqrt C_inv
        try:
            sqrt_C_inv = np.linalg.cholesky(C_inv)
        except np.linalg.LinAlgError:
            sqrt_C_inv = np.eye(self.dim)

        # --- Phase 2: CMA-ES Loop ---
        while calls_made < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            batch_size = min(lambda_pop, self.budget - calls_made)
            
            # Generate population
            Z = rng.standard_normal((batch_size, self.dim))
            # Y = x_best + sigma * L @ Z.T
            # Z is (batch, dim), L is (dim, dim)
            # We want (L @ z_i) for each z_i.
            # So Y_i = x + sigma * (L @ Z[i].T).T = x + sigma * (Z[i] @ L.T)
            Y = x_best + sigma * (Z @ L.T)
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            f_Y = np.array([func(y) for y in Y])
            calls_made += batch_size
            
            # Sort
            order = np.argsort(f_Y)
            Y_sorted = Y[order]
            f_Y_sorted = f_Y[order]
            
            # Update best
            if f_Y_sorted[0] < f_best:
                f_best = f_Y_sorted[0]
                x_best = Y_sorted[0].copy()

            # Update CMA-ES
            # y_w = weighted mean of top mu
            y_w = np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            
            # z_w = weighted mean of top mu Z (in original order)
            # We need to map sorted indices back to Z
            # Z_sorted = Z[order]
            Z_sorted = Z[order]
            z_w = np.sum(w[:, None] * Z_sorted[:mu], axis=0)
            
            # Update pc
            pc = (1 - c1) * pc + c1 * (y_w - x_best) / sigma
            
            # Update ps
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * z_w
            
            # Calculate hsig (heuristic for covariance update)
            norm_ps = np.linalg.norm(ps)
            norm_pc = np.linalg.norm(pc)
            # Approximate expected norm of ps
            # E[||ps||] = sqrt(1 - (1-c_sigma)^2) * sqrt(mu_eff) * sqrt(1 - 1/mu_eff) * ...
            # Simplified check:
            hsig = 1.0 if norm_ps < (1.4 + 2 / (self.dim + 1)) * np.sqrt((self.dim + 1) / (self.dim - 1)) * np.sqrt(1 - (1 - c_sigma) ** 2) * np.sqrt(mu_eff) else 0.0
            
            # Update C
            # Delta C = sum(w_i * (y_i - x) @ (y_i - x).T)
            # We use the top mu points
            # Y_top = Y_sorted[:mu]
            # Y_top_centered = Y_top - x_best
            # Delta C = sum(w_i * (Y_top_centered[i] @ Y_top_centered[i].T) / sigma^2)
            
            # Note: The update is on the covariance matrix C.
            # The term (y_i - x) / sigma is the normalized step.
            # Let's compute Delta C
            Y_top = Y_sorted[:mu]
            Y_top_norm = (Y_top - x_best) / sigma
            # Delta C = sum(w_i * outer(Y_top_norm[i], Y_top_norm[i]))
            Delta_C = np.zeros((self.dim, self.dim))
            for i in range(mu):
                Delta_C += w[i] * np.outer(Y_top_norm[i], Y_top_norm[i])
            
            # C update
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (hsig - 1) * c1 * (1 - c1) * C) + cmu * Delta_C
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||ps|| / E[||N(0,1)||] - 1))
            # E[||N(0,1)||] in dim d is approx sqrt(d)
            # More precisely, chi distribution mean.
            # For dim=5, mean is approx 2.07
            chi_mean = np.sqrt(self.dim) * (1 - 0.25 / self.dim)  # Approx
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / chi_mean - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Recompute L and C_inv periodically or every step?
            # Dim=5 is small, every step is fine.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, add regularization
                C += 1e-6 * np.eye(self.dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(self.dim)
                    C = np.eye(self.dim)
            
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(self.dim)
                
            try:
                sqrt_C_inv = np.linalg.cholesky(C_inv)
            except np.linalg.LinAlgError:
                sqrt_C_inv = np.eye(self.dim)

        return x_best, f_best
# EVOLVE-BLOCK-END
