"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        rng = np.random.default_rng(42)
        
        # 1. Initialization
        # Start near the center, as Schwefel's global minimum is at 0
        x_mean = np.zeros(self.dim)
        sigma = 2.0  # Initial step size, covering a good portion of the box
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(np.floor(3 * np.log(self.dim)))
        # Weights for recombination (logarithmic weights)
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1)) - np.log(mu)
        w /= w.sum()
        mu_eff = 1 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # State variables
        C = np.eye(self.dim)
        C_sqrt_inv = np.eye(self.dim)  # Inverse square root of C
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        
        # Precompute for efficiency
        # We need to generate lam samples from N(0, C)
        # x = x_mean + sigma * C_sqrt_inv @ z, where z ~ N(0, I)
        # Actually, standard CMA-ES samples x_i = x_mean + sigma * sum(w_j * (y_j - y_{j+1})) ... no, that's for the update.
        # Sampling: x_i = x_mean + sigma * C_sqrt_inv @ z_i
        
        # Initial evaluation
        f_vals = [func(x_mean)]
        if f_vals[0] < best_f:
            best_f = f_vals[0]
            best_x = x_mean.copy()
            
        calls = 1
        
        # Main loop
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Sample population
            # z_i ~ N(0, I)
            Z = rng.standard_normal((lam, self.dim))
            # Transform to N(0, C)
            # X = x_mean + sigma * C_sqrt_inv @ Z.T
            # Note: C_sqrt_inv is updated in the update step.
            # For efficiency, we can precompute C_sqrt_inv @ Z.T if C changes slowly, but it changes every step.
            # Let's just do the matrix multiplication.
            Y = C_sqrt_inv @ Z.T
            X = x_mean + sigma * Y.T
            X = np.clip(X, -5, 5)
            
            # 2. Evaluate
            F = []
            for i in range(lam):
                f_x = func(X[i])
                F.append(f_x)
                calls += 1
                if f_x < best_f:
                    best_f = f_x
                    best_x = X[i].copy()
            
            F = np.array(F)
            
            # 3. Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # 4. Update mean
            x_mean_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # 5. Update sigma path
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (x_mean_new - x_mean) / sigma
            diff_mean = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ diff_mean)
            
            # 6. Update C path
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff_mean
            
            # 7. Update Covariance Matrix
            # C = (1 + c_1 * (h_sigma - 1) + c_mu - 1) * C + c_1 * (h_sigma - 1 + 2) * p_c * p_c.T + c_mu * sum(w_i * y_i * y_i.T)
            # where y_i = (x_i - x_mean) / sigma
            
            # h_sigma: heuristic to avoid rank loss
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2(gen+1)}) < (1.4 + 2/(dim+1)) * sqrt((mu_eff+1) - 2)
            # We don't track generations explicitly in a simple loop, but we can approximate or just use a fixed threshold logic.
            # For simplicity and robustness, we can check the norm.
            ps_norm = np.linalg.norm(p_sigma)
            # Theoretical expected norm of p_sigma
            # Let's use a simplified h_sigma check
            h_sigma = 1.0
            # This is an approximation, usually requires tracking history.
            # For a short run, this might be less critical, but let's include a basic check.
            # If ps_norm is too large, h_sigma = 0 to dampen adaptation.
            # Threshold: (1.4 + 2/(dim+1)) * sqrt((mu_eff+1) - 2)
            thresh = (1.4 + 2 / (self.dim + 1)) * np.sqrt((mu_eff + 1) - 2)
            if ps_norm > thresh:
                h_sigma = 0.0
            
            # y_i vectors
            Y_vecs = (X_sorted[:mu] - x_mean) / sigma  # shape (mu, dim)
            
            # Update C
            # Term 1: Decay
            decay = 1 + c_1 * (h_sigma - 1) + c_mu - 1
            C = decay * C
            
            # Term 2: p_c * p_c.T
            if h_sigma == 1.0:
                C += c_1 * (h_sigma - 1 + 2) * np.outer(p_c, p_c)
            
            # Term 3: sum(w_i * y_i * y_i.T)
            # C += c_mu * sum(w_i * np.outer(Y_vecs[i], Y_vecs[i]))
            # Efficiently: C += c_mu * (Y_vecs.T * w) @ Y_vecs
            # Y_vecs.T is (dim, mu), w is (mu,)
            # (Y_vecs.T * w) scales columns of Y_vecs.T by w
            # Then @ Y_vecs gives (dim, dim)
            C += c_mu * ((Y_vecs.T * w) @ Y_vecs)
            
            # Update x_mean and sigma
            x_mean = x_mean_new
            
            # Sigma update
            # sigma *= exp(c_sigma / d_sigma * (||p_sigma|| / E||N(0,I)|| - 1))
            # E||N(0,I)|| = sqrt(dim) * (1 - 1/(4dim) + 1/(32dim^2))
            e_norm = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (32 * self.dim**2))
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / e_norm - 1))
            
            # Ensure sigma is within reasonable bounds
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 100:
                sigma = 100
            
            # 8. Update C_sqrt_inv
            # We need the inverse square root of C.
            # C is symmetric positive definite.
            # Eigendecomposition: C = V * D * V.T
            # C_sqrt_inv = V * D^(-0.5) * V.T
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues due to numerical errors
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                # Fallback to identity if decomposition fails (should not happen often)
                C_sqrt_inv = np.eye(self.dim)
                C = np.eye(self.dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
