"""
Optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
This is a robust, state-of-the-art black-box optimizer for continuous,
non-separable functions.
"""

import numpy as np


class Optimizer:
    """CMA-ES based optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        # Standard formula: lambda = 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + 3*1.6 = 8.8 -> 8 or 9. Let's use 10 for better exploration.
        lam = 10
        
        # Initial mean: Center of the box
        x_mean = np.zeros(dim)
        
        # Initial step size
        # The box is [-5, 5]. The minimizer is in [-4, 4].
        # A step size of 2.0 allows covering the space in a few steps.
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Adaptation rates
        mu = lam // 2  # Number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2.0 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(c_mu, 2 * c_1)
        
        # Cumulation paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluations
        # Generate first population
        # We need to keep track of the best point and value
        best_x = None
        best_f = np.inf
        
        # Pre-compute Cholesky factor of C (initially I)
        C_sqrt = np.eye(dim)
        
        # Main loop
        evals_used = 0
        while evals_used < budget:
            # Sample population
            ys = np.array([rng.standard_normal(dim) for _ in range(lam)])
            # Transform to x
            xs = np.array([x_mean + sigma * (C_sqrt @ y) for y in ys])
            
            # Clip to bounds
            xs = np.clip(xs, -5, 5)
            
            # Evaluate
            fs = np.array([func(x) for x in xs])
            evals_used += lam
            
            # Update best
            idx_best_pop = np.argmin(fs)
            if fs[idx_best_pop] < best_f:
                best_f = fs[idx_best_pop]
                best_x = xs[idx_best_pop]
                
            # Check if budget is exhausted
            if evals_used >= budget:
                break
                
            # Sort by fitness
            order = np.argsort(fs)
            y_selected = ys[order[:mu]]
            f_selected = fs[order[:mu]]
            x_selected = xs[order[:mu]]
            
            # Weighted mean of selected points
            x_new_mean = np.sum(weights[:, None] * x_selected, axis=0)
            
            # Update step size
            # Need inverse of C_sqrt
            # For efficiency, we can compute C_sqrt_inv via Cholesky of C
            # But C changes. Let's compute the inverse square root properly.
            # Actually, standard CMA-ES uses the transformation matrix B and D.
            # B @ B.T = C. D = 1/sqrt(diag(B.T @ B)).
            # Here we maintain C and use Cholesky.
            
            # Recompute Cholesky of C
            # C = B @ B.T
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular
                B = np.eye(dim)
                
            # Compute D = 1/sqrt(diag(B.T @ B))
            BtB = B.T @ B
            D = 1.0 / np.sqrt(np.diag(BtB))
            
            # Inverse of C_sqrt is B.T @ diag(D)
            # C_sqrt_inv = B.T @ np.diag(D)
            
            # Compute the step in the standard normal space
            # y_bar = B.T @ D @ (x_new_mean - x_mean) / sigma
            # Note: x_new_mean - x_mean = sigma * C_sqrt @ y_bar
            # So y_bar = C_sqrt_inv @ (x_new_mean - x_mean) / sigma
            
            # However, we sampled x = x_mean + sigma * C_sqrt @ y
            # So y = C_sqrt_inv @ (x - x_mean) / sigma
            
            # Let's compute the mean of the selected y's
            # y_selected are the original standard normal samples
            y_bar = np.sum(weights[:, None] * y_selected, axis=0)
            
            # Update p_sigma
            # h_sigma = (||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2/(gen+1)})) / chi_n
            # chi_n = sqrt(n) * (1 - 1/(4n) - 1/(25n^2))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) - 1.0 / (25 * dim ** 2))
            
            # Approximate h_sigma using the condition that ||y_bar|| is small
            # A more robust check:
            norm_y_bar = np.linalg.norm(y_bar)
            # Threshold for h_sigma
            # If the norm of the mean of the selected y's is small, it's likely a step size adaptation
            # rather than a directional move.
            # Standard heuristic:
            h_sigma = norm_y_bar < (1.4 + 2.0 / (dim + 1)) * chi_n
            h_sigma = float(h_sigma)
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_bar
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update p_c
            h_c = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (budget - evals_used))) < (1.4 + 2.0 / (dim + 1)) * chi_n
            h_c = float(h_c)
            
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt_inv @ (x_new_mean - x_mean) / sigma if False else y_bar)
            # Wait, the update for p_c uses the difference in the original space transformed back.
            # p_c = (1 - c_c) * p_c + h_c * sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ D @ (x_new_mean - x_mean) / sigma)
            # Which is equivalent to using y_bar if the transformation was exact.
            # Let's stick to the y_bar formulation for p_c as well, as it is standard.
            # Actually, p_c is in the "C-space".
            # Let's use the standard formula:
            # p_c = (1 - c_c) * p_c + h_c * sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ D @ (x_new_mean - x_mean) / sigma)
            # Since (x_new_mean - x_mean) / sigma = C_sqrt @ y_bar
            # B.T @ D @ C_sqrt @ y_bar
            # Note that C_sqrt = B. So B.T @ D @ B @ y_bar.
            # This is not simply y_bar.
            # Let's compute the term explicitly.
            delta_mean = (x_new_mean - x_mean) / sigma
            # delta_mean is in the "B-space" coordinates? No, it's the vector in R^n.
            # We need to map it to the standard normal space for the cumulation?
            # No, p_c is a vector in R^n that accumulates the step directions.
            # The update is:
            # p_c = (1 - c_c) * p_c + h_c * sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ D @ delta_mean)
            
            term_c = B.T @ (D * (B @ delta_mean)) # This is B.T @ D @ B @ delta_mean? No.
            # D is diagonal. B @ delta_mean is a vector.
            # Let's just compute:
            # v = B @ delta_mean
            # w = D * v
            # term = B.T @ w
            
            v = B @ delta_mean
            w = D * v
            term_c = B.T @ w
            
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * term_c
            
            # Update C
            # C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C
            #    + c_1 * (p_c @ p_c.T - C)
            #    + c_mu * sum(weights_i * (y_i - y_bar) @ (y_i - y_bar).T)
            
            # Note: The sum term uses the y vectors (standard normal).
            # delta_y = y_selected - y_bar
            # sum_term = sum(weights_i * delta_y_i @ delta_y_i.T)
            
            delta_ys = y_selected - y_bar[None, :]
            # Weighted sum of outer products
            sum_term = np.einsum('i,ij,ik->jk', weights, delta_ys, delta_ys)
            
            C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C \
                + c_1 * (p_c @ p_c.T - C) \
                + c_mu * sum_term
                
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            
            # Update mean
            x_mean = x_new_mean
            
            # Update C_sqrt for next iteration
            # We will recompute Cholesky at the start of the next loop
            # But we need it for the next sampling.
            # The loop starts with sampling, so we need C_sqrt ready.
            # Let's compute it here to be safe, or at the start of the loop.
            # It's better to compute it at the start of the loop where we sample.
            # But we already sampled.
            # Let's just let the next iteration handle it.
            # Wait, the sampling happens at the top of the loop.
            # So the C_sqrt used for sampling in iteration t is computed from C at end of iteration t-1.
            # This is correct.
            
            # Optimization: If the population has collapsed, break
            if sigma < 1e-15:
                break

        return best_x, best_f

    # Helper for RNG
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(42)

    def __call__(self, func):
        # ... (same as above but using self.rng)
        dim = self.dim
        budget = self.budget
        rng = self.rng
        
        lam = 10
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2.0 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        c_mu = min(c_mu, 2 * c_1)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = None
        best_f = np.inf
        
        evals_used = 0
        while evals_used < budget:
            ys = np.array([rng.standard_normal(dim) for _ in range(lam)])
            
            # Compute Cholesky
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                B = np.eye(dim)
            D = 1.0 / np.sqrt(np.diag(B.T @ B))
            C_sqrt = B
            
            xs = np.array([x_mean + sigma * (C_sqrt @ y) for y in ys])
            xs = np.clip(xs, -5, 5)
            
            fs = np.array([func(x) for x in xs])
            evals_used += lam
            
            idx_best_pop = np.argmin(fs)
            if fs[idx_best_pop] < best_f:
                best_f = fs[idx_best_pop]
                best_x = xs[idx_best_pop]
                
            if evals_used >= budget:
                break
                
            order = np.argsort(fs)
            y_selected = ys[order[:mu]]
            x_selected = xs[order[:mu]]
            
            x_new_mean = np.sum(weights[:, None] * x_selected, axis=0)
            
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) - 1.0 / (25 * dim ** 2))
            y_bar = np.sum(weights[:, None] * y_selected, axis=0)
            
            h_sigma = float(np.linalg.norm(y_bar) < (1.4 + 2.0 / (dim + 1)) * chi_n)
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_bar
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            h_c = float(np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (budget - evals_used)) < (1.4 + 2.0 / (dim + 1)) * chi_n)
            
            delta_mean = (x_new_mean - x_mean) / sigma
            v = B @ delta_mean
            w = D * v
            term_c = B.T @ w
            
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * term_c
            
            delta_ys = y_selected - y_bar[None, :]
            sum_term = np.einsum('i,ij,ik->jk', weights, delta_ys, delta_ys)
            
            C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C \
                + c_1 * (p_c @ p_c.T - C) \
                + c_mu * sum_term
            C = (C + C.T) / 2.0
            
            x_mean = x_new_mean
            
            if sigma < 1e-15:
                break

        return best_x, best_f
