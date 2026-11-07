"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard: 330s wall-clock limit, return safely before.
        t_start = time.time()
        deadline = t_start + 300.0

        # CMA-ES Parameters
        # Standard CMA-ES uses population size lambda = 4 + floor(3 log dim)
        # For dim=5, lambda = 4 + 3*1.6 = 8.8 -> 8. Let's use 10 for a bit more diversity.
        lambda_pop = 10
        mu = lambda_pop // 2  # 5

        # Weights: standard CMA-ES weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)

        # Step size sigma
        # Start with a relatively large sigma to explore the box [-5, 5]^5
        # Box diagonal is sqrt(5)*10 ~ 22.3. Center is 0.
        # A sigma of 2.0 covers a reasonable chunk of the space initially.
        sigma = 2.0

        # Starting point: center of the box
        x_mean = np.zeros(dim)
        
        # Covariance matrix C
        C = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation parameters
        # c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        # d_sigma = 1 + 2 * max(0, sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        # c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        # c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2)**2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2)**2 + mu_eff)

        # Clamp probabilities for sigma update
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim**2))

        # Pre-compute Cholesky factor
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(dim)

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        # Main loop
        while calls < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Sample population
            # y_i ~ N(0, C)
            # We generate standard normals and transform
            ys = rng.standard_normal((lambda_pop, dim))
            # Transform: z = C_sqrt @ y
            # Efficiently: Z = ys @ C_sqrt.T
            Z = ys @ C_sqrt.T
            
            # x_i = x_mean + sigma * z_i
            X = x_mean[None, :] + sigma * Z
            
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate
            Fs = np.array([func(x_i) for x_i in X])
            calls += lambda_pop
            
            # Sort by fitness
            order = np.argsort(Fs)
            X_sorted = X[order]
            Fs_sorted = Fs[order]
            
            # Update best
            if Fs_sorted[0] < best_f:
                best_f = Fs_sorted[0]
                best_x = X_sorted[0].copy()

            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:, None] * X_sorted[:mu], axis=0)
            
            # Update evolution paths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_mean - x_mean_old)
            if sigma > 1e-12:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_mean - x_mean_old)
            
            # h_sigma condition
            h_sigma = 1.0
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // lambda_pop))) < (1.4 + 2 / (dim + 1)) * chi_n:
                h_sigma = 1.0
            else:
                h_sigma = 0.0

            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * C^{-1/2} (x_mean - x_mean_old) / sigma
            # C^{-1/2} is inverse of Cholesky factor
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
            
            diff_mean = x_mean - x_mean_old
            # Note: C^{-1/2} (x_mean - x_mean_old) / sigma
            # Since C_sqrt is lower triangular such that C = C_sqrt @ C_sqrt.T
            # C^{-1/2} is the inverse of the upper triangular factor? 
            # Standard CMA-ES uses C^{-1/2} which is the inverse of the Cholesky factor used for sampling if we used upper triangular.
            # Here we used C_sqrt @ C_sqrt.T. So C^{-1/2} is inv(C_sqrt).T? No.
            # If C = L L^T, then C^{-1/2} = (L^{-1})^T L^{-1} is not quite right for the path update formula which usually assumes a specific square root.
            # Let's stick to the standard formulation: update p_c using C^{-1/2}.
            # In many implementations, C^{-1/2} is computed as the inverse of the Cholesky factor of C.
            # Let's use: v = C_inv_sqrt @ diff_mean / sigma
            v = C_inv_sqrt @ (diff_mean / sigma)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v

            # Update Covariance Matrix
            # C = (1 - c_1 - c_mu) C + c_1 (p_c p_c^T + (1 - h_sigma) c_c (2 - c_c) C) + c_mu sum w_i (y_i y_i^T)
            
            # Calculate y_i (the standardized steps)
            # y_i = (x_i - x_mean_old) / sigma
            Y = (X_sorted[:mu] - x_mean_old[None, :]) / sigma
            
            # Outer products
            Y_outer = (Y.T * w) * Y  # shape (dim, dim)
            
            # Update C
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            C += c_mu * Y_outer
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Clamp sigma
            if sigma > 1e4:
                sigma = 1e4
            elif sigma < 1e-12:
                # Stagnation or convergence
                # If sigma is too small, we might be stuck.
                # For Rastrigin, we might be in a local min.
                # Simple restart: if no improvement in last 50 calls, reset.
                # But we don't track history easily here. 
                # Let's just break if sigma is extremely small, as further progress is unlikely.
                break

            # Update Cholesky factor
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, regularize
                C = C + 1e-6 * np.eye(dim)
                C = (C + C.T) / 2.0
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(dim)
                    break

        return best_x, best_f

# Note: rng is not defined in the class scope, need to initialize it in __call__ or __init__
# Let's fix the rng usage.
# The previous code used rng.standard_normal but rng was not defined.
# I will add rng initialization in __call__.

# Corrected class with rng initialization
class Optimizer:
    """CMA-ES for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Deadline guard: 330s wall-clock limit, return safely before.
        t_start = time.time()
        deadline = t_start + 300.0

        # CMA-ES Parameters
        lambda_pop = 10
        mu = lambda_pop // 2  # 5

        # Weights: standard CMA-ES weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)

        # Step size sigma
        sigma = 2.0

        # Starting point: center of the box
        x_mean = np.zeros(dim)
        
        # Covariance matrix C
        C = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2)**2 + mu_eff)

        # Clamp probabilities for sigma update
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim**2))

        # Pre-compute Cholesky factor
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(dim)

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        # Main loop
        while calls < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Sample population
            ys = rng.standard_normal((lambda_pop, dim))
            Z = ys @ C_sqrt.T
            X = x_mean[None, :] + sigma * Z
            X = np.clip(X, -5, 5)
            
            # Evaluate
            Fs = np.array([func(x_i) for x_i in X])
            calls += lambda_pop
            
            # Sort by fitness
            order = np.argsort(Fs)
            X_sorted = X[order]
            Fs_sorted = Fs[order]
            
            # Update best
            if Fs_sorted[0] < best_f:
                best_f = Fs_sorted[0]
                best_x = X_sorted[0].copy()

            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:, None] * X_sorted[:mu], axis=0)
            
            # Update evolution paths
            if sigma > 1e-12:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_mean - x_mean_old)
            
            # h_sigma condition
            h_sigma = 1.0
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Approximate generations
            gen = calls // lambda_pop
            if gen > 0:
                denom = np.sqrt(1 - (1 - c_sigma)**(2 * gen))
                if denom > 1e-12:
                    if norm_p_sigma / denom < (1.4 + 2 / (dim + 1)) * chi_n:
                        h_sigma = 1.0
                    else:
                        h_sigma = 0.0

            # p_c update
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
            
            diff_mean = x_mean - x_mean_old
            v = C_inv_sqrt @ (diff_mean / sigma)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v

            # Update Covariance Matrix
            Y = (X_sorted[:mu] - x_mean_old[None, :]) / sigma
            Y_outer = (Y.T * w) * Y
            
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            C += c_mu * Y_outer
            
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Clamp sigma
            if sigma > 1e4:
                sigma = 1e4
            elif sigma < 1e-12:
                break

            # Update Cholesky factor
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-6 * np.eye(dim)
                C = (C + C.T) / 2.0
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(dim)
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
