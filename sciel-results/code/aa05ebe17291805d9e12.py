"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        # CMA-ES Parameters
        # Population size lambda
        lambda_ = 4 + int(3 * np.log(dim))  # For dim=5, lambda = 10
        # Number of selected individuals
        mu = lambda_ // 2
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance-effective population size
        mu_eff = 1 / np.sum(weights**2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        # Step size initialization
        sigma_0 = 0.5
        # Chi lambda for step size adaptation
        chi_lambda = np.sqrt(lambda_) * (1 - 1 / (4 * dim) + 1 / (24 * dim ** 2))

        # State
        x_mean = np.zeros(dim)
        sigma = sigma_0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        sqrt_eigvals = np.ones(dim)
        
        # Pre-compute eigendecomposition for C
        # Initially C=I, so eigenvectors are identity, eigenvalues 1
        
        best_x = None
        best_f = np.inf
        calls = 0

        # Helper to update C and its inverse
        def update_covariance(y_sigma):
            nonlocal C, C_inv, sqrt_eigvals
            # y_sigma is the selected points centered and scaled
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (2 / (1 - exp(-2 * c_c)) - 1) * np.outer(pc, pc)
            # The second term of the rank-mu update
            rank_mu_term = c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * y_sigma[:, np.newaxis, :] * y_sigma[:, np.newaxis, :] * np.transpose(y_sigma[:, np.newaxis, :]), axis=0)
            C += rank_mu_term
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2.0
            # Re-estimate eigenvalues and vectors
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigvals = np.clip(eigvals, 1e-16, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            sqrt_eigvals = np.sqrt(eigvals)
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

        # Main Loop
        # We will run CMA-ES for the entire budget, but we need to handle the "global" aspect.
        # Since Weierstrass is multimodal, starting at 0 might be bad.
        # Let's do a quick random initialization to find a better start.
        rng = np.random.default_rng(42)
        
        # Phase 1: Quick random search to find a good initial mean
        # Use 10% of budget for initialization? No, 500 calls is small.
        # Let's just start at 0.0 but with a larger initial sigma?
        # Or use the multi-start idea: Run CMA-ES from 0, but if it stagnates, restart?
        # Given the budget, a single run is safer for time.
        # However, to beat the current score, we need better exploration.
        # Let's initialize x_mean with a random point in [-2, 2] to avoid boundary issues?
        # Actually, the optimum is in [-4, 4].
        # Let's stick to x_mean = 0, sigma = 2.0 to explore wider initially.
        x_mean = np.zeros(dim)
        sigma = 2.0

        while calls < budget and time.time() < deadline:
            # Generate population
            Y = np.zeros((lambda_, dim))
            for i in range(lambda_):
                # Sample from N(0, C)
                # Using Cholesky or Eigendecomposition
                # x = x_mean + sigma * z
                z = np.random.multivariate_normal(np.zeros(dim), C, 1).flatten()
                # Alternative faster method using eigvecs
                # z = eigvecs @ (sqrt_eigvals * rng.standard_normal(dim))
                # Let's use the eigvec method for speed
                z = eigvecs @ (sqrt_eigvals * rng.standard_normal(dim))
                Y[i] = x_mean + sigma * z
            
            # Evaluate
            F = np.zeros(lambda_)
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                x_i = Y[i]
                # Clip to box
                x_i_clipped = np.clip(x_i, -5.0, 5.0)
                F[i] = func(x_i_clipped)
                calls += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x_i_clipped.copy()
            
            if calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(F)
            # Take top mu
            Y_sel = Y[order[:mu]]
            F_sel = F[order[:mu]]
            
            # Update mean
            x_new_mean = np.sum(weights[:, np.newaxis] * Y_sel, axis=0)
            
            # Update step size
            y_sigma = (x_new_mean - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv @ y_sigma
            sigma *= np.exp(c_sigma / chi_lambda * (np.linalg.norm(ps) / chi_lambda - 1))
            
            # Update C
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            update_covariance(Y_sel)
            
            # Update state
            x_mean = x_new_mean
            # Prevent sigma from exploding or vanishing
            sigma = np.clip(sigma, 1e-10, 10.0)

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
