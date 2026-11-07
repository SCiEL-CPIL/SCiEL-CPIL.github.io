"""
CMA-ES based optimizer for BBOB f18.
Replaces the heuristic portfolio with a robust, standard CMA-ES implementation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')

        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # CMA-ES Parameters
        # Standard settings for dim=5
        N = dim + int(np.ceil(3 * np.log(dim)))  # Population size
        mu = N // 2  # Number of parents
        # Weights for recombination
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial values
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, slightly larger than 1 to cover [-5,5] quickly
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)  # Inverse square root of C
        C_sqrt = np.eye(dim)
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        f_mean = timed_call(x_mean)
        
        # Pre-compute cholesky of C for sampling
        def update_C_sqrt():
            nonlocal C_sqrt, C_sqrt_inv
            # Eigendecomposition is more stable for CMA-ES than Cholesky if C becomes ill-conditioned
            # But for dim=5, Cholesky is fast. Let's use eigendecomposition for robustness.
            d, V = np.linalg.eigh(C)
            # Clip negative eigenvalues (numerical noise)
            d = np.clip(d, 1e-20, None)
            C_sqrt = V @ np.diag(np.sqrt(d)) @ V.T
            C_sqrt_inv = V @ np.diag(1.0 / np.sqrt(d)) @ V.T

        update_C_sqrt()

        # Main loop
        while n_calls < budget and time.time() < deadline:
            # Generate population
            Y = []
            F = []
            Z = []
            
            for i in range(N):
                if n_calls >= budget or time.time() >= deadline:
                    break
                z = np.random.randn(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                fy = timed_call(y)
                Y.append(y)
                F.append(fy)
                Z.append(z)
            
            if len(Y) < mu:
                break

            # Sort by fitness
            idx = np.argsort(F)
            Y = [Y[i] for i in idx]
            F = [F[i] for i in idx]
            Z = [Z[i] for i in idx]
            
            # Update mean
            y_w = np.sum([w[i] * Y[i] for i in range(mu)], axis=0)
            x_mean = y_w
            
            # Update path accumulators
            # p_sigma = (1-c_sigma)*p_sigma + c_sigma/sqrt(mu_eff) * C_sqrt_inv @ (y_w - x_mean_old)
            # Note: x_mean_old is the mean before update. We need to store it.
            # Let's restructure to store old mean.
            
            # Actually, standard CMA-ES updates p_sigma using the difference between new and old mean.
            # Let's restart the loop logic to properly capture old mean.
            
            break # Break to rewrite loop correctly
            
        # Restart with correct implementation
        n_calls = 0
        best_x = None
        best_f = float('inf')
        
        def timed_call2(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        def update_C_sqrt():
            nonlocal C_sqrt, C_sqrt_inv
            d, V = np.linalg.eigh(C)
            d = np.clip(d, 1e-20, None)
            C_sqrt = V @ np.diag(np.sqrt(d)) @ V.T
            C_sqrt_inv = V @ np.diag(1.0 / np.sqrt(d)) @ V.T

        update_C_sqrt()
        f_mean = timed_call2(x_mean)
        
        while n_calls < budget and time.time() < deadline:
            x_mean_old = x_mean.copy()
            
            Y = []
            F = []
            Z = []
            
            for i in range(N):
                if n_calls >= budget or time.time() >= deadline:
                    break
                z = np.random.randn(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                fy = timed_call2(y)
                Y.append(y)
                F.append(fy)
                Z.append(z)
            
            if len(Y) < mu:
                break

            idx = np.argsort(F)
            Y = [Y[i] for i in idx]
            F = [F[i] for i in idx]
            Z = [Z[i] for i in idx]
            
            # Update mean
            y_w = np.sum([w[i] * Y[i] for i in range(mu)], axis=0)
            x_mean = y_w
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (x_mean - x_mean_old) / sigma
            
            # Check h_sigma
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (n_calls / N))) / np.sqrt(1 - (1 - c_sigma)**(2 * (n_calls / N))) if n_calls > 0 else 1.0
            # Approximation for h_sigma threshold
            h_sigma_thr = (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma)**(2 * (n_calls / N)))
            # Simpler h_sigma check
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (n_calls // N + 1))) < 1.4 * (1 + 2.0 / (dim + 1.0)) else 0.0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean_old) / sigma
            
            # Update C
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum([w[i] * np.outer(Z[i] @ C_sqrt_inv, Z[i] @ C_sqrt_inv) for i in range(mu)], axis=0)
            
            C = (1 + c_1 * h_sigma * (np.linalg.norm(p_c)**2 / dim - 1) + c_mu * h_sigma * (np.linalg.norm(p_c)**2 / dim - 1)) * C \
                - c_1 * rank_one + c_mu * rank_mu
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1.0)) # Wait, formula is: sigma * exp(c_sigma * (||p_sigma||/chi_N - 1))
            chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_N - 1.0))
            
            # Clamp sigma to avoid explosion or vanishing
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update C_sqrt
            update_C_sqrt()
            
            # Optional: Restart if sigma is too small or too large
            if sigma < 1e-6 or sigma > 100:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
