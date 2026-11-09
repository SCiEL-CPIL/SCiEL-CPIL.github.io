# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with Halton warm-start.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Helper: Halton sequence
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q /= base
                r += q * (index % base)
                index //= base
            return r

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        def generate_halton(n, dim):
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for j in range(dim):
                    base = primes[j % len(primes)]
                    points[i - 1, j] = halton_sequence(i, base)
            return points

        # Global state
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Phase 1: Coarse Halton initialization
        # Use 20 calls for global exploration to find a good starting point
        n_coarse = 20
        halton_coarse = generate_halton(n_coarse, dim)
        halton_coarse = halton_coarse * 10.0 - 5.0
        
        # Add a few strategic points to ensure coverage
        strategic = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
        ]
        
        all_init_points = list(halton_coarse) + strategic
        rng.shuffle(all_init_points)

        for pt in all_init_points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 2: CMA-ES
        # Parameters
        lambda_pop = 4 + int(3 * np.log(dim))  # Population size
        mu = lambda_pop // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial state
        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute inverse square root of C using SVD or eigendecomposition
        # We will update C_sqrt_inv periodically or use a simpler update rule
        # For efficiency in 5D, we can recompute every few steps or use a robust method.
        # Let's use a helper to update C and its factors.
        
        def update_cma_factors(C):
            # Use eigendecomposition for stability in low dimensions
            evals, evecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            evals = np.maximum(evals, 1e-12)
            # Inverse square root
            C_sqrt_inv = (evecs / np.sqrt(evals)) @ evecs.T
            C_inv = (evecs / evals) @ evecs.T
            return C_inv, C_sqrt_inv

        C_inv, C_sqrt_inv = update_cma_factors(C)
        
        no_improve_count = 0
        stagnation_counter = 0
        
        # Restart pool for if CMA-ES gets stuck
        n_restart_pool = 50
        restart_pool = generate_halton(n_restart_pool, dim)
        restart_pool = restart_pool * 10.0 - 5.0
        restart_idx = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                # Transform z using C_sqrt_inv? No, we generate from N(0, C)
                # x = x_mean + sigma * (C_sqrt @ z)
                # We have C_inv, we need C_sqrt.
                # Let's just use the fact that if we have C, we can sample.
                # For 5D, np.linalg.cholesky is fast.
                try:
                    L = np.linalg.cholesky(C)
                    y = x_mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C becomes singular
                    y = x_mean + sigma * z
                y = np.clip(y, -5, 5)
                f_y = func(y)
                calls += 1
                offspring.append((f_y, y))
                
                if calls >= budget:
                    break
            
            if calls >= budget:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda x: x[0])
            
            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()
                no_improve_count = 0
                stagnation_counter = 0
            else:
                no_improve_count += 1
                stagnation_counter += 1

            # Select parents
            parents = [o[1] for o in offspring[:mu]]
            
            # Calculate weighted mean
            x_new_mean = np.zeros(dim)
            for i, p in enumerate(parents):
                x_new_mean += weights[i] * p
            x_new_mean = np.clip(x_new_mean, -5, 5)
            
            # Update step size (CSA)
            y_old = x_new_mean - x_mean
            norm_y_old = np.linalg.norm(y_old)
            
            # Expected norm of N(0, C)
            c_sigma_norm = 1.0 # Approximation for simplicity, usually c_sigma * sqrt(dim)
            # More accurate: c_sigma * E[||N(0, C)||]
            # For now, use standard CSA update
            h_sigma = 1.0
            if norm_y_old > 1e-12:
                h_sigma = min(1.0, (np.trace(C_inv) - 1) / (2 * (dim + 1) * np.sqrt(np.trace(C_inv) * np.trace(C_inv) - 1)))
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ y_old) / sigma
            
            norm_p_sigma = np.linalg.norm(p_sigma)
            expected_norm = 1.0 # Approximation
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / expected_norm - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance matrix
            y_c = y_old / sigma
            p_c = (1 - c_c) * p_c + c_c * y_c
            
            # Rank-1 update
            C = (1 + c_1 * (c_c * (1 - c_c) + (1 - c_c) * mu_eff) - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) + (c_c * (1 - c_c) * (np.eye(dim) - C))) \
                + c_mu * np.sum([weights[i] * np.outer(parents[i] - x_mean, parents[i] - x_mean) / (sigma ** 2) for i in range(mu)], axis=0)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2.0
            # Add small regularization to prevent singularity
            C += 1e-12 * np.eye(dim)
            
            # Update factors
            C_inv, C_sqrt_inv = update_cma_factors(C)
            
            # Update mean
            x_mean = x_new_mean
            
            # Check for stagnation/restart
            # If sigma is too small or no improvement for a long time
            stagnation_limit = 50 + int(20 * (1 - calls / budget))
            
            if no_improve_count > stagnation_limit or sigma < 1e-4:
                # Restart
                if restart_idx < len(restart_pool):
                    new_center = restart_pool[restart_idx].copy()
                    restart_idx += 1
                else:
                    new_center = rng.uniform(-5, 5, dim)
                
                # Check if new center is better
                f_new_center = func(new_center)
                calls += 1
                if f_new_center < best_f:
                    best_f = f_new_center
                    best_x = new_center.copy()
                
                x_mean = new_center
                sigma = 1.0
                C = np.eye(dim)
                C_inv, C_sqrt_inv = update_cma_factors(C)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0
                stagnation_counter = 0
                
                if calls >= budget:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
