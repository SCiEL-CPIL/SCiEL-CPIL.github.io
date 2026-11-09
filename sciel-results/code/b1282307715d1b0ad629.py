# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Global-Local Optimizer for BBOB f4 (Büche-Rastrigin).
    
    Strategy:
    1. Phase 1 (Global): Sparse Gaussian Process (GP) surrogate with Expected Improvement (EI)
       acquisition to efficiently navigate the rotated landscape and locate the global basin.
       This addresses the 'sparse' nature of the landscape and avoids the fixed-grid limitation.
    2. Phase 2 (Local): CMA-ES refinement starting from the GP's best estimate to converge
       precisely to the minimum within the identified basin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        dim = self.dim
        budget = self.budget

        # --- Phase 1: GP Surrogate with EI Acquisition ---
        # Use a portion of the budget for global exploration.
        # 100 calls for initial sampling + 200 for sequential EI optimization = 300 calls.
        # Remaining 200 calls for CMA-ES refinement.
        gp_budget = 300
        
        X = []
        Y = []
        
        # Initial Sampling: Latin Hypercube Sampling (LHS) for better coverage than random/grid
        n_init = 50
        # Generate LHS points
        # Scale to [-5, 5]
        # Simple LHS implementation
        indices = np.arange(n_init)
        # Shuffle indices for each dimension to create LHS
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            lhs_points[:, d] = perm
        # Normalize to [0, 1) then scale to [-5, 5]
        lhs_points = lhs_points / n_init
        lhs_points = lhs_points * 10.0 - 5.0
        
        # Evaluate initial points
        for point in lhs_points:
            if calls >= budget or time.time() > deadline:
                break
            fx = evaluate(point)
            if fx is None:
                break
            X.append(point)
            Y.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = point.copy()
        
        if len(X) == 0:
            # Fallback if no evaluations were made (should not happen)
            return None, np.inf

        X = np.array(X)
        Y = np.array(Y)
        
        # GP Hyperparameters
        # RBF Kernel: k(x, x') = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / l^2)
        # We will use a simple length-scale and signal variance.
        # Since the landscape is rotated, a single length-scale might not be perfect,
        # but it's a good start. We can use a diagonal kernel or just isotropic.
        # For speed and simplicity, let's use an isotropic RBF with fixed hyperparameters
        # tuned for the box [-5, 5]^5.
        # Typical distances are ~2-4. Length scale l=1.5 seems reasonable.
        sigma_f = 10.0  # Signal variance
        l = 1.5         # Length scale
        sigma_n = 1e-4  # Noise variance (small, as function is deterministic)
        
        def gp_mean_var(X_obs, Y_obs, X_new, sigma_f, l, sigma_n):
            """
            Compute mean and variance of GP at X_new given observations X_obs, Y_obs.
            """
            n_obs = X_obs.shape[0]
            n_new = X_new.shape[0]
            
            # Compute kernel matrix K(X_obs, X_obs)
            # Squared Euclidean distance
            diff = X_obs[:, np.newaxis, :] - X_obs[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            K = sigma_f ** 2 * np.exp(-0.5 * dist_sq / (l ** 2))
            K[np.diag_indices_from(K)] += sigma_n
            
            # Compute kernel matrix K(X_obs, X_new)
            diff_new = X_obs[:, np.newaxis, :] - X_new[np.newaxis, :, :]
            dist_sq_new = np.sum(diff_new ** 2, axis=-1)
            K_new = sigma_f ** 2 * np.exp(-0.5 * dist_sq_new / (l ** 2))
            
            # Compute kernel matrix K(X_new, X_new)
            diff_new_new = X_new[:, np.newaxis, :] - X_new[np.newaxis, :, :]
            dist_sq_new_new = np.sum(diff_new_new ** 2, axis=-1)
            K_new_new = sigma_f ** 2 * np.exp(-0.5 * dist_sq_new_new / (l ** 2))
            
            # Solve for mean and variance
            # mean = K_new.T @ inv(K) @ Y_obs
            # var = K_new_new - K_new.T @ inv(K) @ K_new
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_obs))
                mean = K_new.T @ alpha
                
                v = np.linalg.solve(L, K_new)
                var = np.diag(K_new_new) - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-10) # Ensure positivity
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if Cholesky fails
                K_inv = np.linalg.pinv(K)
                mean = K_new.T @ (K_inv @ Y_obs)
                var = np.diag(K_new_new) - np.sum((K_new.T @ K_inv * K_new.T), axis=0)
                var = np.maximum(var, 1e-10)
            
            return mean, var

        # Sequential EI Optimization
        # We will perform 10 iterations of EI acquisition, each time adding 20 new points.
        # Total new points: 200. Total calls: 50 (init) + 200 (EI) = 250.
        # Remaining budget: 500 - 250 = 250 for CMA-ES.
        
        n_ei_rounds = 10
        n_ei_points_per_round = 20
        
        for _ in range(n_ei_rounds):
            if calls >= budget or time.time() > deadline:
                break
            
            # Candidate points for EI: 
            # 1. Random points in the box
            # 2. Points near current best
            # We generate a large pool of candidates and pick the top EI ones.
            n_candidates = 200
            candidates = rng.uniform(-5, 5, size=(n_candidates, dim))
            
            # Compute GP mean and variance at candidates
            mean, var = gp_mean_var(np.array(X), np.array(Y), candidates, sigma_f, l, sigma_n)
            std = np.sqrt(var)
            
            # Current best
            current_best = min(Y)
            
            # Expected Improvement (EI)
            # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
            # Where Phi and phi are CDF and PDF of standard normal.
            # To avoid underflow/overflow, use scipy.stats.norm if available, else manual.
            # SciPy is available.
            from scipy.stats import norm
            
            z = (current_best - mean) / (std + 1e-10)
            ei = (current_best - mean) * norm.cdf(z) + std * norm.pdf(z)
            
            # Select top n_ei_points_per_round
            top_indices = np.argsort(ei)[-n_ei_points_per_round:]
            
            for idx in top_indices:
                if calls >= budget or time.time() > deadline:
                    break
                x_new = candidates[idx]
                fx = evaluate(x_new)
                if fx is None:
                    break
                X.append(x_new)
                Y.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x_new.copy()

        # Ensure we have at least some points for CMA-ES initialization
        if best_x is None:
            return None, np.inf

        # --- Phase 2: CMA-ES Refinement ---
        # Initialize CMA-ES at the best point found by GP.
        # Use a smaller initial sigma to avoid jumping out of the basin too quickly,
        # but large enough to explore the local structure.
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_N = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        
        # Track best
        best_x_cma = best_x.copy()
        best_f_cma = best_f
        
        # CMA-ES State
        x = best_x.copy()
        sigma = 0.5  # Smaller initial sigma for refinement
        
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Main CMA-ES Loop
        stall_count = 0
        stall_threshold = 15
        max_restarts = 3
        current_restart = 0
        
        while calls < budget and time.time() < deadline:
            # Generate population
            offspring = []
            fitnesses = []
            
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                y = rng.standard_normal(dim)
                z = C_sqrt @ y
                xi = x + sigma * z
                xi = np.clip(xi, -5.0, 5.0)
                fx = evaluate(xi)
                if fx is None:
                    break
                offspring.append(xi)
                fitnesses.append(fx)
                
                if fx < best_f_cma:
                    best_f_cma = fx
                    best_x_cma = xi.copy()
                if fx < best_f:
                    best_f = fx
                    best_x = xi.copy()
            
            if len(offspring) < lambda_:
                break
                
            # Sort by fitness
            indices = np.argsort(fitnesses)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_fitnesses = [fitnesses[i] for i in indices]
            
            # Check for improvement
            if sorted_fitnesses[0] < best_f_cma - 1e-10:
                stall_count = 0
            else:
                stall_count += 1
            
            # Update mean
            x_new = np.sum([weights[i] * sorted_offspring[i] for i in range(mu)], axis=0)
            
            # Update sigma (CSA)
            y_c = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ y_c
            h_sigma = True
            if h_sigma:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            
            # Update Covariance (CSA)
            if h_sigma:
                delta_h = (1 - h_sigma) * (1 - c_c) - 1
                rank_1 = np.outer(p_c, p_c)
                rank_mu = np.sum([weights[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x) for i in range(mu)], axis=0)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            else:
                rank_mu = np.sum([weights[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x) for i in range(mu)], axis=0)
                C = (1 - c_1 - c_mu) * C + c_mu * rank_mu
            
            # Update square roots
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.clip(eigenvalues, 1e-12, None)
            C_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            C_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))
            
            x = x_new
            
            # Termination condition: sigma too small
            if sigma < 1e-6:
                break
                
            # Restart logic
            if stall_count >= stall_threshold:
                current_restart += 1
                if current_restart > max_restarts:
                    break
                
                # Reset CMA-ES state around best found so far
                x = best_x + rng.normal(0, 0.5, dim)
                x = np.clip(x, -5.0, 5.0)
                
                sigma = 0.5
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stall_count = 0
                
                # Evaluate new center
                fx = evaluate(x)
                if fx is None:
                    break
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                if fx < best_f_cma:
                    best_f_cma = fx
                    best_x_cma = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
