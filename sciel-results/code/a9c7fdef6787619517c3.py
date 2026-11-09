"""CMA-ES optimizer for BBOB f7 (Step ellipsoidal) with Adaptive Restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation tuned for 5D BBOB f7 with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters for dim=5
        lam = 10  # Population size
        mu = lam // 2  # Number of recombination weights

        # Compute recombination weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        w = np.concatenate([w, np.zeros(lam - mu)])
        mueff = w.sum() ** 2 / (w ** 2).sum()

        # Strategy parameters
        c_sigma = (mueff + 2) / ((dim + 1) ** 1.5 + mueff)
        c_c = (4 + dim / c_sigma) / ((dim + 2) ** 1.5 + mueff)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Quick random search to find a good starting point
        # f7 is step ellipsoidal, so a broad search helps find the 'step'
        n_random = min(20, budget // 10)
        for i in range(n_random):
            if time.time() > deadline - 5:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # CMA-ES Initialization
        def init_cma(x_start, sigma_start):
            return {
                'x_mean': x_start.copy(),
                'sigma': sigma_start,
                'C': np.eye(dim),
                'p_sigma': np.zeros(dim),
                'p_c': np.zeros(dim),
                'evals': 0
            }

        # Start with a relatively large sigma to explore the step structure
        cma_state = init_cma(best_x, 2.0)
        
        # Stagnation tracking
        stagnation_counter = 0
        # We restart if no improvement in ~150 evaluations (15 generations of size 10)
        # This is a structural change to handle the flat plateaus of f7
        stagnation_threshold = 150 

        # Main CMA-ES loop
        while calls_made < budget:
            if time.time() > deadline - 2:
                break

            # Check if we have enough budget for a full generation
            if budget - calls_made < lam:
                # If not enough for a full generation, do single-step (1+1) ES style
                # or just break if we can't do a full CMA step effectively
                # For simplicity and robustness, break if we can't complete a population
                # unless we have at least 1 eval left, in which case we might do a random step
                if budget - calls_made == 1:
                     # One last random step
                     x_try = best_x + rng.normal(0, cma_state['sigma'], dim)
                     x_try = np.clip(x_try, -5, 5)
                     f_try = func(x_try)
                     calls_made += 1
                     if f_try < best_f:
                         best_f = f_try
                         best_x = x_try.copy()
                break

            # Generate offspring
            Y = []
            for i in range(lam):
                if time.time() > deadline - 2:
                    break
                z = rng.standard_normal(dim)
                # Use Cholesky decomposition for sampling from N(0, C)
                # For dim=5, this is very fast
                try:
                    L = np.linalg.cholesky(cma_state['C'])
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite
                    L = np.eye(dim)
                y = cma_state['x_mean'] + cma_state['sigma'] * (L @ z)
                y = np.clip(y, -5, 5)
                Y.append(y)

            # Evaluate offspring
            F = []
            for y in Y:
                if time.time() > deadline - 2:
                    break
                fy = func(y)
                F.append(fy)
                calls_made += 1

            if len(F) < lam:
                # Pad with worst value if we ran out of time
                F.extend([best_f] * (lam - len(F)))

            # Sort by fitness
            idx = np.argsort(F)
            Y = np.array(Y)[idx]
            F = np.array(F)[idx]

            # Update best
            improved = False
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
                improved = True
                stagnation_counter = 0
            else:
                stagnation_counter += lam

            # Check for Adaptive Restart
            # If stagnated for too long, reset CMA-ES from current best with larger sigma
            if stagnation_counter > stagnation_threshold:
                # Restart
                cma_state = init_cma(best_x, max(cma_state['sigma'] * 2.0, 1.0))
                stagnation_counter = 0
                # Continue to next iteration to generate new population

            cma_state['evals'] += lam

            # Compute weighted mean
            x_new = np.sum(w[:mu, None] * Y[:mu], axis=0)

            # Update evolution paths
            diff = (x_new - cma_state['x_mean']) / cma_state['sigma']
            
            # For path updates, we ideally use C^{-1/2} diff, but standard CMA-ES 
            # implementations often use the raw diff for p_sigma and C @ diff for p_c 
            # in simplified versions, or proper transformations. 
            # Given dim=5, we can afford to compute C_inv_sqrt for better performance.
            try:
                eigvals, eigvecs = np.linalg.eigh(cma_state['C'])
                eigvals = np.clip(eigvals, 1e-20, 1e10)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
            
            # Actually, the standard update for p_sigma is:
            # p_sigma = (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mueff) * C^{-1/2}(x_new - x_mean)/sigma
            # And for p_c:
            # p_c = (1-c_c)p_c + sqrt(c_c(2-c_c)) * C^{-1/2}(x_new - x_mean)/sigma
            
            # Wait, the standard formula uses C^{-1/2} for the transformation of the mean shift.
            # Let's stick to the robust implementation from the inspiration which used 
            # eigendecomposition for C_inv_sqrt.
            
            p_sigma = (1 - c_sigma) * cma_state['p_sigma'] + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ diff)
            p_c = (1 - c_c) * cma_state['p_c'] + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ diff)

            # Update covariance matrix
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * cma_state['evals']))) / chi_N < 1.4 + 2 / (dim + 1) else 0.0
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)

            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                # The rank-mu update uses the y vectors (standardized steps)
                # y_i = (Y[i] - x_mean_old) / sigma
                y_i = (Y[i] - cma_state['x_mean']) / cma_state['sigma']
                rank_mu += w[i] * np.outer(y_i, y_i)

            C_new = (1 + c_1 * delta_h - c_1 - c_mu) * cma_state['C'] + c_1 * (np.outer(p_c, p_c) + delta_h * cma_state['C']) + c_mu * rank_mu
            C_new = (C_new + C_new.T) / 2 # Symmetrize
            C_new += 1e-12 * np.eye(dim) # Regularize

            # Update step size
            sigma_new = cma_state['sigma'] * np.exp(c_sigma / chi_N * (np.linalg.norm(p_sigma) / chi_N - 1))

            # Update mean
            x_mean_new = x_new

            # Assign new state
            cma_state['C'] = C_new
            cma_state['sigma'] = sigma_new
            cma_state['x_mean'] = x_mean_new
            cma_state['p_sigma'] = p_sigma
            cma_state['p_c'] = p_c

            # Check convergence
            if cma_state['sigma'] < 1e-10:
                break
            if np.all(np.abs(cma_state['x_mean'] - best_x) < 1e-8):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
