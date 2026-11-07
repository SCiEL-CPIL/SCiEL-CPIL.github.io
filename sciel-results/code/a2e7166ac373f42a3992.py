import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 320.0  # Keep a 10s margin before the 330s kill

        # CMA-ES Parameters
        # Initial sigma should be large enough to explore the [-5, 5]^5 box
        # Box width is 10, so sigma=2 covers a good portion.
        sigma = 2.0
        # Population size lambda = 4 + floor(3 * log(n))
        lambda_ = 4 + int(3 * np.log(dim))
        # Number of selected individuals mu = lambda_ / 2
        mu = max(1, lambda_ // 2)
        
        # Weights for CMA-ES
        # Standard weights: w_i = log(mu + 0.5) - log(i)
        weights = np.log(mu + 0.05) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        # cc (cumulation rate for covariance matrix)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        # cs (cumulation rate for step size)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        # c1 (base learning rate for covariance matrix)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        # cmu (learning rate for covariance matrix from selected individuals)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial mean: Uniform random start
        # Using a fixed seed for reproducibility is good, but the task implies
        # the function has a hidden shift. A random start is safer to avoid
        # bias if the random seed aligns poorly with the hidden shift.
        # However, deterministic behavior is often preferred in these benchmarks
        # unless the function is known to be adversarial to specific seeds.
        # We'll use a random start from a default RNG seeded by time or fixed?
        # The prompt says "hidden seeded shift". If we use a fixed seed,
        # we might consistently start in a bad spot for some instances.
        # Let's use a random seed based on system time to diversify starts
        # across potential re-runs, or just a fixed seed if we assume
        # the evaluator runs it once per instance.
        # Actually, the task says "For each instance... evaluator creates Optimizer".
        # If we use np.random.default_rng() without seed, it's random.
        # Let's use a fixed seed for the *algorithm's internal* randomness
        # but initialize the mean with a random point from a separate RNG
        # or just a fixed random point.
        # Given the "hidden shift", a random initialization is safer.
        rng = np.random.default_rng()
        
        mean = rng.uniform(-5, 5, dim)
        # Initial covariance matrix is identity
        C = np.eye(dim)
        # Precompute D = sqrt(trace(C)) for normalization? No, standard CMA-ES uses
        # the Cholesky factorization of C for sampling.
        
        # State variables
        ps = np.zeros(dim)  # Step size adaptation path
        pc = np.zeros(dim)  # Covariance adaptation path
        
        # Cache for function evaluations to avoid duplicate calls?
        # The task says "Every call is recorded". Caching doesn't reduce the
        # *count* of calls if we just skip the call, but we must return a value.
        # However, CMA-ES generates unique points usually.
        # We will not cache to keep it simple and compliant with "call count".
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # We need to handle the case where we run out of budget
        # We will loop until calls >= budget or time limit
        
        while calls < budget:
            if time.time() - start_time > time_limit:
                break
            
            # 1. Sample lambda_ points
            # Compute Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or add jitter
                C += np.eye(dim) * 1e-10
                try:
                    L = np.linalg.cholesky(C)
                except:
                    # Fallback to identity if still failing
                    L = np.eye(dim)
            
            # Sample standard normal vectors
            z = rng.standard_normal((lambda_, dim))
            # Transform to x
            # x_i = mean + sigma * (L @ z_i.T).T
            pop = mean + sigma * (z @ L.T)
            
            # Clip to box [-5, 5]
            # Note: Clipping can distort the distribution, but the task
            # requires the input to be clipped by func anyway.
            # We clip here to ensure we don't waste calls on points that
            # will be clipped to the same boundary, although func handles it.
            # Actually, func clips. So we can pass raw points.
            # However, if we pass raw points, many might be far outside.
            # It's better to clip our proposals to stay within the box
            # to maintain the statistical properties of the search
            # within the feasible region.
            pop = np.clip(pop, -5, 5)
            
            # 2. Evaluate population
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                if calls >= budget:
                    break
                fvals[i] = func(pop[i])
                calls += 1
                
                # Update global best
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()
            
            # If we broke early due to budget, we might have NaNs in fvals
            # We need to handle this. If calls >= budget, we should break
            # after this iteration if we didn't finish the population eval.
            if calls >= budget:
                break
            
            # 3. Sort indices by fitness
            idx = np.argsort(fvals)
            # Take the best mu individuals
            best_idx = idx[:mu]
            best_pop = pop[best_idx]
            best_fvals = fvals[best_idx]
            
            # 4. Update mean
            # m_new = sum(weights[i] * x_i)
            mean_new = np.sum(weights[:, None] * best_pop, axis=0)
            
            # 5. Update step size adaptation path ps
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) / sigma * (C^-1/2 * (m_new - m))
            # We need C^-1/2. We can use the Cholesky factor L.
            # C = L L^T. C^-1/2 is not simply L^-1.
            # Standard CMA-ES implementation often computes the inverse square root
            # or uses the eigendecomposition.
            # For simplicity and speed in Python, we can approximate or use
            # the fact that we have L.
            # Actually, the term is C^{-1/2} (m_new - m).
            # Let's compute the inverse square root of C.
            # This is computationally expensive.
            # Alternative: Use the "CMA-ES with BFGS" or simplified update?
            # No, let's stick to standard.
            # We can compute C^-1/2 via eigendecomposition: C = V D V^T, C^-1/2 = V D^-1/2 V^T
            # This is O(n^3). n=5, so it's fast.
            
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            diff = mean_new - mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ diff) / sigma
            
            # 6. Update sigma (step size)
            # hsig = (||ps|| / sqrt(1 - (1 - cs)^2)) / E||N(0, I)|| < 1.4 + 2/(dim+1)
            ps_norm = np.linalg.norm(ps)
            e_norm = np.sqrt(np.pi / 2) * np.sqrt(dim)  # Approx E||N(0,I)|| for dim=5 is ~2.5, sqrt(pi/2)*sqrt(5) ~ 1.77*2.23 ~ 3.9? 
            # Actually E||N(0, I_d)|| = sqrt(2) * Gamma((d+1)/2) / Gamma(d/2).
            # For d=5: sqrt(2) * Gamma(3) / Gamma(2.5) = 1.414 * 2 / 1.329 = 2.12.
            # Let's use a more accurate approximation or just the standard formula.
            # Standard formula: hsig = (||ps|| / sqrt(1 - (1 - cs)^2)) / E||N(0, I)||
            # E||N(0, I)|| for d=5 is approx 2.12.
            e_norm = 2.12 
            hsig = (ps_norm / np.sqrt(1 - (1 - cs) ** 2)) / e_norm
            if hsig < 1.4 + 2.0 / (dim + 1):
                sigma *= np.exp(cs / (dim + 3) * (ps_norm / e_norm - 1))
            
            # 7. Update covariance matrix
            # hc = (1 - cc) ... similar to hsig but for pc
            pc_norm = np.linalg.norm(pc)
            hc = (pc_norm / np.sqrt(1 - (1 - cc) ** 2)) / e_norm < 1.4 + 2.0 / (dim + 1)
            
            # pc = (1 - cc) * pc + hc * sqrt(cc * (2 - cc) * mu_eff) * (diff / sigma)
            pc = (1 - cc) * pc + hc * np.sqrt(cc * (2 - cc) * mu_eff) * (diff / sigma)
            
            # Rank-one update
            c1_term = c1 * (np.outer(diff, diff) / sigma**2 + (1 - hc**2) * cc * (2 - cc) * C - C)
            # Wait, the rank-one update formula is:
            # C += c1 * ( (diff/sigma) (diff/sigma)^T + (1 - hc^2) cc (2 - cc) C - C )
            # Actually, the term is:
            # C <- (1 - c1 - cmu) C + c1 * ( (diff/sigma)(diff/sigma)^T + (1 - hc^2) cc (2 - cc) C ) + cmu * sum w_i (z_i)(z_i)^T
            # Where z_i = (x_i - m_old) / (sigma * L) ... no, z_i are the sampled normals?
            # The rank-mu update uses the transformed points y_i = (x_i - m_old) / sigma.
            # y_i = L z_i. So y_i y_i^T = L z_i z_i^T L^T.
            
            # Let's re-construct the update properly.
            # y_diff = diff / sigma
            y_diff = diff / sigma
            
            # Rank-one update part
            # C <- (1 - c1 - cmu) * C
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(y_diff, y_diff) + (1 - hc**2) * cc * (2 - cc) * C)
            
            # Rank-mu update
            # We need the y_i for the best mu individuals
            # y_i = (x_i - m_old) / sigma
            y_best = (best_pop - mean) / sigma
            # Sum of weighted outer products
            # sum w_i y_i y_i^T
            # We can do this as: Y = y_best (mu x dim). 
            # Sum = (Y * weights[:, None]).T @ Y ? No.
            # Sum_i w_i y_i y_i^T = (Y * weights[:, None]).T @ Y is wrong.
            # It is: sum_i w_i * outer(y_i, y_i)
            # Which is: (Y * weights[:, None]).T @ Y is not quite right either.
            # Let Y be mu x dim.
            # We want S = sum_{i=0}^{mu-1} w_i * Y[i, :]^T Y[i, :]
            # This is equivalent to: (Y * weights[:, None]).T @ Y is a dim x dim matrix.
            # Let's verify:
            # (Y * w)^T is dim x mu.
            # (dim x mu) @ (mu x dim) = dim x dim.
            # Element (a, b) = sum_i (Y[i, a] * w[i]) * Y[i, b] = sum_i w_i Y[i, a] Y[i, b].
            # Yes, this is correct.
            
            S_mu = (y_best * weights[:, None]).T @ y_best
            C += cmu * S_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            mean = mean_new
            
            # Check for convergence
            if sigma < 1e-15:
                break
            if np.linalg.norm(diff) < 1e-15:
                # Stagnant, might restart? 
                # For now, just continue, sigma will shrink.
                pass

        return best_x, best_f
