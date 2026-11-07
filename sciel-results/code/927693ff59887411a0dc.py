"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        n = self.dim
        # CMA-ES Parameters
        # Standard lambda for n=5 is 4 + floor(3*log(5)) = 4 + 4 = 8, but 15 is often better for robustness
        # Let's use lambda = 15 for better exploration in early stages
        lambda_ = 15
        
        # Weights: standard CMA-ES weights
        # w_i = log((lambda+1)/2) - log(i) for i=1..lambda
        # mu = sum of weights where w_i > 0
        weights = np.log((lambda_ + 1) / 2) - np.log(np.arange(1, lambda_ + 1))
        weights = weights / np.sum(weights)
        mu = np.sum(weights > 0)
        # Effective population size
        mu_eff = 1.0 / np.sum(weights[:int(mu)] ** 2)
        
        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initial state
        # Start at 0 as it's a reasonable prior for shifted minimizer in [-4,4]
        x_mean = np.zeros(n)
        sigma = 2.0  # Initial step size, covers a good portion of the box
        
        # State variables
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        hs = np.sqrt((1 + c_c / (1 - c_c)) * (n + 1) + 1)
        
        # Initial population evaluation
        # Generate first population
        zs = np.random.default_rng(0).standard_normal((lambda_, n))
        # Normalize to unit length for initial step? No, standard normal is fine, scaled by sigma
        # Actually, CMA-ES usually starts with C=I, so samples are sigma * z
        # But we need to ensure we stay in bounds? CMA-ES doesn't natively handle bounds well.
        # The task says func clips. So we can just sample and let func clip.
        # However, if we sample far outside, we waste calls.
        # Initial sigma=2.0 with x_mean=0 means 95% of points are in [-4, 4] roughly.
        
        pop = x_mean + sigma * zs
        # Clip to box for evaluation? The func clips internally. 
        # But for tracking the "true" position for CMA-ES update, 
        # it's better to work in the unclipped space if possible, 
        # but the function is only defined on the clipped space? 
        # "func(x) clips a finite length-5 vector to the box and returns a float."
        # This means if we pass x outside, it clips. The gradient info is lost at boundaries.
        # Standard practice: just pass x to func.
        
        fvals = np.array([func(x) for x in pop])
        calls_made = lambda_
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx]
        best_f = fvals[best_idx]
        
        # Sort by fitness
        order = np.argsort(fvals)
        xs = pop[order]
        fs = fvals[order]
        
        # Weights for the best mu individuals
        w = weights[:int(mu)]
        
        # Update mean
        x_new = np.dot(w, xs)
        
        # Update sigma (CSA)
        # Calculate norm of (x_new - x_mean) / sigma
        norm_step = np.linalg.norm(x_new - x_mean) / sigma
        
        # Heaviside function for h_sigma
        h_sigma = 1.0 if norm_step < (1.4 + 2 / (n + 1)) * hs else 0.0
        
        p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_new - x_mean) / sigma
        
        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / hs - 1))
        
        # Update covariance matrix (CSA)
        # Update p_c
        # Note: p_c update uses the step in the C-frame? 
        # Standard CMA-ES update:
        # y = (x_new - x_mean) / sigma
        # p_c = (1 - c_c) * p_c + h_c * sqrt(c_c * (2 - c_c) * mu_eff) * y
        # h_c is similar to h_sigma but based on C^{-1/2} norm
        # For simplicity and stability, often h_c is approximated or calculated similarly.
        # Let's calculate h_c properly.
        # We need C^{-1/2}. Since C is initially I, and updates are small, we can approximate.
        # Or just use the same heuristic? No, better to be precise.
        # Let's compute C_inv_sqrt using eigen decomposition if needed, or just use the norm in C-space.
        # For n=5, eigendecomposition is cheap.
        
        # However, in the first iteration, C=I, so C_inv_sqrt = I.
        # Let's implement a helper for C_inv_sqrt if we want to be robust, 
        # but for n=5, we can just do it inline.
        
        # Actually, let's stick to a slightly simplified but robust CMA-ES implementation 
        # that avoids complex matrix square roots in the inner loop if possible, 
        # or just uses them since n=5 is tiny.
        
        # Let's refine the CMA-ES step.
        # The previous block was a bit messy. Let's restart the loop logic cleanly.
        
        # Reset state for a cleaner implementation structure
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        hs = np.sqrt((1 + c_c / (1 - c_c)) * (n + 1) + 1)
        
        # Re-evaluate initial population (we already did, but let's just reuse the loop structure)
        # We already have pop, fvals, order, xs, fs, x_new from above.
        # Let's just continue from there.
        
        # Update p_c
        # Need C_inv_sqrt. Since C=I initially, C_inv_sqrt = I.
        # In general, we need to update C first, then use it? No, p_c update uses current C.
        # y = (x_new - x_mean) / sigma
        y = (x_new - x_mean) / sigma
        
        # Calculate h_c
        # norm of y in C-space: sqrt(y^T C^{-1} y)
        # Since C=I, it's just norm(y).
        # Let's write a generic update loop.
        
        # We need to be careful with the order of operations in the standard CMA-ES algorithm.
        # 1. Sample population
        # 2. Evaluate
        # 3. Sort
        # 4. Update mean
        # 5. Update sigma (CSA)
        # 6. Update p_c (CSA)
        # 7. Update C (CMA)
        
        # We did 1-5 above. Now 6 and 7.
        
        # For step 6, we need C^{-1/2}.
        # Let's implement a function to get C_inv_sqrt.
        def get_c_inv_sqrt(C):
            # Eigen decomposition
            w, v = np.linalg.eigh(C)
            # Clip small eigenvalues to avoid division by zero
            w = np.clip(w, 1e-16, None)
            c_inv_sqrt = v @ np.diag(1.0 / np.sqrt(w)) @ v.T
            return c_inv_sqrt
        
        # In the first iteration, C is I, so C_inv_sqrt is I.
        # But to be general, let's compute it.
        C_inv_sqrt = get_c_inv_sqrt(C)
        
        # h_c calculation
        # norm of C_inv_sqrt @ y
        norm_y_c = np.linalg.norm(C_inv_sqrt @ y)
        # Threshold for h_c is similar to h_sigma
        # h_c = 1 if norm_y_c < (1.4 + 2/(n+1)) * hs else 0
        # Note: hs is defined for p_sigma. For p_c, it's usually the same heuristic or slightly different.
        # Standard CMA-ES uses the same hs for both.
        h_c = 1.0 if norm_y_c < (1.4 + 2 / (n + 1)) * hs else 0.0
        
        p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
        
        # Step 7: Update C
        # Delta H term
        delta_h = (1 - h_c) * c_c * (2 - c_c) * np.dot(p_c, p_c) / n  # Approximation, usually it's (1-h_c)*c_c*(2-c_c)*||p_c||^2/n
        # Actually, the standard formula is:
        # C = (1 - c_1 - c_mu + delta_h) * C + c_1 * (p_c p_c^T + (1-h_c) c_c (2-c_c) C) + c_mu * sum(w_i (y_i y_i^T))
        # Wait, the delta_h term is added to the (1-c_1-c_mu) part?
        # Let's use the standard formulation:
        # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_c) * c_c * (2 - c_c) * C) + c_mu * sum_{i=1}^{mu} w_i * y_i * y_i^T
        # Where y_i = (x_i - x_mean_old) / sigma
        
        # We need y_i for the best mu individuals.
        # xs are the sorted points.
        # x_mean_old is x_mean before update.
        y_s = (xs[:int(mu)] - x_mean) / sigma
        
        # Sum of outer products
        # c_mu * sum(w_i * y_i * y_i^T)
        # We can do this efficiently
        # sum_w_yy = sum(w[i] * outer(y[i], y[i]))
        # Using matrix multiplication:
        # Y = y_s (mu x n)
        # W = w (mu,)
        # Sum = Y.T @ (W[:, None] * Y)  -> (n x mu) @ (mu x n) -> (n x n)
        # Or: (Y * W[:, None]).T @ Y
        sum_w_yy = (y_s * w[:, None]).T @ y_s
        
        # Update C
        # Term 1: (1 - c_1 - c_mu) * C
        # Term 2: c_1 * (p_c p_c^T + (1 - h_c) * c_c * (2 - c_c) * C)
        # Term 3: c_mu * sum_w_yy
        
        C = (1 - c_1 - c_mu) * C + \
            c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C) + \
            c_mu * sum_w_yy
            
        # Ensure C is symmetric (numerical errors)
        C = (C + C.T) / 2
        
        # Update x_mean
        x_mean = x_new
        
        # Update best if we found a better point
        if fs[0] < best_f:
            best_f = fs[0]
            best_x = xs[0]
            
        # Check time and budget
        if time.time() > deadline:
            break
        if calls_made >= self.budget:
            break
            
        # Prepare for next iteration
        # Sample new population from N(x_mean, sigma^2 C)
        # We need to sample from multivariate normal with covariance sigma^2 C.
        # z ~ N(0, I)
        # x = x_mean + sigma * C_inv_sqrt @ z
        # Wait, if we sample z ~ N(0, I), then C_inv_sqrt @ z has covariance C_inv_sqrt @ I @ C_inv_sqrt^T = C^{-1}.
        # That's wrong. We want covariance C.
        # So we need C^{1/2} @ z.
        # Or, we can use the Cholesky decomposition of C.
        # L = cholesky(C)
        # x = x_mean + sigma * L @ z
        
        # Let's use Cholesky.
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # If C is not positive definite, add small identity
            C += 1e-6 * np.eye(n)
            L = np.linalg.cholesky(C)
            
        zs = np.random.default_rng().standard_normal((lambda_, n))
        pop = x_mean + sigma * (L @ zs.T).T
        
        fvals = np.array([func(x) for x in pop])
        calls_made += lambda_
        
        order = np.argsort(fvals)
        xs = pop[order]
        fs = fvals[order]
        
        # Update best
        if fs[0] < best_f:
            best_f = fs[0]
            best_x = xs[0]
            
        # Update mean
        x_new = np.dot(w, xs)
        
        # Update sigma
        norm_step = np.linalg.norm(x_new - x_mean) / sigma
        h_sigma = 1.0 if norm_step < (1.4 + 2 / (n + 1)) * hs else 0.0
        p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_new - x_mean) / sigma
        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / hs - 1))
        
        # Update p_c
        y = (x_new - x_mean) / sigma
        C_inv_sqrt = get_c_inv_sqrt(C)
        norm_y_c = np.linalg.norm(C_inv_sqrt @ y)
        h_c = 1.0 if norm_y_c < (1.4 + 2 / (n + 1)) * hs else 0.0
        p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
        
        # Update C
        y_s = (xs[:int(mu)] - x_mean) / sigma
        sum_w_yy = (y_s * w[:, None]).T @ y_s
        C = (1 - c_1 - c_mu) * C + \
            c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C) + \
            c_mu * sum_w_yy
        C = (C + C.T) / 2
        
        x_mean = x_new
        
        # Safety check for sigma
        if sigma < 1e-12:
            break

        return best_x, best_f
# EVOLVE-BLOCK-END
