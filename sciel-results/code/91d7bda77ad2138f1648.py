"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 280s margin within 300s limit

        best_x = None
        best_f = np.inf
        total_calls = 0

        def safe_call(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))
        # Number of parents
        mu = lambda_pop // 2
        # Weights for the weighted recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Initial mean and step size
        # Start near center, but maybe slightly off to avoid symmetry issues if any, 
        # though standard practice is center or random. 
        # Given the "hidden seeded shift", starting at 0 is a reasonable prior.
        mean = np.zeros(dim)
        sigma = 1.0 
        
        # Covariance matrix C = I
        C = np.eye(dim)
        # Cholesky factor of C
        C_sqrt = np.eye(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial population
        # We need to generate the first population
        # Standard CMA-ES initialization
        # Sample from N(0, I)
        
        def sample_population():
            nonlocal C_sqrt
            # Generate random numbers
            # z ~ N(0, I)
            z = np.random.randn(lambda_pop, dim)
            # x = mean + sigma * C_sqrt @ z
            # Vectorized: (lambda, dim) @ (dim, dim) -> (lambda, dim)
            # Note: C_sqrt is the Cholesky factor such that C = C_sqrt @ C_sqrt.T
            # Usually we use C_sqrt @ z.T or z @ C_sqrt.T depending on orientation.
            # Let's stick to standard: x_i = mean + sigma * (C_sqrt @ z_i)
            # In numpy: z is (lambda, dim). We want each row to be transformed.
            # x = mean + sigma * (z @ C_sqrt.T)
            return mean + sigma * (z @ C_sqrt.T)

        # Main loop
        while total_calls < budget:
            if time.time() > deadline:
                break
            
            # 1. Sample population
            pop = sample_population()
            
            # 2. Evaluate fitness
            fvals = np.array([safe_call(x) for x in pop])
            
            # 3. Sort by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # 4. Update mean
            # Weighted average of the mu best points
            new_mean = np.dot(weights, pop[:mu])
            
            # 5. Update covariance matrix and step size
            # Calculate the "evolution path" for the mean
            # h_sigma condition
            # This is a heuristic to avoid updating C when the step is too small/large
            # Standard implementation:
            # h_sigma = (1 - exp(-2 * c_sigma / (1 - c_sigma))) ... wait, simpler check
            # Let's use the standard update rules
            
            # Update p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + h_sigma * sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * C_sqrt^-1 @ (new_mean - mean) / sigma
            # We need C_sqrt^-1. Let's compute it.
            try:
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                # If singular, reset or break
                break
                
            # The vector (new_mean - mean) / sigma is the "step" in the standardized space
            step_std = (new_mean - mean) / sigma
            # Transform to the "natural" space for the path?
            # Actually, p_sigma is in the "standard" space (isotropic normal space)
            # So we project the step into the standard space: C_sqrt^-1 @ step_std
            step_std_proj = C_sqrt_inv @ step_std
            
            # h_sigma indicator
            # Check if the step is consistent with the path
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-c_sigma)^{2*(gen+1)}) < (1.4 + 2/(dim+1)) * chi_n
            # We don't track gen explicitly in a simple loop, but we can approximate or just use the simple version
            # For robustness, let's use a simplified h_sigma = 1.0 or a basic check.
            # A more robust check:
            # norm_p_sigma = np.linalg.norm(p_sigma)
            # expected_norm = chi_n * np.sqrt(1 - (1 - c_sigma)**(2 * 100)) # Assuming some history
            # This is tricky without a generation counter. Let's just use h_sigma = 1.0 for simplicity 
            # or implement a basic counter.
            gen = 0 # We can add a counter
            # Let's add a generation counter outside the loop
            # Actually, let's just initialize gen = 0 and increment it.
            # I'll restructure slightly to include gen.
            
            # Let's restart the loop logic with a proper gen counter
            
            # Update p_sigma
            # h_sigma = 1.0 (simplification, works reasonably well for short runs)
            # Better: 
            # if gen == 0: h_sigma = 1.0 else ...
            # Let's just use the standard formula with a gen counter.
            
            # I need to define gen before the loop.
            # Let's refactor the code block to include gen.
            
            # Since I can't easily refactor the whole block without seeing the full structure again, 
            # I will assume gen is tracked.
            # Let's insert `gen = 0` before the while loop.
            # And increment `gen += 1` inside.
            
            # Wait, I am writing the code from scratch in the response.
            # I will include `gen = 0` before the while loop.
            
            # Let's continue with the update assuming `gen` exists.
            
            # h_sigma calculation
            # The term (1 - (1 - c_sigma)**(2 * gen)) approaches 1.
            # Let's use a simple heuristic: h_sigma = 1.0 if the step is not too large.
            # Actually, the standard check is:
            # norm_p_sigma = np.linalg.norm(p_sigma)
            # threshold = (1.4 + 2/(dim+1)) * chi_n
            # h_sigma = 1 if norm_p_sigma / np.sqrt(1 - (1-c_sigma)**(2*gen)) < threshold else 0
            # This requires gen.
            
            # Let's implement it properly.
            
            # Update p_c
            # p_c = (1 - c_c) * p_c + h_c * sqrt(c_c * (2 - c_c) * mu_eff) * C^-1/2 @ (new_mean - mean) / sigma
            # h_c is similar to h_sigma but for p_c.
            # Often h_c = 1.0 is used or similar logic.
            
            # For simplicity and robustness in a black-box setting with limited budget, 
            # we can simplify the h_sigma and h_c checks to 1.0 or use the standard ones.
            # Let's use the standard ones.
            
            # First, calculate h_sigma
            # We need to track the norm of p_sigma relative to the expected norm.
            # This is getting complex to write inline. Let's use a simplified CMA-ES 
            # that might be slightly less theoretically pure but very robust:
            # "CMA-ES with simple restarts" or just standard CMA-ES with fixed h=1.
            # Given the budget is only 500, and dim=5, a standard CMA-ES should work well.
            # Let's stick to the standard update rules.
            
            # To avoid complexity of h_sigma/h_c in this short script, I will set them to 1.0 
            # which is a common simplification for small budgets or when adaptation is fast.
            h_sigma = 1.0
            h_c = 1.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * step_std_proj
            
            # Update p_c
            # step_std is in the "x-space" scaled by sigma.
            # We need C^-1/2 @ step_std.
            # C^-1/2 is the inverse of C_sqrt.
            step_std_proj_c = C_sqrt_inv @ step_std
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * step_std_proj_c
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - h_c) * c_c * (2 - c_c) * I) + c_mu * sum(weights_i * (y_i * y_i.T))
            # where y_i = (x_i - old_mean) / sigma
            
            # Calculate y_i for the mu best points
            y_best = (pop[:mu] - mean) / sigma
            # y_best is (mu, dim)
            
            # Term 1: c_1 * (p_c * p_c.T + (1 - h_c) * c_c * (2 - c_c) * I)
            # Since h_c = 1.0, the (1-h_c) term is 0.
            # So Term 1 = c_1 * (p_c * p_c.T)
            
            # Term 2: c_mu * sum(weights_i * (y_i * y_i.T))
            # We can compute this as: c_mu * (y_best.T @ (weights * y_best))
            # weights is (mu,), y_best is (mu, dim)
            # y_best.T @ (weights * y_best) is (dim, dim)
            
            delta_C = c_1 * np.outer(p_c, p_c) + c_mu * (y_best.T @ (weights * y_best))
            
            C = (1 - c_1 - c_mu) * C + delta_C
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update C_sqrt
            # Cholesky decomposition
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, add a small regularization
                C += 1e-12 * np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)
            
            # Update mean
            mean = new_mean
            
            # Increment generation
            gen += 1
            
            # Check for convergence
            # If sigma is very small or variance is very small
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
