"""Optimized BBOB f22 solver using CMA-ES with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Standard population size for N=5 is approx 4 + 3*ln(5) ~ 9. Let's use 10.
        lambda_pop = 10
        mu = lambda_pop // 2  # Number of recombination points
        
        # Initial step size. The box is [-5, 5]^5.
        # A sigma of 4.0 ensures the initial distribution covers the space well.
        sigma = 4.0
        mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        # State variables
        pc = np.zeros(n)
        ps = np.zeros(n)
        cs = (1 + c_sigma + c_c + 0.5 * c_mu) / (1 + c_sigma + c_c)
        
        best_x = mean.copy()
        best_f = np.inf
        calls_made = 0
        
        # Helper to evaluate and track best
        def eval_and_update(x):
            nonlocal best_x, best_f, calls_made
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Main loop
        while calls_made + lambda_pop <= budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            xs = []
            for i in range(lambda_pop):
                # Sample from N(mean, sigma^2 * C)
                # We need to sample from multivariate normal with covariance C
                # z ~ N(0, I), then y = mean + sigma * C^0.5 @ z
                # Since C is updated, we can use Cholesky or eigendecomposition.
                # For simplicity and speed in Python, we can use the fact that 
                # if we maintain C, we can sample via np.random.multivariate_normal
                # but that's slow. Better to do:
                # z = rng.standard_normal(n)
                # y = mean + sigma * (C ** 0.5) @ z
                # We need C ** 0.5. We can compute it every step or update it.
                # For N=5, Cholesky is cheap.
                
                # Actually, let's just use the standard CMA-ES update which maintains C.
                # Sampling:
                z = np.random.standard_normal(n)
                # Need square root of C. 
                # To avoid computing sqrt every time, we can just do:
                # y = mean + sigma * np.linalg.cholesky(C) @ z
                # But C is positive definite.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes singular
                    L = np.eye(n)
                    C = np.eye(n)
                    
                y = mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                xs.append(y)
                ys.append(y)
            
            # Evaluate
            fs = []
            for y in xs:
                f = eval_and_update(y)
                fs.append(f)
            
            # Sort by fitness
            idx = np.argsort(fs)
            xs = np.array(xs)[idx]
            fs = np.array(fs)[idx]
            
            # Update mean
            new_mean = np.sum(weights[:, None] * xs[:mu], axis=0)
            
            # Update covariance
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c)**2) * C) + c_mu * sum(wi * (xi - old_mean) @ (xi - old_mean).T)
            
            # First, update pc
            pc = (1 - c_c) * pc + c_c * (new_mean - mean) / sigma
            
            # Update C
            # Note: The standard formula uses (xi - mean_old)
            # Let's compute the rank-mu update
            diff = xs[:mu] - mean
            rank_mu_update = np.sum(weights[:, None] * (diff[:, :, None] * diff[:, None, :]), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c)**2) * C) + c_mu * rank_mu_update
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update step size sigma
            # hsig = np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(2 * gen)) / np.sqrt(n)
            # We need to track generations for the expected norm.
            # For simplicity, we can use a simpler adaptation or track gen.
            # Let's add a generation counter.
            
            # Actually, let's just use the standard update.
            # We need the expected norm of a standard normal vector.
            # E||z|| = sqrt(2) * Gamma((n+1)/2) / Gamma(n/2)
            # For n=5, this is a constant.
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(32*n**2))
            
            # We need to accumulate ps
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) * pc / sigma
            # Wait, the formula is:
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) * (new_mean - mean) / sigma
            # And hsig is based on pc.
            
            # Let's correct the ps update.
            # The term added to ps is the normalized step.
            step = (new_mean - mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * step
            
            # hsig
            # We need the expected norm of pc.
            # pc is updated as (1-c_c)pc + c_c * step/sigma? No, pc is updated with (new_mean - mean)/sigma.
            # So pc is in "sigma units".
            # The expected norm of pc after update is approximately sqrt(1 - (1-c_c)**2) * chi_n?
            # Actually, the standard check is:
            # hsig = np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(2 * gen)) / chi_n < 1.4 + 2/(n+1)
            # We don't have gen tracked easily in this simple loop structure without a counter.
            # Let's just use a simpler sigma adaptation that is robust enough:
            # sigma *= exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Prevent sigma from exploding or vanishing
            if sigma > 10.0: sigma = 10.0
            if sigma < 1e-6: sigma = 1e-6
            
            mean = new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
