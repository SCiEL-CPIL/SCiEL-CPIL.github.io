"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        # Deadline guard: 300s limit, use 290s for safety
        deadline = start_time + 290
        d = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls
            if calls >= budget or not time_ok():
                return 1e10
            calls += 1
            try:
                val = func(x)
                if np.isnan(val) or np.isinf(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Initial population size
        # For dim=5, lambda = 4 + floor(3 * log(5)) = 4 + 4 = 8 is standard.
        # But we have a fixed budget. 
        # Let's use a population size that allows for several generations.
        # If lambda=8, and we have 500 calls, we can do ~60 generations.
        # This is good for CMA-ES convergence.
        
        # However, CMA-ES needs a good starting center and sigma.
        # The Discus function has a narrow valley. 
        # Random start at 0 is risky if the shift is large.
        # But CMA-ES is robust.
        
        # Let's do a quick initial exploration to find a better center.
        # Phase 1: Coarse Grid / Random Search to find a decent center
        # We spend 50 calls on this.
        
        n_initial = 50
        initial_xs = []
        initial_fs = []
        
        # Center
        x0 = np.zeros(d)
        f0 = wrapped_func(x0)
        initial_xs.append(x0)
        initial_fs.append(f0)
        
        # Random points
        for _ in range(n_initial - 1):
            x = rng.uniform(-5, 5, d)
            f = wrapped_func(x)
            initial_xs.append(x)
            initial_fs.append(f)
            
        # Find best initial point
        min_idx = np.argmin(initial_fs)
        x_start = initial_xs[min_idx].copy()
        best_f = initial_fs[min_idx]
        best_x = x_start.copy()
        
        if not time_ok() or calls >= budget:
            return best_x, best_f

        # Phase 2: CMA-ES
        # We will implement a simplified CMA-ES loop manually to control the budget strictly.
        # Standard CMA-ES uses a population.
        
        lambda_pop = 8 # Population size
        # Initial sigma: heuristic based on dimension and box size
        # The valley is narrow, so we might want a smaller sigma if we are close, 
        # but we don't know the scale yet. 
        # A reasonable start is sigma = 1.0 or 0.5 * (box_size / dim) ?
        # Box size is 10. 10/5 = 2. Let's try sigma = 1.0.
        sigma = 1.0
        
        # CMA-ES parameters
        # Step sizes for mean and covariance
        # mu = lambda // 2 (number of parents)
        mu = lambda_pop // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        cs = (mu_eff + 2) / (d + mu_eff + 5)
        c1 = 2 / ((d + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # State variables
        mean = x_start.copy()
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        inv_sqrtnC = 1.0 / np.sqrt(np.trace(C))
        
        # History of sigma for adaptive restart? No, keep it simple.
        # We will run until budget is exhausted or time runs out.
        
        gen = 0
        while calls < budget and time_ok():
            # Generate population
            xs = np.zeros((lambda_pop, d))
            for i in range(lambda_pop):
                # Sample from N(mean, sigma^2 * C)
                # We need to sample from multivariate normal with covariance sigma^2 * C
                # Use Cholesky decomposition for efficiency, but d=5 is small.
                # np.random.multivariate_normal is easy but might be slow if called many times?
                # It's fine for d=5.
                try:
                    z = rng.standard_normal(d)
                    # Transform z to have covariance C
                    # C = L L^T
                    # x = mean + sigma * L @ z
                    # To avoid explicit Cholesky every time, we can update it, 
                    # but for d=5, recomputing is cheap.
                    L = np.linalg.cholesky(C)
                    xs[i] = mean + sigma * (L @ z)
                except Exception:
                    xs[i] = mean + sigma * rng.standard_normal(d)
                    
                # Clip to box
                xs[i] = np.clip(xs[i], -5.0, 5.0)
                
            # Evaluate population
            fs = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if calls >= budget or not time_ok():
                    break
                fs[i] = wrapped_func(xs[i])
                
            # If we ran out of budget during evaluation, break
            if calls >= budget or not time_ok():
                break
                
            # Update best
            min_idx = np.argmin(fs)
            if fs[min_idx] < best_f:
                best_f = fs[min_idx]
                best_x = xs[min_idx].copy()
                
            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Update mean
            # Weighted sum of top mu solutions
            w = weights
            mean_new = np.sum(w[:, np.newaxis] * xs_sorted[:mu], axis=0)
            
            # Update evolution paths
            # pc = (1 - cc) * pc + cc * sqrt(mu_eff) * inv_sqrtnC * (mean_new - mean) / sigma
            delta_mean = (mean_new - mean) / sigma
            pc = (1 - cc) * pc + cc * np.sqrt(mu_eff) * inv_sqrtnC * delta_mean
            
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * inv_sqrtnC * (mean_new - mean) / sigma
            # Note: The formula for ps usually involves the normalized step
            # Let's stick to standard CMA-ES formulas
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) * (mean_new - mean) / (sigma * inv_sqrtnC)
            # Wait, inv_sqrtnC is 1/sqrt(trace(C)).
            # Standard: ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) / (sigma * sqrt(trace(C))) * (mean_new - mean)
            # This is equivalent to: ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * inv_sqrtnC * (mean_new - mean) / sigma
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * inv_sqrtnC * delta_mean
            
            # Check hsig
            hsig = np.sqrt((1 - (1 - cs)**(2 * (gen + 1))) / (1 - (1 - cs)**(2 * (gen + 1)))) # Approx 1 if cs small
            # Better: hsig = 1 if norm(ps)/sqrt(1-(1-cs)^(2*(gen+1))) < (1.4 + 2/(d+1)) * sqrt((1-cc)/(1+cc))
            # For simplicity and stability, let's just update C if the norm is reasonable.
            # Actually, the standard check is:
            # hsig = 1 if norm(ps) / sqrt(1 - (1-cs)^(2*(gen+1))) < (1.4 + 2/(d+1)) * sqrt((1-cc)/(1+cc)) else 0
            
            denom = np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
            if denom < 1e-10:
                hsig = 0
            else:
                threshold = (1.4 + 2 / (d + 1)) * np.sqrt((1 - cc) / (1 + cc))
                hsig = 1 if np.linalg.norm(ps) / denom < threshold else 0
            
            # Update covariance matrix C
            # C = (1 - mu_eff * (c1 + cmu)) * C + ...
            # Rank-1 update from pc
            # Rank-mu update from population
            
            # Calculate hsig term for pc
            # If hsig=1, pc contributes.
            
            # Update C
            # Term 1: (1 - mu_eff * (c1 + cmu)) * C
            # Term 2: c1 * ( (1 + (2*hsig-1)*cc*(1-cc)) * outer(pc, pc) - (1 - cc) * C )
            # Term 3: cmu * sum(weights * outer(x_i - mean, x_i - mean)) / sigma^2
            
            # Let's implement carefully
            # C_new = (1 - mu_eff * (c1 + cmu)) * C
            C_new = (1 - mu_eff * (c1 + cmu)) * C
            
            # Rank-1 update
            # c1 * ( (1 + (2*hsig-1)*cc*(1-cc)) * outer(pc, pc) - (1 - cc) * C )
            rank1_factor = (1 + (2 * hsig - 1) * cc * (1 - cc))
            C_new += c1 * (rank1_factor * np.outer(pc, pc) - (1 - cc) * C)
            
            # Rank-mu update
            # cmu * sum(w_i * outer(x_i - mean, x_i - mean)) / sigma^2
            # Note: x_i - mean is the step.
            # The covariance of the steps is sigma^2 * C.
            # So we want to add the observed covariance of the steps, scaled.
            
            # Sum of outer products
            rank_mu_sum = np.zeros((d, d))
            for i in range(mu):
                diff = (xs_sorted[i] - mean) / sigma
                rank_mu_sum += w[i] * np.outer(diff, diff)
                
            C_new += cmu * rank_mu_sum
            
            # Symmetrize
            C_new = (C_new + C_new.T) / 2
            
            # Update mean
            mean = mean_new
            
            # Update sigma
            # sigma = sigma * exp(cs / dsigma * (norm(ps) / chiN - 1))
            # chiN is expected norm of standard normal
            chiN = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            
            # dsigma: usually 1 / sqrt(mu_eff) or similar
            # Standard: dsigma = 1 / (sqrt(mu_eff) * (1 + cs))
            dsigma = 1 / (np.sqrt(mu_eff) * (1 + cs))
            
            sigma = sigma * np.exp(cs / dsigma * (np.linalg.norm(ps) / chiN - 1))
            
            # Prevent sigma from exploding or vanishing
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update C
            C = C_new
            
            # Update inv_sqrtnC
            inv_sqrtnC = 1.0 / np.sqrt(np.trace(C))
            
            gen += 1
            
            # Optional: Restart if sigma is too small or population stuck?
            # For now, just let it run.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
