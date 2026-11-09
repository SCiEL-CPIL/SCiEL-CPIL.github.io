import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Time management
        start_time = time.time()
        time_limit = 320.0

        # CMA-ES Parameters
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(n)))
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Stagnation threshold for restart
        restart_threshold = 15 # Generations without improvement
        
        # Initialize
        # Start with a diverse set of points to find a good basin
        # But CMA-ES needs a single mean. 
        # Strategy: Use a small number of random starts to find a good initial mean, then run CMA-ES.
        # Or, run CMA-ES starting from origin, but with a large initial sigma.
        
        # Given the hidden shift/rotation, origin might be far.
        # Let's use a "Multi-Start CMA-ES" approach:
        # 1. Spend 20% of budget on random sampling to find a good initial point.
        # 2. Run CMA-ES from that point.
        # 3. If stagnant, restart CMA-ES from a new random point or perturb the best.
        
        # However, 300 calls is small. 20% is 60 calls.
        # Let's do 10 random points first.
        
        x_best = None
        f_best = np.inf
        eval_count = 0
        
        # Phase 1: Global Exploration (Random Sampling)
        # Find a good starting point
        num_initial = 10
        for i in range(num_initial):
            if eval_count >= budget or time.time() - start_time > time_limit:
                break
            x = np.random.uniform(-5, 5, n)
            try:
                fx = func(x)
            except Exception:
                continue
            eval_count += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
        
        if x_best is None:
             # Fallback if all failed
             x_best = np.zeros(n)
             f_best = func(x_best)
             eval_count += 1

        # Phase 2: CMA-ES Optimization
        # Initialize CMA-ES state
        mean = x_best.copy()
        sigma = 2.0 # Initial step size, somewhat large to explore local basin
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Track stagnation
        best_so_far = f_best
        stagnant_gens = 0
        
        # Main CMA-ES Loop
        while eval_count < budget and time.time() - start_time < time_limit:
            # Generate population
            pop = []
            for i in range(lam):
                if eval_count >= budget or time.time() - start_time > time_limit:
                    break
                # Sample from N(mean, sigma^2 * C)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C is not positive definite
                    L = np.eye(n)
                
                z = np.random.standard_normal(n)
                x = mean + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                
                try:
                    fx = func(x)
                except Exception:
                    fx = np.inf
                
                eval_count += 1
                pop.append((fx, x))
            
            if not pop:
                break
                
            # Sort population by fitness
            pop.sort(key=lambda p: p[0])
            
            # Update best
            if pop[0][0] < f_best:
                f_best = pop[0][0]
                x_best = pop[0][1].copy()
                stagnant_gens = 0
            else:
                stagnant_gens += 1
            
            # Check for stagnation and restart
            if stagnant_gens >= restart_threshold:
                # Restart strategy:
                # 1. Perturb the mean towards the best point found in this run
                # 2. Increase sigma to explore more
                # 3. Reset covariance to identity (or damp it)
                mean = x_best.copy()
                sigma *= 2.0
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                stagnant_gens = 0
                
                # Optional: If we've been stuck for a long time, try a random restart
                # But we only have 300 calls. Let's stick to local restarts first.
                # If eval_count < budget * 0.8, we can afford a full random restart sometimes.
                if eval_count < budget * 0.7 and np.random.random() < 0.5:
                    # Full random restart
                    x = np.random.uniform(-5, 5, n)
                    try:
                        fx = func(x)
                        eval_count += 1
                        if fx < f_best:
                            f_best = fx
                            x_best = x.copy()
                            mean = x_best.copy()
                    except Exception:
                        pass
                    sigma = 2.0
                    C = np.eye(n)
                    pc = np.zeros(n)
                    ps = np.zeros(n)
                    stagnant_gens = 0
                    continue # Skip CMA-ES update for this generation

            # CMA-ES Update Step
            
            # Weighted mean of best mu points
            w_mean = np.zeros(n)
            for i in range(mu):
                w_mean += weights[i] * pop[i][1]
            
            # Update path cumulation for covariance matrix
            # pc = (1 - c_c) * pc + c_c * (sigma / sqrt(c_c * (2 - c_c))) * (w_mean - mean)
            # Note: The term (w_mean - mean) is the normalized step
            # Actually, standard CMA-ES uses:
            # pc = (1-c_c)*pc + c_c/sqrt(c_c*(2-c_c)) * (1/sigma) * (w_mean - mean) * sigma ?
            # Let's use the standard update:
            # pc = (1-c_c)*pc + c_c * (w_mean - mean) / sigma
            
            # Correction: The path vector ps is for sigma adaptation, pc is for covariance.
            # ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff)/sigma * (w_mean - mean)
            # pc = (1-c_c)*pc + c_c * (w_mean - mean) / sigma
            
            # Let's verify the standard formulas.
            # From Hansen's CMA-ES tutorial:
            # pc <- (1-c_c)pc + c_c/sqrt(c_c(2-c_c)) * (x_w - m)/sigma
            # ps <- (1-c_sigma)ps + sqrt(c_sigma(2-c_sigma)mu_eff)/sigma * (x_w - m)
            
            x_w = w_mean
            m = mean
            
            # Update pc
            pc = (1 - c_c) * pc + c_c / np.sqrt(c_c * (2 - c_c)) * (x_w - m) / sigma
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (x_w - m)
            
            # Update sigma
            h_sig = min(1, (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((2 * eval_count // lam + 1)))) / np.sqrt(n) * (1 + c_sigma))
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Update covariance matrix C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - h_sig) * c_c * (2 - c_c) * I) + c_mu * sum w_i (x_i - m)(x_i - m).T
            
            # Rank-one update part
            delta_hsig = (1 - h_sig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_hsig * np.eye(n))
            
            # Rank-mu update part
            for i in range(mu):
                diff = pop[i][1] - m
                C += c_mu * weights[i] * np.outer(diff, diff)
            
            # Update mean
            mean = x_w
            
            # Ensure C is symmetric and positive definite (numerical stability)
            C = (C + C.T) / 2.0
            # Eigenvalue check/clamping
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 1e-10):
                    # Add small diagonal
                    C += 1e-10 * np.eye(n)
            except Exception:
                C = np.eye(n)
                
        return x_best, f_best
