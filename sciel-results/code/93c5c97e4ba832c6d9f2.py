import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim)) # 4 + 3*1.6 = 8
        pop_size = max(pop_size, 2)
        
        # Initial state
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0
        sigma_init = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Adaptation parameters (standard CMA-ES)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (weights[0] - 0.5) / ((dim + 2)**2 + mu_eff))
        
        dnorm = np.sqrt(dim) * (1 - 1. / (4.*dim) - 1. / (21.*dim**2))
        
        best_x = x.copy()
        best_f = func(x)
        calls_made = 1
        
        no_improve_count = 0
        max_no_improve = 15 # Restart after 15 generations without improvement
        
        while calls_made < budget:
            # Check time limit
            if time.time() > deadline:
                break
            
            # Generate offspring
            ys = []
            for i in range(pop_size):
                # Sample from multivariate normal
                # Use Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim) + 1e-10 * np.eye(dim)
                    L = np.linalg.cholesky(C)
                
                z = rng.standard_normal(dim)
                y = x + sigma * L @ z
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            # Evaluate offspring
            fs = []
            for y in ys:
                if time.time() > deadline:
                    break
                fy = func(y)
                calls_made += 1
                fs.append(fy)
            
            if len(fs) < pop_size:
                # Pad if we ran out of time mid-generation
                while len(fs) < pop_size:
                    fs.append(float('inf'))
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Check for stagnation and restart
            if no_improve_count >= max_no_improve:
                # Restart from a new random point or perturbed current best
                # Strategy: 50% chance new random, 50% chance large perturbation of best
                if rng.random() < 0.5:
                    x = rng.uniform(-5, 5, dim)
                else:
                    x = best_x + 2.0 * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                
                sigma = sigma_init
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                no_improve_count = 0
                # Note: We do NOT increment calls_made here, but we have already used the budget for the generation.
                # The restart happens at the end of the generation.
                
                # We should probably re-evaluate the new x to update best if it's better?
                # No, best_x is already tracked. The new x is just the mean for the next generation.
                # However, if the new x is better than best_x, we should update.
                # But func(x) consumes a call. Let's just let the next generation handle it.
                # Actually, it's safer to just proceed.
                
            # Update CMA-ES parameters (if not restarted in this iteration, or even if restarted, we update based on this gen)
            # Standard CMA-ES update
            xnew = sum(weights[i] * ys[i] for i in range(mu))
            
            # Update path evolution strategies
            pc = (1 - cc) * pc + cc * (xnew - x) / sigma
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc)**(2 * (calls_made - pop_size)))) < (1.4 + 2 / (dim + 1)) * cs * dnorm
            ps = (1 - cs) * ps + float(hsig) * np.sqrt(cs * (2 - cs) * mu_eff) * (xnew - x) / sigma
            
            # Update covariance matrix
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) + (delta_hsig - 1 + cc * (2 - cc)) * np.eye(dim)) \
                + cmu * sum(weights[i] * np.outer(ys[i] - x, ys[i] - x) for i in range(mu))
            
            # Update step size (Sigma update)
            sigma = sigma * np.exp(cs / dnorm * (np.linalg.norm(ps) / dnorm - 1))
            sigma = np.clip(sigma, 1e-12, 100.0)
            
            x = xnew

        return best_x, best_f
