import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        n = self.dim
        best_x = None
        best_f = np.inf
        
        # We will run multiple CMA-ES instances (restarts) if needed
        # But with 500 budget, one good run is better.
        # Let's use a single run with a large population to ensure exploration.
        
        lambda_ = 15
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        rng = np.random.default_rng(0)
        
        # Initial state
        mean = rng.uniform(-4, 4, n)
        C = np.eye(n)
        sigma = 2.0
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt(mu_eff/n - 1)) + c_sigma
        c_c = (4 + mu_eff/n) / (n + 4 + 2*mu_eff/n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (np.sqrt(mu_eff/n - 1) + 1) / ((n + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        calls = 0
        no_improve_count = 0
        last_best_f = np.inf
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Check for stagnation/restart
            if no_improve_count > 20: # 20 generations without improvement
                # Restart with new center
                mean = rng.uniform(-4, 4, n)
                C = np.eye(n)
                sigma = 2.0
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                no_improve_count = 0
                # Consume a few evaluations to re-evaluate? No, just reset state.
                # The next loop iteration will evaluate new population.
                continue
            
            # Generate population
            pop = []
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize C
                C += np.eye(n) * 1e-6
                L = np.linalg.cholesky(C)
            
            for i in range(lambda_):
                z = rng.standard_normal(n)
                x_new = mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                pop.append(x_new)
            
            # Evaluate
            f_vals = []
            for x in pop:
                if calls >= self.budget:
                    break
                f_vals.append(func(x))
                calls += 1
            
            if len(f_vals) < lambda_:
                # Pad with infinity if we ran out of budget mid-population
                while len(f_vals) < lambda_:
                    f_vals.append(np.inf)
                    pop.append(pop[-1]) # Dummy
            
            # Sort
            order = np.argsort(f_vals)
            sorted_pop = [pop[i] for i in order]
            sorted_f = [f_vals[i] for i in order]
            
            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_pop[0]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update mean
            new_mean = np.sum([weights[i] * sorted_pop[i] for i in range(mu)], axis=0)
            
            # Update paths
            try:
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ (new_mean - mean) / sigma)
            
            chi_n = np.sqrt(n) * (1 + 1/(2*n) + 1/(8*n*n))
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls / lambda_))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * (new_mean - mean) / sigma
            
            # Update covariance
            v = (new_mean - mean) / sigma
            C = (1 + c_1 * (h_sigma * (1 + (1 - c_sigma) * c_sigma) - c_1) - c_mu) * C \
                + c_1 * (np.outer(v, v) + (1 - h_sigma) * c_sigma * (1 - c_sigma) * C)
            
            y = np.array([(sorted_pop[i] - mean) / sigma for i in range(mu)])
            C += c_mu * np.sum([weights[i] * np.outer(y[i], y[i]) for i in range(mu)], axis=0)
            
            C = (C + C.T) / 2
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            mean = new_mean
            
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1
            
        return best_x, best_f
