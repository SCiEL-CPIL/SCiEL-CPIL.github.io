import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        self.calls = 0
        self.best_x = np.zeros(dim)
        self.best_f = np.inf
        self.deadline = time.time() + 285
        
        def eval_func(x):
            if self.calls >= budget or time.time() > self.deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f
        
        # CMA-ES parameters
        mu = dim // 2 + 1  # number of parents
        lambda_ = 4 * dim  # larger population for better exploration
        cc = (4 + mu/dim) / (dim + 4 + 2*mu/dim)
        cs = (mu + 2) / (dim + mu + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu)
        cmu = min(0.5 - c1, 2 * ((mu - 2 + 1/mu) / ((dim + 2)**2 + mu)))
        damps = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1) - 1)) + cs
        
        # Add Nelder-Mead polish at the end if budget allows
        def nelder_mead_polish(func, x0, budget_remaining, deadline):
            dim = len(x0)
            calls_used = 0
            best_x = x0.copy()
            best_f = float(func(x0))
            calls_used += 1
            
            # Initialize simplex
            simplex = [x0.copy()]
            f_values = [best_f]
            
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.1
                simplex.append(xi)
                fi = float(func(xi))
                f_values.append(fi)
                calls_used += 1
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            
            while calls_used < budget_remaining and time.time() < deadline:
                # Sort simplex by function values
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                if calls_used >= budget_remaining:
                    break
                
                x_worst = simplex[-1]
                f_worst = f_values[-1]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_reflect = x_centroid + alpha * (x_centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = float(func(x_reflect))
                calls_used += 1
                if f_reflect < best_f:
                    best_f = f_reflect
                    best_x = x_reflect.copy()
                
                if f_reflect < f_values[0]:
                    # Expansion
                    x_expand = x_centroid + gamma * (x_reflect - x_worst)
                    x_expand = np.clip(x_expand, -5, 5)
                    if calls_used < budget_remaining:
                        f_expand = float(func(x_expand))
                        calls_used += 1
                        if f_expand < best_f:
                            best_f = f_expand
                            best_x = x_expand.copy()
                        if f_expand < f_reflect:
                            simplex[-1] = x_expand
                            f_values[-1] = f_expand
                        else:
                            simplex[-1] = x_reflect
                            f_values[-1] = f_reflect
                    else:
                        simplex[-1] = x_reflect
                        f_values[-1] = f_reflect
                elif f_reflect < f_values[-2]:
                    simplex[-1] = x_reflect
                    f_values[-1] = f_reflect
                else:
                    # Contraction
                    x_contract = x_centroid + rho * (x_worst - x_centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    if calls_used < budget_remaining:
                        f_contract = float(func(x_contract))
                        calls_used += 1
                        if f_contract < best_f:
                            best_f = f_contract
                            best_x = x_contract.copy()
                        if f_contract < f_worst:
                            simplex[-1] = x_contract
                            f_values[-1] = f_contract
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                if calls_used < budget_remaining:
                                    f_values[i] = float(func(simplex[i]))
                                    calls_used += 1
                                    if f_values[i] < best_f:
                                        best_f = f_values[i]
                                        best_x = simplex[i].copy()
                    else:
                        simplex[-1] = x_contract
                        f_values[-1] = f_contract
                
                # Check for convergence
                if max(f_values) - min(f_values) < 1e-10:
                    break
            
            return best_x, best_f, calls_used
        
        # Precompute weights
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 0.5) - np.log(i + 1)
        w /= w.sum()
        mu_w = (w.sum() ** 2) / (w ** 2).sum()
        
        # Phase 1: Initial exploration with random restarts (first 10% of budget)
        explore_budget = max(10, budget // 10)
        np.random.seed(42)
        for _ in range(explore_budget):
            if self.calls >= budget or time.time() > self.deadline:
                break
            x_trial = np.random.uniform(-5, 5, size=dim)
            eval_func(x_trial)
        
        # Phase 2: CMA-ES optimization
        sigma = 2.0  # larger initial step for better exploration
        x_mean = self.best_x.copy()
        C = np.eye(dim)
        L = np.linalg.cholesky(C)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Calculate max generations based on remaining budget
        remaining = budget - self.calls
        max_generations = max(1, remaining // lambda_)
        
        for gen in range(max_generations):
            if self.calls >= budget or time.time() > self.deadline:
                break
            
            # Check for stagnation - restart if no improvement in last 3 generations
            if gen > 0 and gen % 3 == 0:
                # Simple stagnation check: if sigma is very small or very large, restart
                if sigma < 1e-6 or sigma > 5.0:
                    x_mean = np.random.uniform(-4, 4, size=dim)
                    sigma = 2.0
                    C = np.eye(dim)
                    L = np.linalg.cholesky(C)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    continue
            
            # Generate population
            offspring = []
            for i in range(lambda_):
                if self.calls >= budget or time.time() > self.deadline:
                    break
                z = np.random.randn(dim)
                x_new = x_mean + sigma * (L @ z)
                x_new = np.clip(x_new, -5, 5)
                f = eval_func(x_new)
                if np.isfinite(f):
                    offspring.append((f, x_new))
            
            if len(offspring) < mu:
                break
            
            # Sort by fitness
            offspring.sort(key=lambda x: x[0])
            
            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += w[i] * offspring[i][1]
            
            # Update paths
            y_vec = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_w) * y_vec
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_w) * y_vec
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            norm_ones = np.sqrt(dim)
            sigma *= np.exp(cs / damps * (norm_p_sigma / norm_ones - 1))
            sigma = np.clip(sigma, 1e-8, 5.0)
            
            # Update covariance
            delta_sigma = (1 + cs / damps) * (norm_p_sigma / norm_ones - 1)
            C = (1 + c1 * delta_sigma**2 - c1 - cmu) * C + c1 * np.outer(p_c, p_c)
            for i in range(mu):
                diff = (offspring[i][1] - x_mean) / sigma
                C += cmu * w[i] * np.outer(diff, diff)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except:
                C = np.eye(dim)
                L = np.eye(dim)
                sigma = 1.0
            
            x_mean = x_new_mean
        
        # Phase 3: Nelder-Mead polish with remaining budget
        remaining_budget = budget - self.calls
        if remaining_budget > 20:
            polish_x, polish_f, polish_calls = nelder_mead_polish(
                func, self.best_x, remaining_budget, self.deadline
            )
            if polish_f < self.best_f:
                self.best_x = polish_x
                self.best_f = polish_f
            self.calls += polish_calls
        
        return self.best_x, self.best_f
