import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        eval_count = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES for global optimization
        # Initialize around origin with reasonable spread
        x = np.zeros(d)
        fx = evaluate(x)
        
        # CMA-ES parameters
        lambda_ = 4 * d  # Population size
        mu = lambda_ // 2  # Number of parents
        sigma = 2.0  # Initial step size
        cc = (4 + d / (lambda_ + 4)) / ((d + 4) + (lambda_ + 1))
        cs = (lambda_ + 2 + 2 / (d + 1)) / ((d + lambda_ + 5) + 2 / (d + 1))
        cmu = 2 * (mu_eff - 1 + 2 / (mu_eff + 1)) / ((d + 2) ** 2 + mu_eff - 2)
        mu_eff = sum(w for w in np.log(np.arange(1, mu + 1)) / np.sum(np.log(np.arange(1, mu + 1))))
        
        # Cumulation paths
        ps = np.zeros(d)
        pc = np.zeros(d)
        C = np.eye(d)
        C_inv_sqrt = np.eye(d)
        
        # Weights
        weights = np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        weights = weights[:mu]
        
        # Initial population
        population = [x.copy()]
        fitnesses = [fx]
        
        # Run CMA-ES for most of the budget
        cma_budget = int(budget * 0.85)
        
        while eval_count < cma_budget:
            if time.time() > deadline:
                break
            
            # Generate population
            new_population = []
            new_fitnesses = []
            
            for i in range(lambda_):
                if eval_count >= cma_budget:
                    break
                
                # Sample from multivariate normal
                z = rng.standard_normal(d)
                
                # Apply covariance transformation
                # Using Cholesky-like approach for simplicity
                # In practice, we'd maintain C^(1/2) properly
                y = z @ C_inv_sqrt
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                
                fi = evaluate(xi)
                new_population.append(xi)
                new_fitnesses.append(fi)
            
            if len(new_population) < 2:
                break
            
            # Sort by fitness
            idx = np.argsort(new_fitnesses)
            sorted_pop = [new_population[i] for i in idx]
            sorted_fit = [new_fitnesses[i] for i in idx]
            
            # Update mean
            x_mean_new = sum(weights[i] * sorted_pop[i] for i in range(mu))
            
            # Update step size (CSA)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - 2 * cs) ** ((eval_count // lambda_) + 1)) < (1 + 2 / (d + 1)) * np.sqrt((d + 1) / (d - 1))
            hsig = 1.0 if hsig else 0.0
            
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (x_mean_new - x)
            sigma *= np.exp(cs / d * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update covariance matrix (rank-mu update)
            if hsig:
                pc = (1 - cc) * pc + cc * (x_mean_new - x) / sigma
            
            deltaD = cc * (1 - cc) * (np.linalg.norm(pc) ** 2 / d - 1)
            C = (1 + cmu * deltaD) * C
            for i in range(mu):
                zi = (sorted_pop[i] - x) / sigma
                C += cmu * weights[i] * np.outer(zi, zi)
            
            # Compute C_inv_sqrt via eigenvalue decomposition
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-10, None)
                C_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            except:
                C_inv_sqrt = np.eye(d)
            
            x = x_mean_new
        
        # Phase 2: Nelder-Mead for local refinement
        center = best_x.copy() if best_x is not None else np.zeros(d)
        simplex = [(center.copy(), best_f)]
        
        step = max(0.05, sigma * 0.3)
        
        for i in range(d):
            if time.time() > deadline or eval_count >= budget - 1:
                break
            pt = center.copy()
            pt[i] += step
            fval = evaluate(pt)
            simplex.append((pt.copy(), fval))
        
        simplex.sort(key=lambda p: p[1])
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        
        nm_iterations = 0
        max_nm_iter = 200
        
        while eval_count < budget - 1 and nm_iterations < max_nm_iter:
            if time.time() > deadline:
                break
            nm_iterations += 1
            
            x_worst, f_worst = simplex[-1]
            x_second, f_second = simplex[-2]
            
            x_centroid = np.mean([p[0] for p in simplex[:-1]], axis=0)
            
            x_ref = x_centroid + alpha * (x_centroid - x_worst)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = evaluate(x_ref)
            
            if f_ref < simplex[0][1]:
                if eval_count >= budget - 1:
                    break
                x_exp = x_centroid + gamma * (x_centroid - x_worst)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = evaluate(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = (x_exp, f_exp)
                else:
                    simplex[-1] = (x_ref, f_ref)
            elif f_ref < f_second:
                simplex[-1] = (x_ref, f_ref)
            else:
                if f_ref < f_worst:
                    x_con = x_centroid + 0.5 * (x_ref - x_centroid)
                else:
                    x_con = x_centroid - 0.5 * (x_worst - x_centroid)
                x_con = np.clip(x_con, -5, 5)
                
                if eval_count >= budget - 1:
                    break
                f_con = evaluate(x_con)
                
                if f_con < min(f_ref, f_worst):
                    simplex[-1] = (x_con, f_con)
                else:
                    x_best, f_best_cur = simplex[0]
                    new_simplex = []
                    for i, (x_i, f_i) in enumerate(simplex):
                        if i == 0:
                            new_simplex.append((x_i, f_i))
                        else:
                            if eval_count >= budget - 1:
                                break
                            x_shrink = x_best + rho * (x_i - x_best)
                            x_shrink = np.clip(x_shrink, -5, 5)
                            f_shrink = evaluate(x_shrink)
                            new_simplex.append((x_shrink, f_shrink))
                    if len(new_simplex) == len(simplex):
                        simplex = new_simplex
                    else:
                        break
            
            simplex.sort(key=lambda p: p[1])
            
            x_min = np.min([p[0] for p in simplex], axis=0)
            x_max = np.max([p[0] for p in simplex], axis=0)
            if np.max(x_max - x_min) < 1e-10:
                break
        
        return best_x, best_f
