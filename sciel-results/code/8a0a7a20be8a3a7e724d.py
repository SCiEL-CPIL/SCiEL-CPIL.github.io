import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 290

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        evals_used = 0
        
        # Island Model CMA-ES
        n_islands = 4
        pop_size = 6 # 1.5 * (4 * dim) roughly, but small for budget
        
        # Initialize islands
        islands = []
        for i in range(n_islands):
            x = rng.uniform(-4, 4, self.dim)
            fx = func(x)
            evals_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # CMA-ES state per island
            # sigma starts moderate
            sigma = 1.0
            # Covariance matrix C starts as identity
            C = np.eye(self.dim)
            # Inverse of C
            C_inv = np.eye(self.dim)
            # Eigen decomposition for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            
            island = {
                'x': x,
                'fx': fx,
                'sigma': sigma,
                'C': C,
                'C_inv': C_inv,
                'eigvals': eigvals,
                'eigvecs': eigvecs,
                'p_sigma': np.zeros(self.dim),
                'p_c': np.zeros(self.dim),
                'no_improve': 0
            }
            islands.append(island)

        # CMA-ES parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma

        max_no_improve = 20
        migration_interval = 10 # Migrate every 10 iterations
        
        gen = 0
        
        while evals_used < self.budget:
            if time.time() > self.deadline:
                break
            
            # Check if we can afford a full population update
            if evals_used + pop_size > self.budget:
                break
                
            for i in range(n_islands):
                if evals_used + pop_size > self.budget:
                    break
                if time.time() > self.deadline:
                    break
                    
                isl = islands[i]
                x_mean = isl['x']
                sigma = isl['sigma']
                C = isl['C']
                eigvals = isl['eigvals']
                eigvecs = isl['eigvecs']
                
                # Sample offspring
                offspring = []
                for j in range(pop_size):
                    z = rng.standard_normal(self.dim)
                    # Sample y ~ N(0, C)
                    y = eigvecs @ (np.sqrt(eigvals) * z)
                    x_off = x_mean + sigma * y
                    x_off = np.clip(x_off, -5, 5)
                    offspring.append(x_off)
                
                # Evaluate offspring
                f_offspring = np.array([func(x) for x in offspring])
                evals_used += pop_size
                
                # Update best
                min_idx = np.argmin(f_offspring)
                min_f = f_offspring[min_idx]
                min_x = offspring[min_idx]
                
                if min_f < best_f:
                    best_f = min_f
                    best_x = min_x.copy()
                
                # Selection
                order = np.argsort(f_offspring)
                selected = [offspring[order[j]] for j in range(mu)]
                selected_f = [f_offspring[order[j]] for j in range(mu)]
                
                # Weighted mean
                x_new = np.sum([w * x for w, x in zip(weights, selected)], axis=0)
                
                # Update step size control
                y_old = (x_new - x_mean) / sigma
                p_sigma = (1 - c_sigma) * isl['p_sigma'] + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_old
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # Update covariance matrix
                # Check h_sigma
                norm_p_sigma = np.linalg.norm(p_sigma)
                norm_p_c = np.linalg.norm(isl['p_c'])
                chi_mu = np.sqrt(mu_eff) * (1 + 1 / (2 * mu_eff) + 1 / (8 * mu_eff**2))
                h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (self.dim + 1)) * chi_mu else 0.0
                
                p_c = (1 - c_c) * isl['p_c'] + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_old
                
                # Rank-one update
                rank1 = np.outer(y_old, y_old)
                # Rank-mu update
                rank_mu = np.zeros((self.dim, self.dim))
                for j in range(mu):
                    y_j = (selected[j] - x_mean) / sigma
                    rank_mu += weights[j] * np.outer(y_j, y_j)
                
                C_new = (1 + c_1 * (h_sigma - 1) + c_mu) * C - c_1 * np.outer(p_c, p_c) - c_mu * rank_mu
                
                # Symmetrize and ensure positive definiteness
                C_new = (C_new + C_new.T) / 2.0
                try:
                    eigvals_new, eigvecs_new = np.linalg.eigh(C_new)
                    # Clip eigenvalues to be positive
                    eigvals_new = np.maximum(eigvals_new, 1e-12)
                    C_new = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T
                    C_new = (C_new + C_new.T) / 2.0
                except:
                    C_new = C
                    eigvals_new = eigvals
                    eigvecs_new = eigvecs
                
                # Update island state
                isl['x'] = x_new
                isl['sigma'] = sigma
                isl['C'] = C_new
                isl['eigvals'] = eigvals_new
                isl['eigvecs'] = eigvecs_new
                isl['p_sigma'] = p_sigma
                isl['p_c'] = p_c
                
                if min_f < isl['fx']:
                    isl['fx'] = min_f
                    isl['no_improve'] = 0
                else:
                    isl['no_improve'] += 1

            gen += 1
            
            # Migration: 'diff' operator test
            if gen % migration_interval == 0:
                # Sort islands by fitness
                sorted_islands = sorted(islands, key=lambda x: x['fx'])
                # Migrate best from best island to worst island
                best_isl = sorted_islands[0]
                worst_isl = sorted_islands[-1]
                
                # Replace worst island's center with best island's center
                # This is a form of migration.
                worst_isl['x'] = best_isl['x'].copy()
                # Keep its own C and sigma to allow exploration around the new center
                # Or reset C to identity? Let's keep C for now.
                
                # The "diff" aspect: We are moving the best solution to the worst island.
                # This changes the chance that a child (offspring) improves the worst island.
                # If the worst island was stuck, this gives it a fresh start.
                # If it wasn't, it might disrupt a good lineage.
                
                # Also, we can perform a "diff" between two islands to create a new direction?
                # For now, simple migration.

            # Restart if stagnation
            for i in range(n_islands):
                if islands[i]['no_improve'] > max_no_improve:
                    # Reset island
                    x = rng.uniform(-4, 4, self.dim)
                    fx = func(x)
                    evals_used += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    
                    islands[i]['x'] = x
                    islands[i]['fx'] = fx
                    islands[i]['sigma'] = 1.0
                    islands[i]['C'] = np.eye(self.dim)
                    eigvals, eigvecs = np.linalg.eigh(np.eye(self.dim))
                    islands[i]['eigvals'] = eigvals
                    islands[i]['eigvecs'] = eigvecs
                    islands[i]['p_sigma'] = np.zeros(self.dim)
                    islands[i]['p_c'] = np.zeros(self.dim)
                    islands[i]['no_improve'] = 0

        return best_x, best_f
