"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid LHS + CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320  # Safety margin before 330s kill
        n = self.dim
        B = self.budget
        
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f, calls
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Global Exploration via Latin Hypercube Sampling
        # LHS ensures good coverage of the space, helping to find the basin of attraction early.
        # For n=5, 40 points is a good trade-off between coverage and budget usage.
        n_lhs = 40
        if B > n_lhs:
            # Generate LHS points
            # Split [0, 1] into n_lhs intervals, pick one random point in each, then scale to [-5, 5]
            perm = [rng.permutation(n_lhs) for _ in range(n)]
            lhs_points = np.zeros((n_lhs, n))
            for i in range(n_lhs):
                for j in range(n):
                    lhs_points[i, j] = (perm[j][i] + rng.random()) / n_lhs * 10.0 - 5.0
            
            for x in lhs_points:
                if calls >= B:
                    break
                if time.time() - start_time > time_limit:
                    break
                fx = func(x)
                calls += 1
                update_best(x, fx)
        
        # If we didn't use all budget in Phase 1, proceed to Phase 2
        if calls < B and best_x is not None:
            # Phase 2: Local Refinement via CMA-ES
            # Initialize CMA-ES at the best point found
            x = best_x.copy()
            # Start with a smaller sigma since we are likely closer to the minimum
            # than the origin. 1.0 is a good starting step for a 10-wide box after LHS.
            sigma = 1.0 
            
            # CMA-ES Parameters
            lam = 20
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            C = np.eye(n)
            D, B_eig = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-10))
            
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            
            def reflect(y):
                y = y.copy()
                for i in range(n):
                    if y[i] < -5:
                        y[i] = -5 + (-5 - y[i]) * 0.5
                        y[i] = max(y[i], -5)
                    elif y[i] > 5:
                        y[i] = 5 - (y[i] - 5) * 0.5
                        y[i] = min(y[i], 5)
                return np.clip(y, -5, 5)

            # Main CMA-ES Loop
            while calls < B:
                if time.time() - start_time > time_limit:
                    break
                
                offspring = []
                fvals = []
                
                for i in range(lam):
                    if calls >= B:
                        break
                    
                    z = rng.standard_normal(n)
                    y = x + sigma * (B_eig @ (D * z))
                    y = reflect(y)
                    
                    offspring.append(y)
                    fy = func(y)
                    fvals.append(fy)
                    calls += 1
                    
                    update_best(y, fy)
                    
                if len(fvals) < lam:
                    break
                
                order = np.argsort(fvals)
                offspring = [offspring[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Update mean
                x_new = np.sum([weights[i] * offspring[i] for i in range(mu)], axis=0)
                
                # Update covariance and step size
                y_inv = np.linalg.solve(C, (x_new - x) / sigma)
                
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_inv
                
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls // lam + 1)))
                        < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))
                
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_inv
                
                delta_hsig = (1 - hsig) * cc * (2 - cc)
                rank_one = np.outer(pc, pc)
                
                rank_mu_terms = []
                for i in range(mu):
                    zi = np.linalg.solve(C, (offspring[i] - x) / sigma)
                    rank_mu_terms.append(weights[i] * np.outer(zi, zi))
                rank_mu = np.sum(rank_mu_terms, axis=0)
                
                C = (1 + c1 * delta_hsig + cmu - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C += (1e-10 - np.min(eigvals)) * np.eye(n)
                
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                x = x_new
                
                D, B_eig = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-10))

        # If no calls were made (should not happen), return origin
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
