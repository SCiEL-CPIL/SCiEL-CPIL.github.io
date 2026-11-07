"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid CMA-ES and Nelder-Mead optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0 # Safety margin before 300s limit
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0
        rng = np.random.default_rng(42)

        # Helper to evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: CMA-ES ---
        
        # CMA-ES Parameters
        # Using lambda=8 for dim=5 to allow for ~43 generations (350/8)
        lambda_pop = 8
        mu = lambda_pop // 2
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        # State variables
        mean = np.zeros(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        sigma = 2.0
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Initial evaluation of mean
        evaluate(mean)
        
        # CMA-ES Loop
        # We stop CMA-ES when we have used 70% of the budget or time is running out
        cma_budget_limit = int(budget * 0.7)
        cma_time_limit = start_time + 200.0 # 200s for CMA-ES, leaving 80s for NM
        
        while call_count < cma_budget_limit and time.time() < cma_time_limit:
            # Generate population
            Z_list = []
            X = []
            F = []
            
            for i in range(lambda_pop):
                Z = rng.standard_normal(dim)
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ Z
                except np.linalg.LinAlgError:
                    y = Z
                
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                f = evaluate(x)
                
                Z_list.append(Z)
                X.append(x)
                F.append(f)
                
                if call_count >= cma_budget_limit or time.time() > cma_time_limit:
                    break
            
            if len(X) < mu:
                break
                
            # Sort by fitness
            sort_idx = np.argsort(F)
            X = [X[i] for i in sort_idx]
            Z_list = [Z_list[i] for i in sort_idx]
            F = [F[i] for i in sort_idx]
            
            # Update Mean
            old_mean = mean.copy()
            mean = np.sum([w * X[i] for i, w in enumerate(weights)], axis=0)
            
            # Update Covariance Matrix
            Z_norm = [Z_list[i] / sigma for i in range(mu)]
            
            # Rank-mu update
            C_update = np.zeros((dim, dim))
            for i in range(mu):
                C_update += weights[i] * np.outer(Z_norm[i], Z_norm[i])
            
            # Update pc and ps
            step = (mean - old_mean) / sigma
            pc = (1 - c_sigma) * pc + c_sigma * np.sqrt(mu_eff) * step
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * step
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma)**2) * C) + c_mu * C_update
            C = (C + C.T) / 2
            
            # Adapt sigma
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))
            
            # Clamp sigma
            if sigma < 1e-12:
                break
            if sigma > 10:
                sigma = 10

        # --- Phase 2: Nelder-Mead Polishing ---
        
        # If we still have budget and time, run Nelder-Mead
        if call_count < budget and time.time() < deadline:
            self._nelder_mead(func, evaluate, best_x.copy(), call_count, budget, deadline)

        return best_x, best_f

    def _nelder_mead(self, func, evaluate, start_x, call_count, budget, deadline):
        """
        Performs Nelder-Mead optimization starting from start_x.
        Updates the global best_x and best_f via the evaluate closure.
        """
        dim = self.dim
        # Simplex initialization
        simplex = [start_x.copy()]
        f_simplex = [evaluate(start_x)]
        
        # Initialize other simplex points with a small step
        step_size = 0.5
        for i in range(dim):
            x_i = start_x.copy()
            # Ensure we stay within bounds for initialization
            if start_x[i] + step_size > 5:
                x_i[i] -= step_size
            else:
                x_i[i] += step_size
            x_i = np.clip(x_i, -5, 5)
            simplex.append(x_i)
            f_simplex.append(evaluate(x_i))
            
        if call_count >= budget:
            return

        # Sort simplex by fitness
        order = np.argsort(f_simplex)
        simplex = [simplex[i] for i in order]
        f_simplex = [f_simplex[i] for i in order]

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5

        it = 0
        max_iter = 500 # High limit, but constrained by budget/time
        
        while it < max_iter and call_count < budget and time.time() < deadline:
            it += 1
            # Re-sort if necessary (should be mostly sorted, but contraction/shrinkage can mess it up)
            # Actually, standard NM updates one point at a time, so we just need to identify worst/best
            # But for safety with our custom evaluate tracking, let's sort each iteration if needed?
            # No, standard NM:
            # 1. Sort
            # 2. Centroid of all but worst
            # 3. Reflect
            # 4. Expand/Contract/Shrink
            
            # Sort
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            # Check convergence
            if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
                break
                
            # Centroid of best N-1 points
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Worst point
            x_worst = simplex[-1]
            f_worst = f_simplex[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = evaluate(x_r)
            
            if call_count >= budget:
                break
                
            if f_r < f_simplex[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = evaluate(x_e)
                
                if call_count >= budget:
                    break
                    
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_simplex[-2]:
                # Accept Reflection
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                f_c = evaluate(x_c)
                
                if call_count >= budget:
                    break
                
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    f_simplex[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_simplex[i] = evaluate(simplex[i])
                        
                        if call_count >= budget:
                            break
                    if call_count >= budget:
                        break

# EVOLVE-BLOCK-END
