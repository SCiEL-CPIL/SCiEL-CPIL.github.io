"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with multi-start for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin (330s wall clock limit)
        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if not np.all(np.isfinite(x)):
                return np.inf
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cma_es(x0, sigma0, budget, rng):
            """Run CMA-ES starting from x0 with initial sigma0, using up to budget evaluations."""
            nonlocal calls_made, best_f, best_x, deadline
            
            if calls_made >= total_budget or time.time() > deadline:
                return x0, best_f
            
            # CMA-ES parameters
            lam = 4 * n  # Population size
            mu = lam // 2  # Number of selected parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Strategy parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # State variables
            mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            cs = (1 + c_sigma + c_c + c_mu) / (1 - c_sigma - c_c - c_mu)
            
            # Precompute Cholesky factor
            C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            
            best_local_x = mean.copy()
            best_local_f = np.inf
            
            for gen in range(10000):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                # Generate population
                offspring = []
                for i in range(lam):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    x = mean + sigma * (C_inv_sqrt @ z)
                    x = np.clip(x, -5.0, 5.0)
                    f = evaluate(x)
                    if np.isfinite(f):
                        offspring.append((f, x))
                
                if len(offspring) < mu:
                    break
                
                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                
                # Update best
                for f, x in offspring[:mu]:
                    if f < best_local_f:
                        best_local_f = f
                        best_local_x = x.copy()
                
                # Compute new mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * offspring[i][1]
                y_w /= sigma
                
                # Update path evolution strategies
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ y_w
                norm_ps = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))
                
                delta_hs = 1.4 + 2 / (n + 1)
                if norm_ps < (1.4 + 2 / (n + 1)) * np.sqrt((1 - (1 - c_sigma)**(2 * (gen + 1))) * n):
                    pc = (1 - c_c) * pc + c_c * y_w
                
                # Update covariance matrix
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * np.eye(n))
                for i in range(mu):
                    y_i = (offspring[i][1] - mean) / sigma
                    C += c_mu * weights[i] * np.outer(y_i, y_i)
                
                # Update mean
                mean = mean + sigma * y_w
                mean = np.clip(mean, -5.0, 5.0)
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(n) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Recompute Cholesky factor periodically
                if gen % 10 == 0:
                    try:
                        C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C + 1e-10 * np.eye(n)))
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        C_inv_sqrt = np.eye(n)
                        pc = np.zeros(n)
                        ps = np.zeros(n)
                        sigma = sigma0 * 0.5
                
                # Convergence check
                if sigma < 1e-8 or np.all(np.abs(mean) < 1e-8):
                    break
            
            return best_local_x, best_local_f

        # Phase 1: Multi-start CMA-ES
        # Try several different starting points to find both Lunacek basins
        start_points = [
            np.zeros(n),
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            rng.uniform(-3.0, 3.0, n),
            rng.uniform(-3.0, 3.0, n),
        ]
        
        # Budget allocation: give more budget to earlier starts if they find good solutions
        remaining_budget = total_budget
        for idx, x0 in enumerate(start_points):
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            # Allocate remaining budget proportionally, with diminishing returns
            if idx == 0:
                alloc = remaining_budget // 2
            elif idx == 1:
                alloc = remaining_budget // 3
            elif idx == 2:
                alloc = remaining_budget // 4
            else:
                alloc = remaining_budget // 5
            
            alloc = max(alloc, 30)  # Minimum 30 evaluations per start
            alloc = min(alloc, remaining_budget - 10)  # Keep some budget for others
            
            if alloc < 10:
                continue
            
            x_local, f_local = cma_es(x0, 2.0, alloc, rng)
            
            if f_local < best_f:
                best_x = x_local.copy()
                best_f = f_local
            
            remaining_budget = total_budget - calls_made
        
        # Phase 2: Final polish from best point with remaining budget
        if calls_made < total_budget and time.time() < deadline:
            if best_x is None:
                best_x = np.zeros(n)
                best_f = evaluate(best_x)
            
            # Run CMA-ES from best point with remaining budget
            x_final, f_final = cma_es(best_x, 0.5, total_budget - calls_made, rng)
            if f_final < best_f:
                best_x = x_final.copy()
                best_f = f_final

        if best_x is None:
            best_x = np.zeros(n)
            if calls_made < total_budget:
                best_f = evaluate(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
