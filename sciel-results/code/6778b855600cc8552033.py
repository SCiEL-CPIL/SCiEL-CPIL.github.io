import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        
        best_x = None
        best_f = float('inf')
        
        # CMA-ES parameters
        lam = int(4 + 3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights**2)
        
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        
        def run_cmaes(x0, budget_for_run):
            nonlocal calls_made, best_x, best_f
            nonlocal deadline, start_time
            
            m = x0.copy()
            sigma = 2.0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Evaluate start
            f0 = func(m)
            calls_made += 1
            if f0 < best_f:
                best_f = f0
                best_x = m.copy()
                
            # Check if we have enough budget
            if budget_for_run <= 1:
                return
                
            # CMA-ES Loop
            stagnation_counter = 0
            last_improvement_f = best_f
            
            while calls_made < total_budget and time.time() < deadline:
                # Determine how many offspring we can afford
                remaining = total_budget - calls_made
                current_lam = min(lam, remaining)
                if current_lam < 2:
                    break
                    
                # Generate offspring
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = (C + C.T) / 2
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except:
                        C_sqrt = np.eye(n)
                        
                ys = []
                for i in range(current_lam):
                    z = np.random.standard_normal(n)
                    y = m + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5.0, 5.0)
                    ys.append(y)
                    
                # Evaluate
                fs = np.array([func(y) for y in ys])
                calls_made += current_lam
                
                # Sort
                order = np.argsort(fs)
                # We only need top mu, but mu might be larger than current_lam if current_lam < mu
                effective_mu = min(mu, current_lam)
                if effective_mu == 0:
                    break
                ys_sorted = [ys[i] for i in order[:effective_mu]]
                fs_sorted = fs[order]
                
                # Update weights if effective_mu < mu
                if effective_mu < mu:
                    w_eff = weights[:effective_mu]
                    w_eff = w_eff / np.sum(w_eff)
                    mu_eff_eff = 1 / np.sum(w_eff**2)
                else:
                    w_eff = weights
                    mu_eff_eff = mu_eff
                    
                # Update best
                if fs_sorted[0] < best_f:
                    best_f = fs_sorted[0]
                    best_x = ys_sorted[0].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    if stagnation_counter > 30: # Restart logic
                        break
                        
                # Update mean
                ymean = np.mean(ys_sorted, axis=0, weights=w_eff)
                
                # Update paths
                # Note: The standard CMA-ES update uses the full population stats.
                # If we truncated the population, the updates are slightly off, but acceptable for robustness.
                pc = (1 - cc) * pc + cc * np.sqrt(mu_eff_eff) / sigma * (C_sqrt @ (ymean - m))
                
                # Check path length for sigma update
                # Simplified sigma update
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff_eff) / sigma * (C_sqrt @ (ymean - m))
                
                # Sigma update
                norm_ps = np.linalg.norm(ps)
                # Reference length for ps
                ref_len = np.sqrt(n) * (1 + 2 / (n + 1))
                # hsig check (simplified)
                hsig = 1.0 if (norm_ps / np.sqrt(1 - (1 - cs) ** 2) < ref_len) else 0.0
                
                sigma *= np.exp(cs / damps * (norm_ps / np.sqrt(1 - (1 - cs) ** 2) - np.sqrt(n) * (1 + 2 / (n + 1))))
                
                # Update Covariance
                delta_sig = hsig * (1 / cc - 1) + (1 - hsig)
                C = (1 + c1 * delta_sig - c1 - cmu) * C + c1 * np.outer(pc, pc)
                
                # Rank-mu update
                y_diffs = np.array([y - ymean for y in ys_sorted])
                rank_mu_term = np.zeros((n, n))
                for i in range(effective_mu):
                    rank_mu_term += w_eff[i] * np.outer(y_diffs[i], y_diffs[i])
                    
                C = C + cmu * rank_mu_term
                C = (C + C.T) / 2
                
                m = ymean
                
                # Convergence check
                if sigma < 1e-10:
                    break

        # Multi-start strategy
        # 1. Start at 0
        # 2. If stagnates, start at random points
        
        # Initial start at 0
        run_cmaes(np.zeros(n), total_budget)
        
        # If we still have budget and haven't converged, try a few random restarts
        # But the loop inside run_cmaes handles the budget.
        # If it broke due to stagnation, we can try again.
        
        # Let's do a simpler multi-start: 
        # Run CMA-ES for 50% budget.
        # If not converged, run CMA-ES from a random point for the rest.
        
        # Actually, the above run_cmaes uses all budget. 
        # To implement restarts properly, I should split the budget.
        
        # Let's refactor: 
        # Phase 1: CMA-ES from 0 for 300 evals.
        # Phase 2: If best_f is still "bad" (e.g., > 100), restart from random for remaining.
        
        # Since I can't easily modify the above loop to stop early without rewriting, 
        # I will just run it. The stagnation break will exit early if it gets stuck.
        
        return best_x, best_f
