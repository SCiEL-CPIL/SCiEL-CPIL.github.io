"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _run_cmaes(self, func, x0, sigma0, budget, rng, deadline, global_best_x, global_best_fx):
        """
        Runs a single CMA-ES instance.
        Returns (best_x, best_fx) from this run.
        Updates global_best if found better.
        """
        dim = self.dim
        x = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Parameters for dim=5
        mu = max(1, dim // 2)  # 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        lam = 4 + int(3 * np.log(dim))  # 4 + 3*1.609 = 8 (approx, int(4.8)=4 -> 8? No, log(5)=1.609, 3*1.609=4.82, int=4, 4+4=8)
        
        # Initial evaluation
        fx = func(x)
        calls_made = 1
        gen = 0
        
        local_best_x = x.copy()
        local_best_fx = fx
        
        if fx < global_best_fx:
            global_best_fx = fx
            global_best_x = x.copy()

        # Main loop for this instance
        while calls_made + lam <= budget:
            if time.time() > deadline:
                break
                
            # Decompose C
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D2, 1e-12))
            invB = np.linalg.inv(B)
            
            # Generate offspring
            z = rng.standard_normal((lam, dim))
            y = x[None, :] + sigma * (z @ (B / D[None, :]))
            y = np.clip(y, -5, 5)
            
            # Evaluate
            fy = np.array([func(yi) for yi in y])
            calls_made += lam
            gen += 1
            
            # Track best
            idx_min = np.argmin(fy)
            if fy[idx_min] < local_best_fx:
                local_best_fx = fy[idx_min]
                local_best_x = y[idx_min].copy()
                if local_best_fx < global_best_fx:
                    global_best_fx = local_best_fx
                    global_best_x = local_best_x.copy()
            
            # Sort by fitness
            order = np.argsort(fy)
            y_sel = y[order[:mu]]
            
            # Weighted mean
            m = (w[:, None] * y_sel).sum(axis=0)
            
            # Update path accumulators
            y_m_sigma = (m - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invB @ y_m_sigma)
            
            # hsig check
            ps_norm = np.linalg.norm(ps)
            denom = np.sqrt(1 - (1 - cs) ** (2 * gen))
            if denom > 1e-10:
                ratio = ps_norm / denom
            else:
                ratio = 0
            hsig = 1.0 if ratio < (1 + 2 / dim) * np.sqrt(cs * (2 - cs)) else 0.0
            
            # Update pc
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * (invB @ y_m_sigma)
            
            # Update C
            rank1 = np.outer(pc, pc)
            y_sel_sigma = (y_sel - x) / sigma
            z_sel = (D[None, :] * (invB @ y_sel_sigma.T).T)
            rank_mu = np.sum([w[i] * np.outer(z_sel[i], z_sel[i]) for i in range(mu)], axis=0)
            
            C = (1 - c1 - cmu) * C + c1 * (rank1 + (1 - hsig) * cs * (2 - cs) * C) + cmu * rank_mu
            C = (C + C.T) / 2
            
            # Update x
            x = m
            
            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            if denom > 1e-10:
                log_sigma_update = (ps_norm / denom - chi_n)
            else:
                log_sigma_update = 0
            sigma *= np.exp(cs / ds * log_sigma_update)
            sigma = np.clip(sigma, 1e-10, 10.0)
            
        return local_best_x, local_best_fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        
        # Multi-start strategy
        # Split budget into N runs. 
        # dim=5, budget=500. 
        # Let's try 5 runs of 100 calls each.
        # Or maybe 4 runs of 125 calls.
        # CMA-ES needs at least 1 + lam calls per generation. lam=8.
        # 100 calls is about 12 generations. Might be enough for coarse search.
        # Let's do 5 restarts.
        
        n_restarts = 5
        budget_per_run = self.budget // n_restarts
        
        global_best_x = None
        global_best_fx = np.inf
        
        for i in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Initialize new run
            # Random start in the box
            x0 = rng.uniform(-5, 5, self.dim)
            # Start with a reasonably large sigma to explore
            sigma0 = 2.0
            
            # Run CMA-ES
            best_x, best_fx = self._run_cmaes(
                func, x0, sigma0, budget_per_run, rng, deadline, 
                global_best_x, global_best_fx
            )
            
            if best_fx < global_best_fx:
                global_best_fx = best_fx
                global_best_x = best_x
        
        # If we still have time and budget, do a final polish with a smaller sigma around the best found
        if global_best_x is not None and time.time() < deadline:
            # Check remaining budget? The _run_cmaes tracks its own budget.
            # We can just try to run one more short CMA-ES starting from global_best_x with small sigma
            # But we need to be careful with the total call count.
            # The task says "More than 500 calls... makes the whole result invalid".
            # My _run_cmaes uses a local budget counter. 
            # I need to track total calls globally to ensure I don't exceed 500.
            
            # Let's refactor to track global calls.
            # For now, the loop above uses budget_per_run which sums to self.budget.
            # So total calls <= self.budget.
            # I will not do a final polish to avoid exceeding the budget or complexity.
            pass
            
        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
