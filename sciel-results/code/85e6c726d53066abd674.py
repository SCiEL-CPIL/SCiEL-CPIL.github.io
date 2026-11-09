"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        rng = np.random.default_rng(12345)
        n = self.dim
        total_budget = self.budget

        # Global best
        best_x = None
        best_f = np.inf

        # Phase 1: Broad exploration with CMA-ES
        # Start at center, large sigma
        x0 = np.zeros(n)
        sigma0 = 3.0
        pop_size = 12  # lambda
        max_iter_phase1 = 250  # Max evaluations for phase 1
        
        x_c = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # CMA-ES parameters
        mu = pop_size // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / w.sum()
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))

        calls_used = 0
        stagnation_counter = 0
        last_best_f = np.inf

        # Evaluate initial center
        f0 = func(x0)
        calls_used += 1
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()
        
        # Main CMA-ES loop
        while calls_used < total_budget:
            # Check time
            if time.time() > deadline:
                break

            # Sample population
            pop = []
            for i in range(pop_size):
                # B = C^0.5
                # Use eigendecomposition for simplicity and stability in low dim
                # In 5D, this is fast enough
                vals, vecs = np.linalg.eigh(C)
                B = vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
                # Actually, simpler: just use Cholesky if C is PD
                # But C might not be perfectly PD. Let's use eigendecomp for B.
                # Wait, B is the square root.
                # y = B * z
                z = rng.standard_normal(n)
                y = B @ z
                pop.append(x_c + sigma * y)

            # Evaluate population
            pop_f = []
            for y in pop:
                if time.time() > deadline:
                    break
                fy = func(y)
                calls_used += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                pop_f.append(fy)
            
            if len(pop_f) < pop_size:
                break

            # Sort by fitness
            idx = np.argsort(pop_f)
            pop_sorted = [pop[i] for i in idx]
            pop_f_sorted = [pop_f[i] for i in idx]

            # Update mean
            x_old = x_c.copy()
            x_c = sum(w[i] * pop_sorted[i] for i in range(mu))

            # Update step size control vector
            d_sigma = np.linalg.norm(B) # Wait, B depends on C which is updated.
            # Standard CMA-ES uses the old C for the ps update? 
            # Actually, ps update uses the old C^(-1/2) or similar? 
            # Let's stick to the standard update rules.
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff)/d_sigma * (C_old^(-1/2) * (x_new - x_old))
            # This is getting complicated to implement from scratch correctly in 5D without bugs.
            
            # Alternative: Use a simpler, more robust variant or just use the population mean update and covariance update.
            # Given the risk of CMA-ES implementation bugs leading to NaNs (as seen in inspiration 1), 
            # and the success of "Random Local Search" (which is just hill climbing with restarts),
            # maybe a robust (mu+lambda)-ES is safer?
            
            # Let's pivot to a robust (mu+lambda) CMA-ES implementation that is known to be stable.
            # Or even simpler: Nelder-Mead with restarts? No, CMA-ES is better for Rosenbrock.
            
            # Let's try a very simplified CMA-ES that is less prone to numerical issues.
            # We will use the standard formulas but carefully.
            
            # 1. Update mean
            # x_c is already updated above.
            
            # 2. Update ps
            # We need C^(-1/2). Let's compute it from the current C (before update).
            vals, vecs = np.linalg.eigh(C)
            C_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
            diff = (x_c - x_old)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ diff
            
            # 3. Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_used // pop_size))) < (1.4 + 2 / (n + 1)) * chi)
            pc = (1 - cc) * pc + hsig * cc * (x_c - x_old) / sigma
            
            # 4. Update C
            delta_hs = (hsig - 1) * (2 - cc)
            C = (1 + c1 * delta_hs - c1 - cmu) * C
            for i in range(mu):
                y_i = (pop_sorted[i] - x_old) / sigma
                C += cmu * w[i] * np.outer(y_i, y_i)
            
            # 5. Update sigma
            sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
            
            # 6. Check stagnation
            if pop_f_sorted[0] < last_best_f:
                last_best_f = pop_f_sorted[0]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Restart if stagnated or sigma too small/large
            if stagnation_counter > 10 or sigma < 1e-6 or sigma > 10:
                # Restart from best found so far with smaller sigma
                x_c = best_x.copy()
                sigma = max(sigma * 0.5, 0.1)
                C = np.eye(n)
                ps = np.zeros(n)
                pc = np.zeros(n)
                stagnation_counter = 0
                last_best_f = np.inf
                # Don't reset calls_used, we continue until budget or time out

        return best_x, best_f
# EVOLVE-BLOCK-END
