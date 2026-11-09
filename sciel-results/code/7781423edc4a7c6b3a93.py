"""Black-box continuous optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Phase 1: Coarse grid/random initialization to find a decent starting point
        # Use a small number of random points to estimate the landscape
        n_init = min(20, budget // 10)
        init_points = []
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            init_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort init points by fitness
        init_points.sort(key=lambda t: t[1])
        
        # Use the best point as center for CMA-ES
        x0 = init_points[0][0].copy()
        f0 = init_points[0][1]
        
        # Determine initial sigma based on spread of good points
        if len(init_points) > 1:
            # Use distance to second best as a heuristic for sigma
            d = np.linalg.norm(init_points[0][0] - init_points[1][0])
            sigma = max(0.1, min(2.0, d * 0.5))
        else:
            sigma = 1.0

        calls_made = n_init
        remaining = budget - calls_made

        if remaining <= 0:
            return best_x, best_f

        # Phase 2: CMA-ES optimization
        # Population size
        lam = 4 + int(3 * np.log(dim))  # Standard CMA-ES population size
        mu = lam // 2  # Number of recombination weights

        # CMA-ES parameters
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + 4 + dim / (dim + 4))
        cs = (dim + 1.7) / (dim + 3.2 + mu)
        c1 = 2 / ((dim + 1.3) ** 2 + mu)
        cmu = min(1 - c1, 2 * (mu - 2 + 1. / mu) / ((dim + 2.) ** 2 + mu))

        # CMA-ES state
        mean = x0.copy()
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)  # We'll update this properly
        # Actually, let's use a simpler approach: maintain C and its Cholesky or eigen decomposition
        # For simplicity and robustness, let's use a simplified CMA-ES with diagonal adaptation initially
        # No, let's do proper CMA-ES but carefully.

        # Let's restart with a cleaner CMA-ES implementation
        # We'll use the standard CMA-ES algorithm

        def cma_es_step(mean, sigma, C, ps, pc, x0_center, budget_left):
            nonlocal best_x, best_f, calls_made
            
            # Population size
            lam = 4 + int(3 * np.log(dim))
            mu = lam // 2
            
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1. / np.sum(w ** 2)
            
            # Step size control
            cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + 4 + dim / (dim + 4))
            cs = (dim + 1.7) / (dim + 3.2 + mu_eff)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2.) ** 2 + mu_eff))
            
            # Initialize if needed
            if 'ps' not in cma_es_step.state:
                cma_es_step.state = {
                    'ps': np.zeros(dim),
                    'pc': np.zeros(dim),
                    'C': np.eye(dim),
                    'sigma': sigma,
                    'mean': mean.copy()
                }
            
            ps = cma_es_step.state['ps']
            pc = cma_es_step.state['pc']
            C = cma_es_step.state['C']
            sigma = cma_es_step.state['sigma']
            mean = cma_es_step.state['mean']
            
            # Generate population
            B = np.linalg.cholesky(C)
            D = np.sqrt(np.diag(C))
            
            offspring = []
            for i in range(lam):
                if time.time() > deadline or calls_made >= budget:
                    break
                zi = rng.standard_normal(dim)
                yi = mean + sigma * (B @ zi)
                yi = np.clip(yi, -5, 5)
                fi = func(yi)
                calls_made += 1
                offspring.append((yi, fi, zi))
            
            if len(offspring) < 2:
                return False
            
            # Sort by fitness
            offspring.sort(key=lambda t: t[1])
            
            # Update best
            if offspring[0][1] < best_f:
                best_f = offspring[0][1]
                best_x = offspring[0][0].copy()
            
            # Recombination
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * offspring[i][2]
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ y_w)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - n_init + 1))) <
                    (1 + 2 / (dim + 1)) * np.sqrt(dim + 2 - 1))
            pc = (1 - cc) * pc + hsig * cc * y_w
            
            # Update covariance matrix
            C = (1 + c1 * cmu) * C  # This is wrong, let me fix
            # Correct update:
            C = (1 - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            for i in range(mu):
                C += cmu * w[i] * np.outer(offspring[i][2], offspring[i][2])
            
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            
            # Update mean
            mean = mean + sigma * y_w
            
            # Update sigma
            sigma *= np.exp(cs / (dim + 4) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = max(1e-10, min(10.0, sigma))
            
            # Update state
            cma_es_step.state = {
                'ps': ps,
                'pc': pc,
                'C': C,
                'sigma': sigma,
                'mean': mean
            }
            
            return True

        # Simple CMA-ES loop with restarts
        n_restarts = 3
        calls_per_restart = max(10, remaining // n_restarts)
        
        for restart in range(n_restarts):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Reset CMA-ES state for this restart
            cma_es_step.state = None
            
            # Start from current best or a new random point
            if restart == 0:
                center = best_x.copy()
            else:
                center = rng.uniform(-5, 5, dim)
                fc = func(center)
                calls_made += 1
                if fc < best_f:
                    best_f = fc
                    best_x = center.copy()
            
            # Run CMA-ES for a fixed number of calls
            n_calls_this_restart = calls_per_restart
            for step in range(n_calls_this_restart):
                if calls_made >= budget or time.time() > deadline:
                    break
                if not cma_es_step(center, 1.0, np.eye(dim), np.zeros(dim), np.zeros(dim), center, n_calls_this_restart):
                    break
            
            # After each restart, if no improvement, try a new center
            if restart < n_restarts - 1:
                # Add some jitter to escape local minima
                if best_x is not None:
                    center = best_x + 0.5 * rng.standard_normal(dim)
                    center = np.clip(center, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
