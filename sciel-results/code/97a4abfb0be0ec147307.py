"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with proper covariance adaptation and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with initial step sigma0 for up to budget calls."""
        d = self.dim
        # Population size: larger for better exploration
        n = max(6, int(4 + 3 * np.log(d)))
        calls = 0
        x = np.asarray(x0, dtype=float).copy()
        fx = func(x)
        calls += 1
        best_x = x.copy()
        best_f = fx

        sigma = sigma0
        mean = x.copy()
        C = np.eye(d)
        
        # CMA-ES parameters
        mu = n // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - d - 2) / (d + mueff)) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        
        pc = np.zeros(d)
        pc_norm = 0.0
        # For proper CMA-ES, we need to track the path
        # But for simplicity and robustness, use a simplified but correct approach
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population using current C
            # Use Cholesky decomposition for proper sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: add regularization
                C += 1e-10 * np.eye(d)
                L = np.linalg.cholesky(C)
            
            pops = []
            fits = []
            for i in range(n):
                z = rng.standard_normal(d)
                y = mean + sigma * L @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                pops.append(y)
                fits.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

            if len(pops) < n:
                break

            # Sort by fitness
            order = np.argsort(fits)
            pops = [pops[i] for i in order]
            fits = [fits[i] for i in order]

            # Weighted mean of top mu individuals
            y_w = np.zeros(d)
            for i in range(mu):
                y_w += w[i] * (pops[i] - mean)
            mean_new = mean + y_w
            
            # Update step size using CSA
            norm_yw = np.linalg.norm(y_w)
            if norm_yw > 1e-15:
                sigma *= np.exp(c_sigma / d_sigma * (norm_yw / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 20.0)
            
            # Update covariance
            # Rank-1 update
            y_w_scaled = y_w / sigma if sigma > 1e-15 else y_w
            C = (1 - c_1 - c_mu) * C + c_1 * (2 - c_1) * np.outer(y_w_scaled, y_w_scaled)
            
            # Rank-mu update
            for i in range(mu):
                y_i = (pops[i] - mean) / sigma if sigma > 1e-15 else (pops[i] - mean)
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            # Ensure positive definiteness
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += (1e-12 - np.min(eigvals)) * np.eye(d)
            
            mean = mean_new
            
            # Convergence check
            if sigma < 1e-10 or norm_yw < 1e-12:
                break

        return best_x, best_f, calls

    def _local_polish(self, func, x0, budget, rng, deadline):
        """Simple local search to polish the solution."""
        d = self.dim
        calls = 0
        x = np.asarray(x0, dtype=float).copy()
        fx = func(x)
        calls += 1
        best_x = x.copy()
        best_f = fx
        
        step = 0.1
        while calls < budget and step > 1e-6:
            if time.time() > deadline:
                break
            improved = False
            for i in range(d):
                for direction in [-1, 1]:
                    x_trial = x.copy()
                    x_trial[i] += direction * step
                    x_trial = np.clip(x_trial, -5, 5)
                    fx_trial = func(x_trial)
                    calls += 1
                    if fx_trial < best_f:
                        best_f = fx_trial
                        best_x = x_trial.copy()
                        x = x_trial.copy()
                        improved = True
                    if calls >= budget or time.time() > deadline:
                        break
                if calls >= budget or time.time() > deadline:
                    break
            if not improved:
                step *= 0.5
            else:
                # Keep same step if improved
                pass
            if calls >= budget or time.time() > deadline:
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        deadline = time.time() + 275  # safety margin before 300s limit
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        used = 0

        # Phase 1: Initial random sampling
        n_init = min(10, budget)
        init_x = []
        init_f = []
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            used += 1
            init_x.append(x)
            init_f.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES from best initial point
        remaining = budget - used
        # Reserve 15% for local polish
        polish_budget = min(30, remaining // 5)
        cma_budget = remaining - polish_budget
        
        if cma_budget < 10:
            cma_budget = remaining
            polish_budget = 0
        
        bx, bf, calls = self._cma_es(func, best_x.copy(), 1.0, cma_budget, rng, deadline)
        used += calls
        if bf < best_f:
            best_f = bf
            best_x = bx.copy()
        
        if used >= budget or time.time() > deadline or polish_budget <= 0:
            return best_x, best_f

        # Phase 3: Local polish
        bx, bf, calls = self._local_polish(func, best_x.copy(), polish_budget, rng, deadline)
        used += calls
        if bf < best_f:
            best_f = bf
            best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
