"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4, path-based covariance adaptation, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return (time.time() - start_time) < deadline

        def evaluate(x):
            nonlocal calls_made
            fx = func(x)
            calls_made += 1
            return fx

        # CMA-ES Parameters
        n = dim
        # Population size: lambda = 4 as per directive
        lam = 4
        mu = lam // 2  # Number of best solutions used for mean update

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + c_sigma * 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # Initial state
        m = np.zeros(n)  # Start at center
        sigma = 5.0      # Large initial step to cover the box [-5, 5]^5
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Track best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Initial random scan to ensure we don't miss a basin if 0 is bad
        # But CMA-ES with sigma=5.0 starting at 0 should cover the box quickly.
        # Let's do a few random starts just in case, then hand off to CMA-ES
        # Actually, the directive says "Initialize the population around the origin with a large initial sigma".
        # So we start m=0, sigma=5.0.
        
        # To be safe and exploit early progress, we can do a few random evaluations first.
        # But CMA-ES generation 1 will already sample widely.
        # Let's stick to pure CMA-ES logic as directed, but ensure we record the best.

        while calls_made < budget and time_ok():
            # Generate population
            xs = np.zeros((lam, n))
            fs = np.zeros(lam)
            
            for i in range(lam):
                if not time_ok() or calls_made >= budget:
                    break
                z = rng.standard_normal(n)
                x = m + sigma * (B @ (D * z))
                # Clip to box
                x = np.clip(x, -5, 5)
                xs[i] = x
                fs[i] = evaluate(x)
                update_best(x, fs[i])
            
            if calls_made >= budget:
                break
            
            # Sort by fitness
            idx = np.argsort(fs)
            xs_sorted = xs[idx]
            fs_sorted = fs[idx]
            
            # Update mean
            m_new = m + np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            
            # Update step size (sigma)
            # ps update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D ** -1) @ ((m_new - m) / sigma))
            
            # Heaviside function
            h_sigma = 1.0 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lam + 1)))) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            # pc update
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / sigma
            
            # Rank-one update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            v = np.outer(pc, pc)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * v + c_mu * np.sum(weights[:, None, None] * np.outer(xs_sorted[:mu] - m, xs_sorted[:mu] - m) / sigma ** 2, axis=0)
            
            # Eigen decomposition to maintain positive definiteness and update B, D
            # Use symmetric eigen decomposition
            try:
                D2, B = np.linalg.eigh(C)
                # Ensure positive definiteness
                D2 = np.clip(D2, 1e-10, None)
                D = np.sqrt(D2)
            except np.linalg.LinAlgError:
                # Fallback if numerical issues
                D2, B = np.linalg.eigh(C + 1e-10 * np.eye(n))
                D2 = np.clip(D2, 1e-10, None)
                D = np.sqrt(D2)
            
            m = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
