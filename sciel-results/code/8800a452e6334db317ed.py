"""CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock) with Two-Stage Strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using a Two-Stage CMA-ES strategy.
    Stage 1: Global exploration with large step size.
    Stage 2: Local exploitation with smaller step size starting from the best point found in Stage 1.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_cma_es(self, func, start_x, sigma, budget, rng, deadline):
        """
        Runs CMA-ES for a given budget.
        Returns best_x, best_f, calls_made.
        """
        d = self.dim
        # CMA-ES parameters
        lam = max(4 * d, 8)
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        x_mean = start_x.copy()
        current_sigma = sigma
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        C = np.eye(d)
        
        # Pre-compute matrices
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-12, None)
        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        invsqrtC = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        best_x = x_mean.copy()
        best_f = float(func(best_x))
        calls = 1

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            Y = []
            for i in range(lam):
                z = rng.standard_normal(d)
                y = x_mean + current_sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                Y.append(y)

            # Evaluate offspring
            F = []
            for y in Y:
                if calls >= budget:
                    F.append(np.inf)
                    continue
                calls += 1
                try:
                    fy = float(func(y))
                except Exception:
                    fy = np.inf
                if not np.isfinite(fy):
                    fy = np.inf
                F.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = [Y[i] for i in idx]
            F_sorted = [F[i] for i in idx]

            # Update mean
            y_w = np.sum([w[i] * Y_sorted[i] for i in range(mu)], axis=0)
            x_mean_old = x_mean.copy()
            x_mean = y_w

            # Update sigma
            z_w = invsqrtC @ (y_w - x_mean_old) / current_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_w
            # Note: The h_sigma condition in standard CMA-ES uses the norm of p_sigma relative to expected norm
            # We use a simplified check or standard formula.
            # Standard: h_sigma = (||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*gen))) < (1.4 + 2/(d+1)) * chi_n
            # We approximate the denominator as 1 for simplicity in short runs, or just use the norm.
            # Actually, let's just use the norm check.
            h_sigma = (np.linalg.norm(p_sigma) < (1.4 + 2 / (d + 1)) * chi_n)
            
            current_sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            # Prevent sigma from becoming too small or too large
            current_sigma = np.clip(current_sigma, 1e-10, 10.0)

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (y_w - x_mean_old) / current_sigma

            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank1 = np.outer(p_c, p_c)
            rankmu = np.sum([w[i] * np.outer(Y_sorted[i] - x_mean_old, Y_sorted[i] - x_mean_old) for i in range(mu)], axis=0)
            C = (1 + c_1 * delta_h + c_mu - c_1) * C + c_1 * rank1 + c_mu * rankmu

            # Ensure positive definiteness and update matrices
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            invsqrtC = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Ensure x_mean is in bounds
            x_mean = np.clip(x_mean, -5, 5)
            
            # Optional: Evaluate mean if budget allows (can help with progress)
            if calls < budget:
                fm = float(func(x_mean))
                calls += 1
                if fm < best_f:
                    best_f = fm
                    best_x = x_mean.copy()

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        rng = np.random.default_rng(42)
        d = self.dim
        total_budget = self.budget

        # Split budget into two stages
        # Stage 1: 20% of budget for global exploration
        # Stage 2: 80% of budget for local exploitation
        stage1_budget = int(total_budget * 0.2)
        stage2_budget = total_budget - stage1_budget

        # Stage 1: Global Search
        # Start at origin with large step
        start_x1 = np.zeros(d)
        sigma1 = 2.0
        
        best_x1, best_f1, calls1 = self._run_cma_es(func, start_x1, sigma1, stage1_budget, rng, deadline)
        
        # If time is up, return best found
        if time.time() > deadline:
            return best_x1, best_f1

        # Stage 2: Local Exploitation
        # Start from the best point found in Stage 1 with a smaller step
        start_x2 = best_x1
        sigma2 = 0.5  # Smaller step to refine
        
        best_x2, best_f2, calls2 = self._run_cma_es(func, start_x2, sigma2, stage2_budget, rng, deadline)

        # Return the overall best
        if best_f2 < best_f1:
            return best_x2, best_f2
        else:
            return best_x1, best_f1

# EVOLVE-BLOCK-END
