"""Rewritten optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f17 (Schaffers F7) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def clip(x):
            return np.clip(x, -5, 5)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_f, best_x
            x = clip(x)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES Parameters
        # Population size lambda
        lam = int(4 + 3 * np.log(n))
        
        # Initial mean: center of the box
        mean = np.zeros(n)
        
        # Initial step size: roughly 1/3 of the box width
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(n)
        
        # Inverse of covariance matrix
        C_inv = np.eye(n)
        
        # Eigenvalues and eigenvectors for C
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Adaptation parameters
        # Step-size control (CSA)
        mu = lam // 2  # Selection weight: best mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_cov = (2 + 2 * mu_eff / n) / (n + 5 + mu_eff / n)
        
        # Memory for path accumulators
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Generate initial population
        # We need to ensure we don't exceed budget.
        # Each generation uses lam evaluations.
        
        # Helper to generate next population
        def generate_population():
            nonlocal mean, sigma, C, C_inv, eigvals, eigvecs, p_sigma, p_c
            ys = []
            for i in range(lam):
                # Sample from multivariate normal
                # Use eigen decomposition for stability
                z = rng.standard_normal(n)
                # Transform z: z -> C^{1/2} z
                # C^{1/2} = eigvecs @ diag(sqrt(eigvals)) @ eigvecs.T
                y = eigvecs @ (np.sqrt(eigvals) * (eigvecs.T @ z))
                x = mean + sigma * y
                ys.append(y)
            return ys, [mean + sigma * y for y in ys]

        # Main loop
        while calls < budget:
            # Generate population
            ys, xs = generate_population()
            
            # Evaluate population
            fs = []
            for x in xs:
                if calls >= budget:
                    break
                fs.append(evaluate(x))
            
            if calls >= budget:
                break
            
            # Sort by fitness
            # Note: if budget ran out mid-evaluation, fs might be shorter.
            # We handle this by only using full generations for adaptation,
            # but we must track best_x correctly.
            # For CMA-ES adaptation, we need a full set of ys and fs.
            # If we didn't finish the population, we skip adaptation and just return best.
            if len(fs) < lam:
                break
            
            order = np.argsort(fs)
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update best
            # (Already done in evaluate, but ensure best_x is from this gen if better)
            # evaluate() already updates best_x and best_f globally.
            
            # Weighted mean of selected individuals
            y_w = sum(weights[i] * ys[i] for i in range(mu))
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ y_w)
            
            # Hessian update for sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update p_c
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam + 1)))) < (1 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** 2)
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv @ y_w)
            
            # Update covariance matrix
            rank_one = np.outer(y_w, y_w)
            rank_mu = sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))
            
            # Correction term
            delta_h = (1 - h_sigma) * c_c * (2 - c_c) * np.outer(p_c, p_c)
            
            C = (1 + c_cov * (c_c + 2 - c_c * (1 - h_sigma)) - c_cov) * C + c_cov * (rank_one + rank_mu - delta_h)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Update C_inv and eigen decomposition
            # For small n, we can recompute eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues to be positive
                eigvals = np.clip(eigvals, 1e-12, None)
            except np.linalg.LinAlgError:
                # Fallback: reset if numerical issues
                C = np.eye(n)
                eigvals, eigvecs = np.linalg.eigh(C)
                C_inv = np.eye(n)
            
            # Update mean
            mean = mean + sigma * y_w
            
            # Clip mean to box? CMA-ES usually doesn't clip the mean,
            # but the samples are clipped. Clipping the mean can help
            # if the optimum is near the boundary.
            # However, standard CMA-ES relies on the distribution.
            # Let's clip the mean to be safe for the next generation's center.
            mean = np.clip(mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
