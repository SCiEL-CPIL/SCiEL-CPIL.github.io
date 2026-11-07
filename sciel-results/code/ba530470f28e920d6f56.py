# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        # Weights for the mu best individuals
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w ** 2)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initial step size and mean
        sigma = 1.0
        m = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        # Precompute sqrt of C for sampling
        # We will use Cholesky decomposition
        try:
            sqrtC = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            sqrtC = np.eye(dim)
            
        # Cumulation path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Evaluate initial mean
        m_clipped = np.clip(m, -5, 5)
        evaluate(m_clipped)

        # Main loop
        while calls_made < budget:
            # Generate population
            Y = np.random.normal(0, 1, size=(lam, dim))
            # Transform to X
            # X = m + sigma * Y @ sqrtC.T
            X = m + sigma * (Y @ sqrtC.T)
            
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate population
            F = np.array([evaluate(x) for x in X])
            
            # Handle inf/nan
            valid = np.isfinite(F)
            if not np.any(valid):
                break
            
            # Sort indices by fitness
            order = np.argsort(F)
            # Select mu best
            I = order[:mu]
            
            # Update mean
            m_old = m.copy()
            m_new = np.sum(w[:, None] * X[I], axis=0)
            
            # Update p_sigma
            # Need to be careful with sigma and C updates order
            # Standard CMA-ES update:
            # 1. Update m
            # 2. Update p_sigma using (m_new - m_old) / (sigma * sqrt(mu_eff))
            # 3. Update sigma
            # 4. Update p_c
            # 5. Update C
            
            if np.linalg.norm(m_new - m_old) > 0:
                p_sigma += c_sigma / sigma * (m_new - m_old)
            else:
                p_sigma += 0
            
            # Update sigma
            # hs is a heuristic for whether the step is in a consistent direction
            hs = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls_made)))
            if hs < 4.4 + 2 / dim:
                sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
            
            # Update p_c
            h_sig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2)) < (1.4 + 2 / (dim + 1)) * np.sqrt((mueff + 1) / (dim + 1))
            if h_sig:
                p_c = (1 - c_c) * p_c + h_sig * c_c / sigma * (m_new - m_old)
            
            # Update C
            # Rank one update
            y = (m_new - m_old) / sigma
            C += c_1 * (np.outer(y, y) + (1 - h_sig * (1 - c_c)) / c_c - 1) * C
            
            # Rank mu update
            # Compute H = sum w_i * (y_i - y_mean) outer (y_i - y_mean)
            # y_i are the selected steps
            Y_sel = (X[I] - m_old) / sigma
            # Center the selected Y
            Y_mean = np.sum(w[:, None] * Y_sel, axis=0)
            Y_centered = Y_sel - Y_mean[None, :]
            H = np.sum(w[:, None, None] * np.outer(Y_centered, Y_centered), axis=0)
            C += c_mu * H
            
            # Regularize C to ensure positive definiteness
            # Simple regularization: add small identity
            C += 1e-10 * np.eye(dim)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sqrtC
            try:
                sqrtC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                sqrtC = eigvecs @ np.diag(np.sqrt(eigvals))
            
            m = m_new
            
            # Check for stagnation
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
