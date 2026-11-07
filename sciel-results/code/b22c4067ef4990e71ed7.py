"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = max(4 + int(3 * np.log(dim)), 8)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # State
        m = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        gen = 0
        
        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            calls_made += 1
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Main Loop
        while calls_made + lam <= budget:
            # Sample
            z = rng.multivariate_normal(np.zeros(dim), C, size=lam)
            x_pop = m + sigma * z
            
            # Evaluate
            f_pop = np.array([safe_eval(x) for x in x_pop])
            
            # Sort
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]
            f_sorted = f_pop[order]
            
            # Update Mean
            m_old = m.copy()
            m = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update Step Size
            y_sigma = (m - m_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            c_n = 1 - 1/(2*dim) + 1/(12*dim**2)
            expected_norm = c_n * np.sqrt(np.trace(C))
            if expected_norm > 0:
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / expected_norm - 1))
            else:
                sigma *= 0.9 # Fallback shrinkage
            
            # Update Covariance
            y_c = (m - m_old) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_c
            
            # h_sigma calculation
            # Simplified check for h_sigma
            # h_sigma = 1 if ||p_sigma|| / sqrt(1-(1-c_sigma)^(2*gen+2)) < (1.4 + 2/(dim+1)) * c_n * sqrt(trace(C))
            # We approximate the denominator as 1 for simplicity in early gens, or track it.
            # For robustness, we use a simplified version often found in implementations:
            # h_sigma = 1 if norm_p_sigma < (1.4 + 2/(dim+1)) * expected_norm else 0
            h_sigma = 1.0 if norm_p_sigma < (1.4 + 2.0 / (dim + 1.0)) * expected_norm else 0.0
            
            C = (1 + c_1 * (h_sigma * c_c * (2 - c_c) - 1) - c_mu) * C \
                + c_1 * (h_sigma * c_c * (2 - c_c) - 1) * np.outer(p_c, p_c) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(x_sorted[:mu] - m_old, x_sorted[:mu] - m_old) / sigma**2, axis=0)
                
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Ensure PSD
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C = C + (1e-12 - np.min(eigvals)) * np.eye(dim)
            
            gen += 1

        # If budget remains, evaluate a few more points around best_x
        while calls_made < budget:
            # Local search around best
            if best_x is not None:
                x_try = best_x + rng.normal(0, 0.1, dim)
                safe_eval(x_try)
            else:
                safe_eval(rng.uniform(-5, 5, dim))

        return best_x, best_f
# EVOLVE-BLOCK-END
