"""Rewritten optimizer using CMA-ES for BBOB-style f18."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 290.0  # Safety margin before 300s limit

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(d)))
        # Weights for the mean update
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        cs = (mueff + 2) / (d + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((d + 2)**2 + mueff))
        
        # Initial state
        m = np.zeros(d)
        sigma = 2.0  # Initial step size, covering a good portion of [-5, 5]
        C = np.eye(d)
        B = np.eye(d)
        D = 1.0
        ps = np.zeros(d)
        pc = np.zeros(d)
        
        # Cholesky factor of C
        try:
            Cholesky = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            Cholesky = np.eye(d)
            
        best_x = m.copy()
        best_f = np.inf
        calls = 0

        while calls < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break
            
            # Generate population
            ys = []
            xs = []
            fs = []
            for _ in range(lam):
                z = np.random.randn(d)
                y = Cholesky @ z
                x = m + sigma * y
                # Clip to box to ensure validity, though func does it too
                x_clipped = np.clip(x, -5.0, 5.0)
                fy = func(x_clipped)
                ys.append(y)
                xs.append(x_clipped)
                fs.append(fy)
                calls += 1
                if calls >= budget:
                    break
                
            if not fs:
                break
                
            fs = np.array(fs)
            xs = np.array(xs)
            ys = np.array(ys)
            
            # Sort by fitness
            order = np.argsort(fs)
            fs_sorted = fs[order]
            xs_sorted = xs[order]
            ys_sorted = ys[order]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = xs_sorted[0]
                
            # Update mean
            # Only use the best mu individuals
            m_new = m + sigma * np.sum(weights[:mu] * ys_sorted[:mu], axis=0)
            
            # Update step size
            # Need inverse Cholesky or use B and D
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * inv(Cholesky) @ (m_new - m) / sigma
            # Using B and D: inv(Cholesky) = B / D
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (m_new - m) / (sigma * D))
            
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls + 1) / ds)) < (1.4 + 2 / (d + 1)) * cs:
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(d) - 1))
                
            # Update covariance matrix
            # pc = (1 - c1) * pc + c1 * (m_new - m) / sigma
            pc = (1 - c1) * pc + c1 * (m_new - m) / sigma
            
            # Rank-one update
            y_sigma = (m_new - m) / sigma
            C = (1 + c1 * cmu * mueff - c1 - cmu) * C + c1 * np.outer(y_sigma, y_sigma)
            
            # Rank-mu update
            # Sum of outer products of the best mu y's
            y_mu = ys_sorted[:mu]
            rank_mu_update = np.sum([w * np.outer(y, y) for w, y in zip(weights[:mu], y_mu)], axis=0)
            C += cmu * rank_mu_update
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update Cholesky factor
            # We need to ensure C is positive definite
            # Add small regularization if needed
            C += 1e-10 * np.eye(d)
            try:
                Cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity or use eigenvalue decomposition
                # Simple fallback: reset C to identity
                C = np.eye(d)
                Cholesky = np.eye(d)
                
            # Update B and D
            # C = B * D^2 * B.T
            # We can update B and D using the Cholesky factor directly if we assume C = Cholesky @ Cholesky.T
            # But standard CMA-ES maintains B and D for efficiency in step size control
            # For simplicity and robustness, we can just use Cholesky for sampling
            # and recompute B and D if needed for ps update.
            # Actually, ps update uses inv(Cholesky).
            # Let's stick to using Cholesky for sampling and recompute B, D if necessary.
            # Or simpler: just use Cholesky for everything.
            # The ps update formula uses inv(Cholesky).
            # We can compute inv(Cholesky) using solve.
            
            m = m_new
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
