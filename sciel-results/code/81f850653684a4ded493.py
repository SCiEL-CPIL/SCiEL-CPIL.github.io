"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        d = self.dim

        # CMA-ES parameters
        # Population size
        lambda_pop = int(4 + 3 * np.log(d))
        
        # Initial step size
        sigma = 2.0
        
        # Initial center
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        
        # Covariance matrix (identity initially)
        C = np.eye(d)
        
        # Eigen decomposition for sampling
        eigval, eigvec = np.linalg.eigh(C)
        
        # Learning rates
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / d) / ((d + 4) + 2 * mueff / (d - 2))
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # State vectors
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        chi_n = np.sqrt(d) * (1 + 1 / (2 * d) - 1 / (8 * d**2))
        
        # Budget management
        calls_made = 1
        
        while calls_made < self.budget:
            # Check time deadline
            if time.time() > deadline - 2.0:
                break
                
            # Sample population
            ys = []
            for i in range(lambda_pop):
                # Sample from N(0, C)
                z = rng.standard_normal(d)
                y = x + sigma * (eigvec @ (np.sqrt(eigval) * z))
                y = np.clip(y, -5, 5)
                ys.append(y)
                
            # Evaluate population
            fs = np.array([func(y) for y in ys])
            calls_made += lambda_pop
            
            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = np.array([ys[i] for i in order])
            fs_sorted = fs[order]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()
                
            # Mean update
            x_old = x
            x = np.sum(ys_sorted[:mu] * weights[:, None], axis=0)
            
            # Sigma update
            y_sigma = (x - x_old) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_sigma
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (calls_made - 1))) / chi_n < 1.4 + 2 / (d + 1))
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # C update
            if hsig:
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * y_sigma
            else:
                p_c = (1 - cc) * p_c
                
            C = (1 + c1 + cmu - c1 - 2 * cmu) * C + \
                c1 * (np.outer(p_c, p_c) + (cc * (2 - cc)) * (np.outer(p_c, p_c) if not hsig else 0)) + \
                cmu * np.sum(weights[i] * np.outer(ys_sorted[i] - x_old, ys_sorted[i] - x_old) for i in range(mu))
            
            # Ensure positive semi-definiteness and update eigen decomposition
            # Simple repair: if eigenvalues become negative, clip them
            eigval, eigvec = np.linalg.eigh(C)
            # If any eigenvalue is too small or negative, reset to identity-like
            if np.any(eigval < 1e-12):
                C = np.eye(d)
                eigval, eigvec = np.linalg.eigh(C)
            else:
                # Renormalize to keep determinant roughly constant or similar scale
                # Actually, standard CMA-ES updates C directly.
                # Let's just recompute eig for sampling.
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
