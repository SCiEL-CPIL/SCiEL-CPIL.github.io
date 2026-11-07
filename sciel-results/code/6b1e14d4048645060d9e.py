import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin

        def elapsed_ok():
            return time.time() < deadline

        # CMA-ES Parameters
        # Large population for fast adaptation on the ridge
        lam = 20
        mu = lam // 2
        # Weights for the best mu individuals
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Damping for covariance update
        d_amps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initial state
        # Start at origin, which is often close to the basin for these shifted problems
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Pre-compute Cholesky factors (will be updated)
        B = np.eye(dim)
        D = np.ones(dim)
        
        def cholesky_update(C):
            # Symmetrize C to avoid numerical issues
            C = (C + C.T) / 2.0
            try:
                L = np.linalg.cholesky(C)
                # B and D such that C = B diag(D^2) B^T
                # From L = B diag(D), so B = L / diag(D)
                d = np.sqrt(np.sum(L ** 2, axis=0))
                B = L / d
                return B, d
            except np.linalg.LinAlgError:
                # Fallback: eigen decomposition
                vals, vecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                vals = np.clip(vals, 1e-12, None)
                D = np.sqrt(vals)
                B = vecs
                return B, D

        B, D = cholesky_update(C)
        
        best_x = x_mean.copy()
        best_f = np.inf
        evaluated = 0
        
        # Helper to evaluate a point
        def eval_point(x):
            nonlocal best_x, best_f, evaluated
            if evaluated >= budget:
                return np.inf
            if not elapsed_ok():
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Initial evaluation of the mean
        best_f = eval_point(x_mean)
        
        while evaluated < budget and elapsed_ok():
            # Generate population
            ys = []
            for i in range(lam):
                # z ~ N(0, I)
                z = np.random.randn(dim)
                # y = B @ (D * z)
                y = B @ (D * z)
                x = x_mean + sigma * y
                ys.append(x)
            
            # Evaluate population
            fs = []
            for x in ys:
                f = eval_point(x)
                if not elapsed_ok() or evaluated >= budget:
                    break
                fs.append(f)
            
            if len(fs) < lam:
                # If we ran out of budget or time, pad with inf to keep structure
                while len(fs) < lam:
                    fs.append(np.inf)
            
            # Sort
            idx = np.argsort(fs)
            ys = np.array([ys[i] for i in idx])
            fs = np.array([fs[i] for i in idx])
            
            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * ys[i]
            
            # Update step size control vectors
            x_diff = x_new_mean - x_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (D * (x_diff / sigma)))
            hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (evaluated % 100 + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff)
            pc = (1 - c_c) * pc + hs * np.sqrt(c_c * (2 - c_c) * mu_eff) * x_diff
            
            # Update covariance matrix
            delta_hs = (1 - hs) * c_c * (2 - c_c)
            delta_pc = np.outer(pc, pc)
            
            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (ys[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * (delta_pc + delta_hs * np.outer(x_diff/sigma, x_diff/sigma)) + c_mu * rank_mu
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_amps * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            x_mean = x_new_mean
            
            # Update Cholesky factors
            B, D = cholesky_update(C)
            
            # Check for convergence or stagnation
            if np.linalg.norm(x_diff) < 1e-10 * max(1.0, np.linalg.norm(x_mean)):
                break
            if sigma < 1e-10:
                break

        return best_x, best_f
