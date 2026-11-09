import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: LHS
        n_lhs = 20
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = -5.0 + 10.0 * (perm + 0.5) / n_lhs

        best_x = None
        best_f = np.inf
        calls_made = 0

        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: CMA-ES
        # Parameters
        lam = 4
        mu = lam // 2
        weights = np.log(mu / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        c_m = c_sigma  # Usually c_m = c_sigma
        
        # State
        x_mean = best_x.copy()
        sigma = 0.5
        C = np.eye(dim)
        C_chol = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        while calls_made < budget and time.time() < deadline:
            # Sample
            pop_x = np.zeros((lam, dim))
            pop_f = np.zeros(lam)
            
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x_i = x_mean + sigma * (C_chol @ z)
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls_made += 1
                pop_x[i] = x_i
                pop_f[i] = f_i
            
            if calls_made < budget and time.time() > deadline:
                break
            
            # Sort
            order = np.argsort(pop_f)
            pop_x = pop_x[order]
            pop_f = pop_f[order]
            
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop_x[0].copy()
                
            # Update Mean
            y = np.dot(weights, pop_x[:mu])
            x_mean_old = x_mean
            x_mean = x_mean + c_m * (y - x_mean)
            
            # Update Sigma
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) * C^-1/2 @ (y - x_mean_old)
            # C^-1/2 is approximated by inv(C_chol)
            C_chol_inv = np.linalg.inv(C_chol)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_chol_inv @ (y - x_mean_old))
            
            sigma = sigma * np.exp(c_sigma / (5 * dim) * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update Covariance
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mueff) * (y - x_mean_old) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (y - x_mean_old) / sigma
            
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - (1 - c_c)**2) * (np.linalg.norm(pc)/chi_n)**2 - 1) * I
            # + c_mu * sum(w_i * (y_i - x_mean_old) (y_i - x_mean_old).T) / sigma^2
            
            # Rank-one update
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) / chi_n < (1.4 + 2/(dim+1)) * chi_n else 0
            
            delta_term = (1 - (1 - c_c)**2) * h_sigma * ((np.linalg.norm(pc) / chi_n)**2 - 1)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + delta_term * np.eye(dim))
            
            # Rank-mu update
            y_mu = np.zeros((dim, mu))
            for i in range(mu):
                y_mu[:, i] = (pop_x[i] - x_mean_old) / sigma
            C += c_mu * np.dot(y_mu, y_mu * weights)
            
            # Regularize and Cholesky
            C = (C + C.T) / 2
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-8 * np.eye(dim)
                C_chol = np.linalg.cholesky(C)
                
            # Stop if sigma is too small
            if sigma < 1e-8:
                break

        return best_x, best_f
