import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # 5s safety margin before 300s limit
        self.deadline = time.time() + 295

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Random Initialization
        # Use 10 calls to find a good starting point
        init_calls = 10
        if budget <= init_calls:
            x0 = np.zeros(dim)
            f0 = func(x0)
            return x0, f0

        best_x = None
        best_f = np.inf
        for _ in range(init_calls):
            x = rng.uniform(-5.0, 5.0, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        calls = init_calls
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f

        # Phase 2: CMA-ES
        # Parameters
        pop_size = 10  # lambda
        mu = pop_size // 2  # number of points for recombination
        
        # CMA-ES Parameters
        c_sigma = 1.0 / (pop_size + 3)
        d_sigma = 1.0 + 2 * max(0, (mu - 1) / (pop_size + 1) - 1) + c_sigma
        c_c = (4 + mu / pop_size) / (pop_size + 4 + 2 * mu / pop_size)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((mu + 0.5) / (dim + 2) ** 2 + pop_size - 2))
        
        # Initial state
        mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Helper to update Cholesky decomposition
        def update_cholesky(C, B, D):
            # Ensure C is symmetric
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C)
                # B is the orthogonal matrix, D is the diagonal
                # C = B diag(D^2) B^T
                # If we use L, then C = L L^T. This is not exactly B D^2 B^T form unless we QR decompose L.
                # Standard CMA-ES uses B and D such that C = B diag(D^2) B^T.
                # We can compute B, D from Cholesky L:
                # L = Q R, then B = Q, D = diag(R)
                Q, R = np.linalg.qr(L)
                # Ensure R has positive diagonal
                d_sign = np.sign(np.diag(R))
                d_sign[d_sign == 0] = 1
                Q = Q * d_sign
                R = np.diag(d_sign) @ R
                B = Q
                D = np.sqrt(np.diag(R))
                return B, D
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity
                return np.eye(dim), np.ones(dim)

        # Initial Cholesky
        B, D = update_cholesky(C, B, D)
        
        generations = 0
        while calls < budget and time.time() < self.deadline:
            if calls + pop_size > budget:
                # Adjust pop size for last iteration if necessary, or just break
                # Simpler: just break if we can't do a full generation
                break
                
            # Sample points
            xs = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                # x = mean + sigma * B @ (D * z)
                x = mean + sigma * (B @ (D * z))
                x = np.clip(x, -5.0, 5.0)
                xs.append(x)
            
            # Evaluate
            fs = []
            for x in xs:
                if calls >= budget:
                    break
                f = func(x)
                calls += 1
                fs.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(fs) < pop_size:
                break
                
            # Sort
            order = np.argsort(fs)
            xs_sorted = [xs[i] for i in order]
            fs_sorted = [fs[i] for i in order]
            
            # Update Mean
            mean_old = mean.copy()
            mean = sum(weights[i] * xs_sorted[i] for i in range(mu))
            mean = np.clip(mean, -5.0, 5.0)
            
            # Update Step Size (Sigma)
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B @ (D @ (D @ (mean_old - mean)))
            # Note: The term (mean_old - mean) is the step direction in transformed space?
            # Standard formula: ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff)/sigma * B @ (D @ (D @ (mean_old - mean)))
            # Actually, the update is in the transformed space.
            # Let's use the standard formulation:
            # y = mean_old - mean
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B @ (D @ (D @ y))
            y = mean_old - mean
            ps = (1 - c_sigma) * ps + (np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma) * (B @ (D * (D * y)))
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (generations + 1))) - 1))
            
            # Update Covariance
            # Rank-One Update
            h_sigma = 1 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (generations + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 + c_sigma) else 0
            # pc = (1 - c_c) * pc + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (mean_old - mean) / sigma
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean_old - mean) / sigma
            
            # C = (1 + c_1 * h_sigma * (c_c * (2 - c_c)) - c_1) * C + c_1 * outer(pc, pc)
            C = (1 + c_1 * h_sigma * (c_c * (2 - c_c)) - c_1) * C + c_1 * np.outer(pc, pc)
            
            # Rank-Mu Update
            # y_i = (mean_old - x_i) / sigma for i in 0..mu-1
            # C += c_mu * sum(weights[i] * outer(y_i, y_i))
            y_sum = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (mean_old - xs_sorted[i]) / sigma
                y_sum += weights[i] * np.outer(y_i, y_i)
            C += c_mu * y_sum
            
            # Ensure C is positive definite and update B, D
            C = (C + C.T) / 2
            B, D = update_cholesky(C, B, D)
            
            generations += 1
            
            # Simple stagnation check: if sigma becomes very small, break
            if sigma < 1e-10:
                break

        return best_x, best_f
