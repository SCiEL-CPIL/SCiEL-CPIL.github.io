import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        # For dim=5, lambda=10 is a good standard choice
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Strategy Parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        # Initial State
        x = np.zeros(dim)
        sigma = 2.0  # Large enough to explore [-5, 5]^5
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        def time_ok():
            return time.time() < deadline

        while calls < budget and time_ok():
            # Generate lam candidates
            candidates = []
            for i in range(lam):
                if not time_ok() or calls >= budget:
                    break
                z = np.random.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                fy = func(y)
                calls += 1
                candidates.append((fy, y))
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(candidates) < mu:
                break

            # Sort by fitness
            candidates.sort(key=lambda t: t[0])
            y_best = np.array([c[1] for c in candidates[:mu]])
            
            # Weighted mean
            m = np.sum(weights[:, None] * y_best, axis=0)

            # Update evolution paths
            old_sigma = sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (m - x)))
            
            # Hessian update check
            # Note: The condition for delta_h is typically based on ||ps|| / sqrt(1 - (1-c_sigma)^(2*t))
            # We approximate t as the number of generations or just use a standard check.
            # For simplicity and robustness, we can use the standard CMA-ES update.
            
            # Calculate delta_h
            # The standard check is:
            # if ||ps|| / sqrt(1 - (1 - c_sigma)^(2 * t)) < (1.4 + 2 / (dim + 1)) * chi_n:
            #     delta_h = 0
            # else:
            #     delta_h = 1
            
            # We need to track the generation count for the denominator.
            # Let's track generations.
            # Actually, simpler: just update pc and C every step as per standard algorithm.
            
            pc = (1 - c_c) * pc + c_c * (m - x) / sigma
            
            # Covariance Matrix Update
            # C = B D^2 B^T
            # New C = (1 - c_c * (2 - c_c + delta_h)) * C + c_c * (2 - c_c + delta_h) * pc * pc^T
            # To avoid full matrix multiplication every step (expensive), we can update B and D via eigen-decomposition.
            # However, dim=5 is small, so full matrix is fine.
            
            # Let's maintain C directly for clarity, or stick to B, D.
            # Using B, D is more stable.
            # C = B @ np.diag(D**2) @ B.T
            # But we can update the "step size" matrix.
            
            # Standard update:
            # C_new = (1 - c_c * (2 - c_c + delta_h)) * C_old + c_c * (2 - c_c + delta_h) * pc * pc^T
            # Then C_new = B_new @ diag(D_new^2) @ B_new^T
            
            # Let's use the C matrix representation for simplicity in code, even if slightly less efficient.
            # C = np.eye(dim) initially.
            # But we started with B, D. Let's convert to C.
            
            # Actually, let's just use a well-tested CMA-ES snippet.
            
            # Recalculate C from B, D
            C = B @ np.diag(D**2) @ B.T
            
            # Determine delta_h
            # We need a counter for the path length normalization.
            # Let's use a simple approximation: if the path is too long, it's diverging.
            # Standard CMA-ES uses the exact formula.
            # Let's assume delta_h = 0 for now, as it's rare in short runs unless diverging.
            # A more robust check:
            # norm_ps = np.linalg.norm(ps)
            # # This requires tracking the generation number t.
            # t_gen = calls // lam
            # # The term sqrt(1 - (1-c_sigma)^(2*t_gen)) approaches 1.
            # # Let's just use 1 for the denominator if t_gen is large, or calculate it.
            
            # For a 500 budget, we have ~50 generations.
            # (1 - c_sigma)^(100) is very small. So the denominator is close to 1.
            # Threshold: (1.4 + 2/(dim+1)) * chi_n
            threshold = (1.4 + 2 / (dim + 1)) * chi_n
            if np.linalg.norm(ps) < threshold:
                delta_h = 0
            else:
                delta_h = 1
                
            # Update C
            C = (1 - c_c * (2 - c_c + delta_h)) * C + c_c * (2 - c_c + delta_h) * np.outer(pc, pc)
            
            # Eigen decomposition to get new B and D
            D_sq, B = np.linalg.eigh(C)
            D_sq = np.clip(D_sq, 1e-10, None)
            D = np.sqrt(D_sq)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            x = m

        return best_x, best_f
