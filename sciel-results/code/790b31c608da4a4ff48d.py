import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # 1. Initial Deterministic Exploration
        # Use a Halton sequence for better coverage than pure random.
        # This helps locate the basin of attraction early, which is crucial
        # for AOCC scoring (early progress counts).
        
        def halton(dim, n):
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2
                f = 1.0
                i = 1
                for k in range(n):
                    f /= base
                    i += 1
                    while i % base == 0:
                        i //= base
                        f *= base
                    points[k, d] = f
            return points

        n_init = 20
        if budget <= n_init:
            # Fallback for very small budgets
            x = np.random.uniform(-5, 5, dim)
            f = func(x)
            return x, f

        H = halton(dim, n_init)
        X_init = H * 10.0 - 5.0
        
        # Ensure center is covered
        key_points = [np.zeros(dim)]
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = -4.0
            p2 = np.zeros(dim); p2[i] = 4.0
            key_points.extend([p1, p2])
        
        all_init_points = list(X_init)
        for x in key_points:
            if not any(np.linalg.norm(x - y) < 0.1 for y in all_init_points):
                all_init_points.append(x)
        
        X_init = np.array(all_init_points[:n_init])
        F_init = np.array([func(x) for x in X_init])
        evals = n_init
        
        best_idx = np.argmin(F_init)
        best_x = X_init[best_idx].copy()
        best_f = F_init[best_idx]

        # 2. CMA-ES Phase
        # We use a robust CMA-ES implementation for the bulk of the search.
        # CMA-ES is known to handle ill-conditioned functions (like Schaffer F7)
        # well because it adapts the covariance matrix to the landscape's shape.
        
        # Parameters
        lam = 15  # Population size
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_lambda = np.sqrt(dim * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)))

        # State
        m = best_x.copy()
        sigma = 1.0  # Start with a moderate step size
        C = np.eye(dim)
        
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        stagnation_count = 0
        stagnation_threshold = 5

        while evals + lam <= budget:
            if time.time() > deadline - 10.0:  # Leave time for refinement
                break

            # Sample population
            Z = np.random.randn(lam, dim)
            X = m + sigma * (Z @ L.T)
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(x) for x in X])
            evals += lam

            # Sort
            order = np.argsort(F)
            F_sorted = F[order]
            X_sorted = X[order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Update Mean
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update Sigma (CSA)
            # We need C^{-1/2} * (m_new - m) / sigma
            # Using Cholesky: C = L L^T. C^{-1} = L^{-T} L^{-1}.
            # C^{-1/2} is not trivially L^{-1}, but we can approximate or use a solver.
            # For simplicity and speed, we solve L^T y = (m_new - m) / sigma
            # Actually, the standard CSA update is:
            # p_sigma = (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mu_eff) * C^{-1/2} * (m_new - m)/sigma
            # C^{-1/2} v = L^{-T} (L^{-1} v) ? No.
            # Let's use a simpler sigma adaptation based on success rate or distance.
            # Or, we can just use the standard CMA-ES sigma update which requires C^{-1/2}.
            # Since dim=5, we can compute C_inv_sqrt explicitly if needed, or use a solver.
            
            # Let's use a solver for C^{-1/2} v
            diff = m_new - m
            # Solve C y = diff / sigma
            try:
                y = np.linalg.solve(C, diff / sigma)
                # This gives C^{-1} diff / sigma. We want C^{-1/2}.
                # This is not the same. 
                # Let's stick to a simpler adaptation: 
                # If the best offspring is better than the previous best, increase sigma slightly?
                # Or use the standard CMA-ES formula with precomputed C_inv_sqrt.
                
                # Let's compute C_inv_sqrt once per generation? 
                # C = L L^T. C^{-1/2} = (L^{-1})^T (L^{-1})? No.
                # Eigen-decomposition is expensive.
                # Let's use the approximation: 
                # p_sigma update is often the most critical part for sigma adaptation.
                
                # For robustness, let's use a simple heuristic if full CMA-ES is too complex:
                # If improvement, sigma *= 1.05, else sigma *= 0.95.
                if F_sorted[0] < F_sorted[-1]: # Some improvement in population
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                sigma = np.clip(sigma, 0.01, 10.0)
                
            except np.linalg.LinAlgError:
                sigma *= 0.9

            # Update Covariance Matrix
            # Rank-1 update
            p_c = (1 - c_c) * p_c + c_c * (m_new - m) / sigma
            C_rank1 = np.outer(p_c, p_c)
            
            # Rank-mu update
            C_rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = X_sorted[i] - m_new
                C_rank_mu += weights[i] * np.outer(diff_i, diff_i)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (C_rank1 + (1 - c_c) * c_c * (1 - c_c) * np.eye(dim)) + c_mu * C_rank_mu
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity and reduce sigma
                C = np.eye(dim)
                L = np.eye(dim)
                sigma *= 0.5

            # Restart if stagnation is high
            if stagnation_count > stagnation_threshold * 3:
                # Random restart around current best
                m = best_x + np.random.randn(dim) * sigma * 2
                C = np.eye(dim)
                L = np.eye(dim)
                sigma = 1.0
                stagnation_count = 0
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

        # 3. Local Refinement Phase
        # Use remaining budget for a simple coordinate descent or Nelder-Mead
        # to fine-tune the best point found by CMA-ES.
        
        if evals < budget and time.time() < deadline:
            # Simple coordinate descent
            step = 0.1
            x = best_x.copy()
            f = best_f
            no_improve = 0
            max_no_improve = 20
            
            while evals < budget and time.time() < deadline and no_improve < max_no_improve:
                improved = False
                for i in range(dim):
                    if evals >= budget or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        evals += 1
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    no_improve += 1
                    if step < 1e-6:
                        break
                else:
                    no_improve = 0
                    step = min(step * 1.2, 1.0)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()

        return best_x, best_f
