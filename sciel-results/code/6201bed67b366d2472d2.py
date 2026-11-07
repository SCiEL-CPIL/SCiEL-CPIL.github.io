"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid: lattice initialization + CMA-ES local refinement for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Lattice initialization to find good baseline ---
        # Rastrigin has minima at integer points (in unrotated space).
        # After rotation/shift, the true minimum is near some integer grid point.
        # Evaluate a subset of integer lattice points within [-4, 4]^dim.
        # For dim=5, that's 9^5 = 59049 points - too many.
        # Instead, evaluate a coarser grid first, then refine.
        
        # Use a smaller set: integers from -4 to 4, but sample strategically.
        # Actually, let's just evaluate all integer points in [-3,3]^5 = 7^5 = 16807
        # That's still a lot. Let's do [-2,2]^5 = 5^5 = 3125 first, then expand if needed.
        
        # Better approach: evaluate a few promising points first.
        # Start with origin and nearby integers.
        
        lattice_points = []
        # Generate all integer points in [-3, 3]^dim
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        for m in range(-3, 4):
                            lattice_points.append([i, j, k, l, m])
        
        lattice_points = np.array(lattice_points, dtype=float)
        
        # Shuffle to get diversity early
        idx = rng.permutation(len(lattice_points))
        lattice_points = lattice_points[idx]
        
        # Evaluate lattice points, keeping track of best
        best_x = None
        best_f = float('inf')
        calls_made = 0
        call_count = 0
        
        for pt in lattice_points:
            if time.time() > deadline:
                break
            if call_count >= budget:
                break
            x = np.clip(pt, -5, 5)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # If we didn't use all budget on lattice, proceed to CMA-ES refinement
        remaining_budget = budget - call_count
        if remaining_budget <= 0 or best_x is None:
            return best_x, best_f

        # --- Phase 2: CMA-ES local refinement around best lattice point ---
        # Implement a simple CMA-ES
        def cma_es(x0, f0, budget_left, sigma_init=1.0):
            nonlocal call_count
            n = dim
            # CMA-ES parameters
            sigma = sigma_init
            lambda_ = int(4 + 3 * np.log(n))  # population size
            mu = lambda_ // 2
            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            
            # Covariance matrix
            C = np.eye(n)
            # Step size adaptation (1/5 rule)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            # Covariance matrix adaptation
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            
            # Mean
            m = x0.copy()
            # Evolution path for sigma
            p_sigma = np.zeros(n)
            # Evolution path for covariance
            p_c = np.zeros(n)
            # Inverse square root of C (approximate)
            C_inv_sqrt = np.eye(n)
            C_sqrt = np.eye(n)
            
            # Precompute
            def inv_sqrt(mat):
                eigvals, eigvecs = np.linalg.eigh(mat)
                eigvals = np.maximum(eigvals, 1e-10)
                return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            def sqrt_mat(mat):
                eigvals, eigvecs = np.linalg.eigh(mat)
                eigvals = np.maximum(eigvals, 1e-10)
                return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            C_inv_sqrt = inv_sqrt(C)
            C_sqrt = sqrt_mat(C)
            
            # Track best
            best_local_x = x0.copy()
            best_local_f = f0
            
            for it in range(budget_left // lambda_):
                if time.time() > deadline:
                    break
                if call_count >= budget:
                    break
                
                # Generate population
                Y = []
                for i in range(lambda_):
                    if call_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = m + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    Y.append(y)
                
                if len(Y) == 0:
                    break
                
                # Evaluate
                Fs = []
                for y in Y:
                    fy = func(y)
                    call_count += 1
                    Fs.append(fy)
                    if fy < best_local_f:
                        best_local_f = fy
                        best_local_x = y.copy()
                
                if len(Fs) < mu:
                    break
                
                # Sort by fitness
                order = np.argsort(Fs)
                Y = np.array(Y)
                Fs = np.array(Fs)
                Y = Y[order]
                Fs = Fs[order]
                
                # Update mean
                m_new = m + sigma * np.sum(weights[:mu][:, np.newaxis] * Y[:mu], axis=0)
                
                # Update sigma
                z = C_inv_sqrt @ (m_new - m) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z
                chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count - budget + 1))) > (1.4 + 1 / n) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update covariance
                v = (m_new - m) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * v
                C = (1 + c_1 * (mueff - 1) - c_1) * C \
                    + c_1 * np.outer(v, v) \
                    + c_mu * np.sum(weights[:mu][:, np.newaxis] * np.outer(Y[:mu] - m, Y[:mu] - m) / sigma ** 2, axis=0) \
                    - c_mu * np.outer(p_c, p_c)
                
                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-10:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
                
                C_inv_sqrt = inv_sqrt(C)
                C_sqrt = sqrt_mat(C)
                m = m_new
            
            return best_local_x, best_local_f

        # Run CMA-ES around the best lattice point
        best_x, best_f = cma_es(best_x, best_f, remaining_budget, sigma_init=1.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
