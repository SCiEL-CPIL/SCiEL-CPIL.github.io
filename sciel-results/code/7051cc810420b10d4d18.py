"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-stage optimizer: Latin Hypercube bracketing + CMA-ES exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 290  # Safety margin before 300s limit

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            if x.shape != (dim,):
                x = x.reshape(dim)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Stage 1: Coarse Latin Hypercube sampling to bracket global basin
        # Use 3 levels per dimension -> 3^5 = 243 points is too many
        # Use a stratified sample: 2 levels per dim would be 2^5=32, 
        # but let's do a smarter LHS with ~20 points
        n_lhs = 20
        # Generate Latin Hypercube points in [0,1]^dim
        lhs_points = []
        for i in range(n_lhs):
            pt = np.zeros(dim)
            for d in range(dim):
                # For each dimension, divide [0,1] into n_lhs strata
                # Pick one point from each stratum
                stratum = (i + d) % n_lhs  # simple cyclic LHS
                pt[d] = (stratum + rng.uniform()) / n_lhs
            lhs_points.append(pt)
        
        lhs_points = np.array(lhs_points)
        # Map from [0,1]^dim to [-4, 4]^dim (minimizer lies in [-4,4]^5)
        lhs_points = lhs_points * 8.0 - 4.0
        lhs_points = np.clip(lhs_points, -5, 5)

        # Evaluate LHS points
        for pt in lhs_points:
            if calls_made >= budget or time.time() > self.deadline:
                break
            safe_call(pt)

        # Stage 2: CMA-ES initialized at best point with small sigma
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        if calls_made >= budget or time.time() > self.deadline:
            return best_x, best_f

        # CMA-ES parameters for exploitation
        center = best_x.copy()
        # Small initial sigma for exploitation
        sigma = 0.3
        lambda_ = 20  # population size
        mu = lambda_ // 2
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_C = np.zeros(dim)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim * dim))

        # Run CMA-ES until budget exhausted
        while calls_made < budget and time.time() < self.deadline:
            # Compute inverse square root of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Generate and evaluate population
            ys = []
            for i in range(lambda_):
                if calls_made >= budget or time.time() > self.deadline:
                    break
                z = rng.standard_normal(dim)
                y = center + sigma * (D_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            if len(ys) == 0:
                break

            # Evaluate all points in this generation
            fs = []
            for y in ys:
                if calls_made >= budget or time.time() > self.deadline:
                    break
                fs.append(safe_call(y))
            
            if len(fs) < mu:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys_arr = np.array(ys)
            fs_arr = np.array(fs)
            ys_sorted = ys_arr[idx]
            fs_sorted = fs_arr[idx]

            # Compute new center
            new_center = np.sum(w[:len(fs_sorted)] * ys_sorted[:len(fs_sorted)], axis=0)

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                D_inv_sqrt.T @ (new_center - center) / sigma
            )
            
            # h_sigma
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 1 / dim) * chi_n:
                h_sigma = 1
            else:
                h_sigma = 0

            p_C = (1 - c_c) * p_C + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (
                D_inv_sqrt.T @ (new_center - center) / sigma
            )

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank1 = np.outer(p_C, p_C)
            rankmu = np.sum(
                w[:len(fs_sorted)][:, None, None] * np.outer(ys_sorted[:len(fs_sorted)] - center, 
                                                            ys_sorted[:len(fs_sorted)] - center),
                axis=0
            ) / (sigma ** 2)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu

            # Ensure positive definite
            eigvals_c, _ = np.linalg.eigh(C)
            if np.any(eigvals_c < 1e-14):
                C = C + 1e-14 * np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 5.0)

            # Update center
            center = new_center

        # Stage 3: Final (1+1)-ES refinement with remaining budget
        if best_x is not None and calls_made < budget and time.time() < self.deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            while calls_made < budget and time.time() < self.deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
