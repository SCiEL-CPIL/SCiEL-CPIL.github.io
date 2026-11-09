"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """
    Portfolio CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    Runs multiple independent CMA-ES restarts with diverse starting points.
    Each restart adapts step size and covariance matrix to the landscape.
    Wall-clock deadline guard ensures we return before the 330s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s

        # Global best tracking
        best_x = np.full(dim, np.nan)
        best_f = np.inf

        def safe_call(x):
            nonlocal best_x, best_f
            try:
                fx = func(x)
                if np.isfinite(fx) and fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return np.inf

        # CMA-ES implementation with PCA-initialized covariance support
        def cma_es(x0, fx0, n_calls, seed, C_init=None):
            nonlocal best_x, best_f
            rng = np.random.default_rng(seed)
            n = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(n)))
            # Weights: log-linear
            mu = lam // 2
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = int(mu * 0.8)  # effective mu for covariance update
            
            # Initialize Covariance
            if C_init is not None:
                C = C_init.copy()
                # Ensure positive semi-definite and symmetric
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, 1e12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    Cinv_sqrt = np.eye(n)
            else:
                C = np.eye(n)
                Cinv_sqrt = np.eye(n)

            # Step size
            # If C is anisotropic, we might want a smaller initial sigma
            sigma = 1.0 
            # Scale sigma based on the trace of C if needed, but standard CMA-ES usually starts with sigma=1
            # For ill-conditioned, a smaller sigma helps avoid clipping
            
            # Evolution paths
            ps = np.zeros(n)
            pc = np.zeros(n)
            # Rates
            c_sigma = (mueff + 2) / ((n + 1.3) * (n + 1.3) + mueff)
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_covpos = (2 + 2 * mueff / n) / (n + 1.5 + 2 * mueff / n)
            c_covneg = (5 - mueff / n) / (n + 1.5 + 5 * mueff / n)
            # Chi-squared correction
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))

            calls_used = 0
            x_best_local = x0.copy()
            fx_best_local = fx0

            for gen in range(2000):
                if calls_used >= n_calls:
                    break
                if time.time() >= deadline:
                    break

                # Generate population
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C becomes non-PSD
                    C = np.eye(n)
                    Cinv_sqrt = np.eye(n)
                    B = np.eye(n)
                
                offsets = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    offsets.append(x0 + sigma * (Cinv_sqrt @ z))
                offsets = np.array(offsets)
                # Clip to box
                offsets = np.clip(offsets, -5, 5)

                # Evaluate
                fvals = np.full(lam, np.inf)
                for i in range(lam):
                    if calls_used >= n_calls:
                        break
                    if time.time() >= deadline:
                        break
                    fvals[i] = safe_call(offsets[i])
                    calls_used += 1

                if calls_used == 0: break

                # Sort
                order = np.argsort(fvals)
                fvals_s = fvals[order]
                offsets_s = offsets[order]

                # Update best
                if fvals_s[0] < fx_best_local:
                    fx_best_local = fvals_s[0]
                    x_best_local = offsets_s[0].copy()

                # Mean update
                mean_old = x0.copy()
                x0 = np.sum(w[:, None] * offsets_s[:mu], axis=0)

                # Step size control (1/5 rule via evolution path)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * \
                     (Cinv_sqrt @ ((x0 - mean_old) / sigma))
                norm_ps = np.linalg.norm(ps)
                h_sig = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used + 1)))) \
                        < (1.4 + 2 / (n + 1)) * chi_n
                if h_sig:
                    pc = (1 - c_c) * pc + c_c * (x0 - mean_old) / sigma
                else:
                    pc = (1 - c_c) * pc

                # Covariance update
                delta_h = (1 - h_sig) * (1 - (1 - c_c) ** 2)
                rank_one = np.outer(pc, pc)
                # Rank-mu
                y = (offsets_s[:mueff] - mean_old) / sigma
                rank_mu = np.sum(w[:mueff, None, None] * np.outer(y, y), axis=0)
                C = (1 + c_sigma * c_covpos * delta_h * (c_c * (2 - c_c)) + c_sigma * c_covneg - c_sigma) * C \
                    + c_sigma * c_covpos * delta_h * (c_c * (2 - c_c)) * rank_one \
                    + c_sigma * c_covneg * rank_mu

                # Ensure symmetry
                C = (C + C.T) / 2

                # Step size update
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)

                # Update Cinv_sqrt periodically
                if calls_used % (2 * lam) < lam:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.clip(eigvals, 1e-12, 1e12)
                        Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                    except np.linalg.LinAlgError:
                        pass

            return x_best_local, fx_best_local, calls_used

        # Phase 1: Initial random sampling for PCA initialization
        rng_init = np.random.default_rng(42)
        n_init_samples = 20
        init_points = rng_init.uniform(-4.5, 4.5, (n_init_samples, dim))
        init_fvals = np.array([safe_call(p) for p in init_points])
        calls_so_far = n_init_samples
        
        # Update global best
        idx_best = np.argmin(init_fvals)
        if init_fvals[idx_best] < best_f:
            best_f = init_fvals[idx_best]
            best_x = init_points[idx_best].copy()

        # Compute PCA-based covariance
        # Select top 10 points
        k_best = min(10, n_init_samples)
        top_indices = np.argsort(init_fvals)[:k_best]
        top_points = init_points[top_indices]
        
        # Center the points
        mean_top = np.mean(top_points, axis=0)
        centered = top_points - mean_top
        
        # Compute covariance
        C_pca = np.cov(centered.T)
        # Normalize to have trace 1 (scale independent)
        trace_C = np.trace(C_pca)
        if trace_C > 1e-8:
            C_pca /= trace_C
        else:
            C_pca = np.eye(dim)
            
        # Scale the covariance to match a reasonable initial step size
        # The standard deviation in each principal direction should be related to the box size
        # If we assume the funnel is roughly within [-5,5], a scale of 1.0 is reasonable for the eigenvalues
        # But we want the *shape* from PCA.
        # Let's set the eigenvalues to be proportional to the PCA variances, but scaled to a base sigma
        eigvals_pca, eigvecs_pca = np.linalg.eigh(C_pca)
        # Ensure positive
        eigvals_pca = np.clip(eigvals_pca, 1e-6, None)
        # Normalize eigenvalues so that the geometric mean is 1? Or just use them directly?
        # Using them directly might make sigma very small or large.
        # Let's normalize so that the mean eigenvalue is 1.
        eigvals_pca /= np.mean(eigvals_pca)
        
        C_init = (eigvecs_pca * eigvals_pca) @ eigvecs_pca.T
        C_init = (C_init + C_init.T) / 2

        # Phase 2: CMA-ES with PCA-initialized covariance
        # Use the best point found in Phase 1 as the starting point
        x_start = best_x.copy()
        fx_start = best_f
        
        # Allocate remaining budget to CMA-ES
        remaining_budget = budget - calls_so_far
        
        if remaining_budget > 0 and time.time() < deadline - 5.0:
            # Run main CMA-ES
            x_best_c, fx_best_c, calls_used = cma_es(
                x_start, fx_start, remaining_budget, seed=100, C_init=C_init
            )
            calls_so_far += calls_used
            
            if fx_best_c < best_f:
                best_f = fx_best_c
                best_x = x_best_c.copy()

        # Phase 3: Optional second restart with identity covariance if budget remains
        if calls_so_far < budget and time.time() < deadline - 5.0:
            # If the first run didn't converge, try a restart with a different center
            # and identity covariance (or scaled identity)
            x_start2 = best_x + rng_init.standard_normal(dim) * 0.5
            x_start2 = np.clip(x_start2, -5, 5)
            fx_start2 = safe_call(x_start2)
            calls_so_far += 1
            
            if fx_start2 < best_f:
                best_f = fx_start2
                best_x = x_start2.copy()
                
            remaining_budget2 = budget - calls_so_far
            if remaining_budget2 > 0:
                x_best_c2, fx_best_c2, calls_used2 = cma_es(
                    x_start2, fx_start2, remaining_budget2, seed=200, C_init=None
                )
                calls_so_far += calls_used2
                
                if fx_best_c2 < best_f:
                    best_f = fx_best_c2
                    best_x = x_best_c2.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
