"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with gradient-assisted initialization and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Safe wrapper that clips and times
        def safe_func(x):
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        # --- Phase 1: Gradient-based initialization to find a good starting point ---
        # Use central finite differences to estimate gradient, then do a few
        # gradient descent steps with backtracking line search.
        x = rng.uniform(-4, 4, dim)
        fx = safe_func(x)
        best_x = x.copy()
        best_fx = fx

        # Estimate gradient via central differences
        eps = 1e-5
        def estimate_grad(x0):
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x0.copy()
                xm = x0.copy()
                xp[i] += eps
                xm[i] -= eps
                grad[i] = (safe_func(xp) - safe_func(xm)) / (2 * eps)
            return grad

        # Gradient descent with backtracking line search
        lr = 0.5
        for gd_step in range(30):
            if time.time() > deadline:
                break
            grad = estimate_grad(x)
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-8:
                break
            # Backtracking line search
            step_dir = -grad / gnorm
            alpha = lr
            improved = False
            for bt in range(20):
                x_try = np.clip(x + alpha * step_dir, -5, 5)
                fx_try = safe_func(x_try)
                if fx_try < fx:
                    x = x_try
                    fx = fx_try
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                lr *= 0.5
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx
            if time.time() > deadline:
                break

        # --- Phase 2: CMA-ES from the best point found ---
        # CMA-ES parameters
        n = dim
        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        # Step size
        sigma = 1.0
        # Initial mean
        m = best_x.copy()
        # Covariance matrix
        C = np.eye(n)
        # Eigen decomposition cache
        eigvals, eigvecs = np.linalg.eigh(C)
        # Step size control
        cs = 1 / (n + 4)
        ds = 1 + 2 * max(0, (n + 1) / 3 - 1) + cs
        # Weighting
        w = np.log((n + 1) / 2) - np.log(np.arange(1, lam + 1))
        w /= w.sum()
        mu = int(lam // 2)
        mu_w = w[:mu]
        mu_w /= mu_w.sum()
        mueff = 1 / mu_w.sum()
        # Damping
        c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        # Hessian update
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        ps = np.zeros(n)
        # Restart / stagnation handling
        stagnation_count = 0
        max_stagnation = 50

        # Track calls used
        calls_used = 30 * 2 * n + 1  # approximate: gradient calls + initial
        # Actually let's count more carefully. We did at most 30 steps * 2*n gradient evals
        # but some may have been cut off. Let's just track remaining budget.
        remaining = budget - calls_used
        if remaining < lam:
            remaining = budget - calls_used  # might be negative, we'll handle

        # CMA-ES main loop
        while remaining > 0 and time.time() < deadline:
            # Check stagnation
            if stagnation_count > max_stagnation:
                # Restart with larger sigma or random restart
                sigma *= 2.0
                stagnation_count = 0
                # Small random perturbation
                m = np.clip(m + 0.5 * rng.standard_normal(n), -5, 5)

            # Generate population
            # Compute C^(1/2)
            # Re-decompose periodically
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            ys = []
            fs = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                zi = rng.standard_normal(n)
                yi = np.clip(m + sigma * (C_sqrt @ zi), -5, 5)
                fi = safe_func(yi)
                ys.append(yi)
                fs.append(fi)
                remaining -= 1

            if len(ys) == 0:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order]
            fs_sorted = [fs[i] for i in order]

            # Update best
            if fs_sorted[0] < best_fx:
                best_x = ys_sorted[0].copy()
                best_fx = fs_sorted[0]
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Update mean
            y_w = np.array(ys_sorted[:mu])
            new_m = np.sum(mu_w[:, None] * y_w, axis=0)

            # Update evolution path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt_inv @ (new_m - m)) / sigma

            # Update covariance
            delta_hess = (1 + cs * ds - cs) * chi_n - np.linalg.norm(ps) / chi_n
            rank_one = np.outer(ps, ps)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                di = (y_w[i] - m) / sigma
                rank_mu += mu_w[i] * np.outer(di, di)

            C = (1 + c1 * delta_hess + cmu - c1) * C + c1 * rank_one + cmu * rank_mu

            m = new_m
            # Keep m in bounds
            m = np.clip(m, -5, 5)

            # Adaptive sigma (1/5 rule variant)
            # Simple: if best improved recently, keep sigma; else reduce
            if stagnation_count > 10:
                sigma *= 0.9
            elif stagnation_count == 0:
                sigma = min(sigma * 1.05, 5.0)

            # Periodic re-centering if m drifts
            if np.linalg.norm(m - best_x) > 3 * sigma:
                m = 0.7 * m + 0.3 * best_x

        return best_x, best_fx
# EVOLVE-BLOCK-END
