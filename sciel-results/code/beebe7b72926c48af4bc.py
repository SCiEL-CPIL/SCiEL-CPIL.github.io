"""BBOB f22 optimizer: GP surrogate + adaptive CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Gaussian Process surrogate + CMA-ES for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 285.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0
        all_points = []  # list of (x, f)

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls_made += 1
            fx = float(func(x))
            all_points.append((x.copy(), fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # --- GP Surrogate ---
        def gp_predict(X_test, X_train, y_train, length_scale=1.0, noise=1e-8):
            """
            Predict mean and variance of GP at X_test given training data.
            Uses RBF kernel: k(x,y) = sigma_f^2 * exp(-||x-y||^2 / (2*l^2))
            """
            n = len(X_train)
            d = X_train.shape[1]

            # Compute kernel matrix K
            # K[i,j] = exp(-||x_i - x_j||^2 / (2*l^2))
            X_train = np.atleast_2d(X_train)
            X_test = np.atleast_2d(X_test)

            # Efficient distance computation
            # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y
            X_train_sq = np.sum(X_train ** 2, axis=1, keepdims=True)
            X_test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)
            dist_sq = X_test_sq + X_train_sq.T - 2 * X_test @ X_train.T
            dist_sq = np.maximum(dist_sq, 0)  # numerical stability

            K = np.exp(-dist_sq / (2 * length_scale ** 2))
            K += noise * np.eye(n)

            # Solve for alpha
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

            # Predict
            K_s = np.exp(-dist_sq / (2 * length_scale ** 2))  # K(X_test, X_train)
            mean = K_s @ alpha

            # Variance
            v = np.linalg.solve(L, K_s.T)
            var = 1.0 - np.sum(v ** 2, axis=0)  # k(x,x) = 1
            var = np.maximum(var, 1e-10)

            return mean, var

        def expected_improvement(mean, var, best_f_current, xi=0.01):
            """Compute EI for minimization."""
            eps = 1e-8
            sigma = np.sqrt(var)
            improvement = best_f_current - mean - xi
            z = improvement / (sigma + eps)
            ei = improvement * (1 + 0.5 * (z ** 2)) + sigma * (0.5 * z * np.sqrt(1 + 0.5 * (z ** 2)) - 0.5 * np.sqrt(1 + 0.5 * (z ** 2)))
            # More standard formula:
            # z = (best_f - mean - xi) / sigma
            # EI = (best_f - mean - xi) * Phi(z) + sigma * phi(z)
            # But for large z, this can be unstable. Let's use a simpler robust version.
            ei2 = np.maximum(improvement, 0)
            # Actually, let's use the standard robust EI
            sigma_safe = np.maximum(sigma, eps)
            z = improvement / sigma_safe
            # Phi(z) and phi(z)
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
            ei = np.maximum(ei, 0)
            return ei

        def acq_func(x, X_train, y_train, best_f_current, length_scale=1.0):
            """Acquisition function: Expected Improvement."""
            x = np.atleast_2d(x)
            mean, var = gp_predict(x, X_train, y_train, length_scale)
            ei = expected_improvement(mean[0], var[0], best_f_current)
            return ei

        # Phase 1: Initial sampling with LHS + a few random points
        n_init = min(40, max(20, budget // 10))
        # LHS points
        lhs_points = []
        for i in range(n_init):
            x = np.zeros(dim)
            for d in range(dim):
                x[d] = -5.0 + (i + rng.random()) * (10.0 / n_init)
            x = np.clip(x, -5, 5)
            lhs_points.append(x)
        
        # Add a few center and corner points for better coverage
        center = np.zeros(dim)
        lhs_points.append(center)
        for _ in range(5):
            x = rng.uniform(-5, 5, dim)
            lhs_points.append(x)
        
        lhs_points = np.array(lhs_points)
        
        for i in range(len(lhs_points)):
            if not check_time() or calls_made >= budget:
                break
            safe_func(lhs_points[i])

        if not all_points:
            return best_x, best_f

        # Phase 2: GP-guided search
        # Adaptive length scale: start with 1.0, adjust based on data spread
        length_scale = 1.5
        
        n_gp_steps = min(200, budget - calls_made - 50)  # Reserve 50 for CMA-ES
        for step in range(n_gp_steps):
            if not check_time() or calls_made >= budget - 50:
                break

            X_train = np.array([p[0] for p in all_points])
            y_train = np.array([p[1] for p in all_points])

            # Generate candidate points: grid + random
            # Grid in [-5, 5]^5 is too large, so use random candidates
            n_candidates = 500
            candidates = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Also add points near current best
            if best_x is not None:
                for _ in range(50):
                    x = best_x + rng.standard_normal(dim) * 0.5
                    x = np.clip(x, -5, 5)
                    candidates = np.vstack([candidates, x])

            # Compute EI for all candidates
            X_test = candidates
            mean, var = gp_predict(X_test, X_train, y_train, length_scale)
            ei = expected_improvement(mean, var, best_f)

            # Select best candidate
            idx = np.argmax(ei)
            x_new = candidates[idx]
            fx_new = safe_func(x_new)

            # Adapt length scale: if best_f improved, decrease LS (more local)
            # If no improvement, increase LS (more global)
            if len(all_points) > 10:
                # Estimate typical distance between points
                if len(all_points) > 2:
                    dists = []
                    for i in range(min(10, len(all_points))):
                        for j in range(i+1, min(10, len(all_points))):
                            dists.append(np.linalg.norm(all_points[i][0] - all_points[j][0]))
                    if dists:
                        typical_dist = np.median(dists)
                        # Keep LS proportional to typical distance
                        length_scale = np.clip(typical_dist * 0.5, 0.5, 5.0)

        # Phase 3: CMA-ES refinement on the best point found
        remaining = budget - calls_made
        if remaining > 20 and best_x is not None and check_time():
            # CMA-ES implementation
            pop_size = min(10, max(4, remaining // 5))
            max_generations = max(5, remaining // (pop_size + 1))
            
            mean = best_x.copy()
            sigma = 0.5
            C = np.eye(dim)
            pc = np.zeros(dim)
            cs = 1.0 / (dim + 3)
            c1 = 2 / ((dim + 1) ** 2 + 10)
            cmu = min(0.4 - c1, 2 / ((dim + 1) ** 2 + 5))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            for gen in range(max_generations):
                if not check_time() or calls_made >= budget:
                    break

                # Generate population
                offspring = []
                for i in range(pop_size):
                    if not check_time() or calls_made >= budget:
                        break
                    z = rng.standard_normal(dim)
                    # Cholesky for C
                    try:
                        L_c = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = C + 1e-6 * np.eye(dim)
                        L_c = np.linalg.cholesky(C)
                    y = mean + sigma * (L_c @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_func(y)
                    offspring.append((fy, y))

                if len(offspring) < pop_size:
                    break

                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()

                # Update mean
                mu_eff = max(1, pop_size // 2)
                w = np.zeros(mu_eff)
                w[:mu_eff] = np.log((mu_eff + 1) / 2) - np.log(np.arange(1, mu_eff + 1))
                w = w / w.sum()
                mu_eff = (w ** 2).sum() / (w ** 2).sum()  # This is always 1, let's fix
                mu_eff = (w.sum() ** 2) / (w ** 2).sum()

                old_mean = mean.copy()
                mean = np.zeros(dim)
                for i in range(mu_eff):
                    mean += w[i] * offspring[i][1]

                # Update step size
                y_vec = (mean - old_mean) / sigma
                pc = (1 - cs) * pc + cs * y_vec
                hsig = np.linalg.norm(pc) / chi_n * np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
                hsig = min(1.0, hsig)
                sigma *= np.exp(cs / (dim + 2) * (np.linalg.norm(pc) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 5.0)

                # Update covariance
                cc = (4 + dim / 4) / (dim + 4 + 2 * cs)
                pc = (1 - cc) * pc + cc * hsig * (mean - old_mean) / sigma
                C = (1 + cmu * hsig * (hsig - 1 + 2 * cmu) * (1 - cc) - c1 - cmu) * C
                C += cmu * np.outer(pc, pc) + c1 * (1 - hsig) * np.outer(pc, pc)
                # Actually, standard CMA-ES covariance update:
                # C = (1 + c1*(2-c1-cmu) - c1 - cmu)*C + c1*pc*pc^T + cmu*sum(w_i*y_i*y_i^T)
                # Let's simplify and just use rank-1 and rank-mu updates
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu_eff):
                    y_i = (offspring[i][1] - old_mean) / sigma
                    C += cmu * w[i] * np.outer(y_i, y_i)
                C = (C + C.T) / 2

                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
