"""Optimizer for BBOB-style f17 (Schaffers F7) using Bayesian Optimization with GP surrogate."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimizer using Gaussian Process surrogate for low-dimensional continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # safety margin below 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        # Box constraints
        lb = -5.0
        ub = 5.0

        # Initial sampling: Latin hypercube + center + random
        # We need at least ~10-15 points for GP to be meaningful
        n_init = min(15, budget // 10)
        
        # Generate Latin hypercube samples
        def latin_hypercube(n, dim, rng):
            points = np.zeros((n, dim))
            for d in range(dim):
                # Stratified sampling in [0,1]
                indices = rng.permutation(n)
                points[:, d] = (indices + rng.uniform(size=n)) / n
            # Scale to [-5, 5]
            points = points * (ub - lb) + lb
            return points

        # Combine initial points
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Latin hypercube
        lhd = latin_hypercube(n_init - 1, dim, rng)
        init_points.extend(lhd)
        # A few random points
        n_rand = min(5, budget - len(init_points))
        if n_rand > 0:
            init_points.extend(rng.uniform(lb, ub, size=(n_rand, dim)))

        # Evaluate initial points
        X = np.array(init_points)
        fvals = np.array([func(x) for x in X])
        
        calls = len(X)
        best_idx = np.argmin(fvals)
        best_x = X[best_idx].copy()
        best_f = fvals[best_idx]

        # GP surrogate model parameters
        # Use a simple RBF kernel with automatic relevance determination (ARD)
        # y ~ N(m(x), k(x, x'))
        # We'll standardize y for better GP performance
        
        y_mean = fvals.mean()
        y_std = fvals.std() + 1e-10
        y_norm = (fvals - y_mean) / y_std

        # Hyperparameters for GP
        # Length scales (ARD), signal variance, noise variance
        # Initialize length scales to spread across the domain
        length_scales = np.ones(dim) * 2.0
        signal_var = 1.0
        noise_var = 1e-4

        # We will optimize these hyperparameters by maximizing log marginal likelihood
        # But for speed, we'll use a fixed set or simple optimization
        
        def rbf_kernel(X1, X2, length_scales, signal_var):
            # X1: (n1, dim), X2: (n2, dim)
            # Compute squared distances with ARD
            # d(x1, x2)^2 = sum_d ((x1_d - x2_d)^2 / l_d^2)
            X1_scaled = X1 / length_scales
            X2_scaled = X2 / length_scales
            # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1.x2
            X1_sq = np.sum(X1_scaled ** 2, axis=1)
            X2_sq = np.sum(X2_scaled ** 2, axis=1)
            dist_sq = X1_sq[:, None] + X2_sq[None, :] - 2 * X1_scaled @ X2_scaled.T
            dist_sq = np.maximum(dist_sq, 0)  # numerical stability
            return signal_var * np.exp(-0.5 * dist_sq)

        def compute_log_mll(X, y_norm, length_scales, signal_var, noise_var):
            n = len(X)
            K = rbf_kernel(X, X, length_scales, signal_var) + noise_var * np.eye(n)
            try:
                L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                K = K + 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return -1e10
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            mll = -0.5 * (y_norm @ alpha + log_det + n * np.log(2 * np.pi))
            return mll

        # Optimize hyperparameters once at the beginning
        def neg_log_mll(params):
            # params: [log(length_scales), log(signal_var), log(noise_var)]
            try:
                log_ls = params[:dim]
                log_sv = params[dim]
                log_nv = params[dim + 1]
                ls = np.exp(log_ls)
                sv = np.exp(log_sv)
                nv = np.exp(log_nv)
                # Clip to reasonable ranges
                ls = np.clip(ls, 0.1, 10.0)
                sv = np.clip(sv, 1e-4, 100.0)
                nv = np.clip(nv, 1e-8, 10.0)
                return -compute_log_mll(X, y_norm, ls, sv, nv)
            except Exception:
                return 1e10

        # Optimize hyperparameters
        initial_params = np.concatenate([np.log(length_scales), np.log(signal_var), np.log(noise_var)])
        try:
            res = minimize(neg_log_mll, initial_params, method='L-BFGS-B',
                           bounds=[(np.log(0.1), np.log(10.0))] * dim + [(np.log(1e-4), np.log(100.0)), (np.log(1e-8), np.log(10.0))],
                           options={'maxiter': 50})
            if res.success:
                log_ls = res.x[:dim]
                log_sv = res.x[dim]
                log_nv = res.x[dim + 1]
                length_scales = np.exp(log_ls)
                signal_var = np.exp(log_sv)
                noise_var = np.exp(log_nv)
                length_scales = np.clip(length_scales, 0.1, 10.0)
                signal_var = np.clip(signal_var, 1e-4, 100.0)
                noise_var = np.clip(noise_var, 1e-8, 10.0)
        except Exception:
            pass  # Keep initial hyperparameters

        # Main BO loop
        while calls < budget:
            # Check deadline
            if time.time() > deadline - 30.0:
                break

            # Update GP model with all data
            n = len(X)
            if n < 2:
                # Not enough data, sample randomly
                x_new = rng.uniform(lb, ub, size=dim)
            else:
                # Re-standardize y
                y_mean = fvals.mean()
                y_std = fvals.std() + 1e-10
                y_norm = (fvals - y_mean) / y_std

                # Compute K and K_inv
                K = rbf_kernel(X, X, length_scales, signal_var) + noise_var * np.eye(n)
                try:
                    L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    K = K + 1e-6 * np.eye(n)
                    try:
                        L = np.linalg.cholesky(K)
                    except np.linalg.LinAlgError:
                        # Fallback: random sampling
                        x_new = rng.uniform(lb, ub, size=dim)
                        fy = func(x_new)
                        calls += 1
                        X = np.vstack([X, x_new])
                        fvals = np.append(fvals, fy)
                        if fy < best_f:
                            best_f = fy
                            best_x = x_new.copy()
                        continue

                # For acquisition, we need to evaluate EI at candidate points
                # Use a grid or random candidates
                n_candidates = 200
                candidates = rng.uniform(lb, ub, size=(n_candidates, dim))
                
                # Compute K_star = k(x_new, X) for all candidates
                # K_star: (n_candidates, n)
                X_scaled = X / length_scales
                candidates_scaled = candidates / length_scales
                X_sq = np.sum(X_scaled ** 2, axis=1)
                cand_sq = np.sum(candidates_scaled ** 2, axis=1)
                dist_sq = cand_sq[:, None] + X_sq[None, :] - 2 * candidates_scaled @ X_scaled.T
                dist_sq = np.maximum(dist_sq, 0)
                K_star = signal_var * np.exp(-0.5 * dist_sq)  # (n_candidates, n)
                
                # Compute K_inv = K^{-1}
                K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
                
                # Predictive mean and variance
                # mu_star = K_star @ K_inv @ y_norm
                # var_star = k(x, x) - K_star @ K_inv @ K_star.T
                # k(x, x) = signal_var for RBF
                mu_star = K_star @ (K_inv @ y_norm)  # (n_candidates,)
                var_star = signal_var - np.sum(K_star @ (K_inv @ K_star.T), axis=1)  # (n_candidates,)
                var_star = np.maximum(var_star, 1e-10)
                std_star = np.sqrt(var_star)
                
                # Expected Improvement acquisition
                # EI(x) = (mu_best - mu_star - xi) * Phi((mu_best - mu_star - xi)/sigma) + sigma * phi((mu_best - mu_star - xi)/sigma)
                # where mu_best is the best observed value (in normalized space)
                mu_best = y_norm[best_idx]
                xi = 0.01  # exploration parameter
                
                z = (mu_best - mu_star - xi) / std_star
                ei = (mu_best - mu_star - xi) * norm.cdf(z) + std_star * norm.pdf(z)
                # Handle cases where std_star is very small
                mask = std_star < 1e-8
                ei[mask] = np.maximum(mu_best - mu_star[mask] - xi, 0)
                
                # Select candidate with highest EI
                best_cand_idx = np.argmax(ei)
                x_new = candidates[best_cand_idx]

            # Evaluate new point
            fy = func(x_new)
            calls += 1
            
            # Update data
            X = np.vstack([X, x_new])
            fvals = np.append(fvals, fy)
            
            # Update best
            if fy < best_f:
                best_f = fy
                best_x = x_new.copy()
                best_idx = len(X) - 1

            # Occasionally re-optimize hyperparameters (every 10 iterations)
            if len(X) % 10 == 0 and len(X) > 20:
                y_mean = fvals.mean()
                y_std = fvals.std() + 1e-10
                y_norm = (fvals - y_mean) / y_std
                
                def neg_log_mll_reopt(params):
                    try:
                        log_ls = params[:dim]
                        log_sv = params[dim]
                        log_nv = params[dim + 1]
                        ls = np.exp(log_ls)
                        sv = np.exp(log_sv)
                        nv = np.exp(log_nv)
                        ls = np.clip(ls, 0.1, 10.0)
                        sv = np.clip(sv, 1e-4, 100.0)
                        nv = np.clip(nv, 1e-8, 10.0)
                        return -compute_log_mll(X, y_norm, ls, sv, nv)
                    except Exception:
                        return 1e10

                current_params = np.concatenate([np.log(length_scales), np.log(signal_var), np.log(noise_var)])
                try:
                    res = minimize(neg_log_mll_reopt, current_params, method='L-BFGS-B',
                                   bounds=[(np.log(0.1), np.log(10.0))] * dim + [(np.log(1e-4), np.log(100.0)), (np.log(1e-8), np.log(10.0))],
                                   options={'maxiter': 30})
                    if res.success:
                        log_ls = res.x[:dim]
                        log_sv = res.x[dim]
                        log_nv = res.x[dim + 1]
                        length_scales = np.exp(log_ls)
                        signal_var = np.exp(log_sv)
                        noise_var = np.exp(log_nv)
                        length_scales = np.clip(length_scales, 0.1, 10.0)
                        signal_var = np.clip(signal_var, 1e-4, 100.0)
                        noise_var = np.clip(noise_var, 1e-8, 10.0)
                except Exception:
                    pass

        # Polish phase: local refinement using remaining budget
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline - 5.0:
            # Use Nelder-Mead or similar local search around best_x
            local_best_x = best_x.copy()
            local_best_f = best_f
            
            # Try a few local optimizations
            n_local = min(remaining // 10, 5)
            for i in range(n_local):
                if time.time() > deadline - 2.0:
                    break
                
                # Perturb initial point
                x0 = local_best_x + 0.1 * rng.standard_normal(dim)
                x0 = np.clip(x0, lb, ub)
                
                def local_func(x):
                    nonlocal calls
                    x = np.clip(x, lb, ub)
                    fx = func(x)
                    calls += 1
                    return fx
                
                try:
                    res = minimize(local_func, x0, method='Nelder-Mead',
                                   options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-6})
                    if res.fun < local_best_f:
                        local_best_f = res.fun
                        local_best_x = np.clip(res.x, lb, ub)
                except Exception:
                    pass
                
                if calls >= budget:
                    break
            
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

        return best_x, best_f
