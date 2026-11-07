"""Multi-objective optimizer on WFG9 using Bayesian Optimization with GP and Min-EI."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import scipy.linalg


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using Bayesian Optimization with
    Gaussian Process surrogates and Minimum Expected Improvement (Min-EI)
    acquisition function.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # safety margin before 330s kill

        rng = np.random.RandomState(42)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        ref_point = np.array([3.0, 5.0, 7.0])

        # Data storage
        X_obs = []
        Y_obs = []  # (N, n_obj)
        
        # Best values seen so far for EI calculation
        f_best = np.full(self.n_obj, np.inf)

        # Archive for final return
        archive_X = []
        archive_F = []

        def add_to_archive(f, x):
            # Check if dominated
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return False
            # Remove dominated
            new_F = []
            new_X = []
            for i, af in enumerate(archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_F.append(af)
                    new_X.append(archive_X[i])
            archive_F = new_F
            archive_X = new_X
            archive_F.append(f)
            archive_X.append(x)
            return True

        # GP Implementation
        def matern_kernel(X1, X2, length_scale):
            D = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            dist = np.sqrt(np.sum(D**2, axis=-1))
            # Matern 5/2
            kernel = (1 + np.sqrt(5) * dist / length_scale + 5 * (dist / length_scale)**2) * np.exp(-np.sqrt(5) * dist / length_scale)
            return kernel

        def fit_gp(Y):
            """Fit GP to Y (N, 1). Returns weights and length_scale."""
            N = len(Y)
            # Initialize length scale
            length_scale = 1.0 
            # Simple heuristic: based on data range
            if N > 1:
                dists = np.linalg.norm(X_obs[:, np.newaxis, :] - X_obs[np.newaxis, :, :], axis=-1)
                length_scale = np.mean(dists[dists > 0]) / 2.0 if np.any(dists > 0) else 1.0
                length_scale = max(0.1, min(length_scale, 10.0))
            
            K = matern_kernel(X_obs, X_obs, length_scale)
            K += 1e-6 * np.eye(N)
            
            try:
                L = scipy.linalg.cholesky(K, lower=True)
            except:
                K += 1e-4 * np.eye(N)
                L = scipy.linalg.cholesky(K, lower=True)
            
            alpha = scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, Y, lower=True), lower=False)
            return alpha, L, length_scale

        def predict_gp(X_new, alpha, L, length_scale):
            """Predict mean and variance for X_new (M, D)."""
            K_s = matern_kernel(X_new, X_obs, length_scale)
            mean = K_s @ alpha
            # Variance: k(x, x) - K_s @ K^-1 @ K_s.T
            # k(x, x) for Matern is 1
            var = 1.0 - np.sum(K_s * scipy.linalg.solve_triangular(L, K_s.T, lower=True), axis=1)
            var = np.maximum(var, 1e-8)
            return mean, np.sqrt(var)

        def calculate_ei(mean, std, f_best_k):
            """Calculate Expected Improvement for a single objective."""
            z = (f_best_k - mean) / (std + 1e-8)
            ei = (f_best_k - mean) * norm.cdf(z) + std * norm.pdf(z)
            # If mean is already better than best, EI is 0 (or negative, clip to 0)
            ei = np.maximum(ei, 0.0)
            return ei

        def acquisition_function(x):
            """Min-EI Acquisition Function."""
            x = np.array(x).reshape(1, -1)
            means, stds = predict_gp(x, alphas, Ls, length_scales)
            # means: (1, n_obj), stds: (1, n_obj)
            eis = np.zeros(self.n_obj)
            for k in range(self.n_obj):
                eis[k] = calculate_ei(means[0, k], stds[0, k], f_best[k])
            # Min-EI: We want to maximize the minimum EI
            # So we minimize the negative of the minimum EI
            return -np.min(eis)

        # 1. Initialization
        n_init = 10
        for _ in range(n_init):
            x = lo + span * rng.random_sample(self.dim)
            f = problem(x)
            X_obs.append(x)
            Y_obs.append(f)
            f_best = np.minimum(f_best, f)
            add_to_archive(f, x)
        
        used = n_init

        # 2. BO Loop
        while used < self.budget:
            if time.time() > deadline - 10:
                break

            # Fit GP for each objective
            alphas = []
            Ls = []
            length_scales = []
            for k in range(self.n_obj):
                y_k = np.array([y[k] for y in Y_obs]).reshape(-1, 1)
                alpha, L, ls = fit_gp(y_k)
                alphas.append(alpha)
                Ls.append(L)
                length_scales.append(ls)
            
            # Optimize Acquisition Function
            # Use multiple restarts
            best_x = None
            best_score = -np.inf
            
            n_restarts = 5
            for _ in range(n_restarts):
                # Start from random or current best
                if rng.random() < 0.5:
                    x0 = lo + span * rng.random_sample(self.dim)
                else:
                    # Perturb a random observed point
                    idx = rng.randint(len(X_obs))
                    x0 = X_obs[idx] + 0.1 * span * rng.standard_normal(self.dim)
                    x0 = np.clip(x0, lo, hi)
                
                res = minimize(acquisition_function, x0, method='L-BFGS-B', 
                               bounds=list(zip(lo, hi)), options={'maxiter': 50, 'gtol': 1e-4})
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = res.x

            # Evaluate best candidate
            x_new = np.clip(best_x, lo, hi)
            f_new = problem(x_new)
            X_obs.append(x_new)
            Y_obs.append(f_new)
            f_best = np.minimum(f_best, f_new)
            add_to_archive(f_new, x_new)
            used += 1

        # Return archive
        if archive_F:
            return np.array(archive_X)
        return np.array(X_obs)
# EVOLVE-BLOCK-END
