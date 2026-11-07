import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 280.0
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            dim = self.dim
            n_obj = self.n_obj
            budget = self.budget
            
            # State
            X_all = [] # All evaluated points
            F_all = [] # All evaluated objectives
            archive_X = []
            archive_F = []
            
            # Hyperparameters for GP
            # Matern 2.5 kernel
            # K(x, y) = sigma^2 * (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
            # d = ||x-y|| / l
            length_scale = 0.3
            sigma_f = 1.0
            sigma_n = 1e-5
            
            def matern25_kernel(X1, X2, l, sigma):
                # X1: (N, D), X2: (M, D)
                # Compute squared distances
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                dists = np.sqrt(sq_dists)
                r = dists / l
                return sigma**2 * (1 + np.sqrt(5)*r + 5/3*r**2) * np.exp(-np.sqrt(5)*r)

            def fit_gp(X, Y, l, sigma_f, sigma_n):
                # X: (N, D), Y: (N, 1)
                N = len(X)
                if N == 0:
                    return None, None
                K = matern25_kernel(X, X, l, sigma_f)
                K += sigma_n**2 * np.eye(N)
                # Cholesky
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Add jitter
                    K += 1e-6 * np.eye(N)
                    L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
                return L, alpha

            def predict_gp(L, alpha, X, X_new, l, sigma_f):
                # X: (N, D), X_new: (M, D)
                # Returns mean (M, 1), var (M,)
                if len(X) == 0:
                    return np.zeros((len(X_new), 1)), np.ones(len(X_new))
                K_s = matern25_kernel(X_new, X, l, sigma_f) # (M, N)
                v = np.linalg.solve(L, K_s.T) # (N, M)
                mean = v.T @ alpha # (M, 1)
                K_ss = sigma_f**2 * np.ones((len(X_new), 1)) # k(x, x) is constant
                var = K_ss.flatten() - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-8)
                return mean, var

            def halton(dim, n, skip=1):
                points = np.zeros((n, dim))
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                for d in range(dim):
                    p = primes[d % len(primes)]
                    for i in range(n):
                        x = 0.0
                        f = 1.0 / p
                        idx = i + skip
                        while idx > 0:
                            idx, rem = divmod(idx, p)
                            x += f * rem
                            f /= p
                        points[i, d] = x
                return points

            def is_dominated(f_new, archive_F):
                for f_old in archive_F:
                    if np.all(f_old <= f_new) and np.any(f_old < f_new):
                        return True
                return False

            def update_archive(X, F, archive_X, archive_F):
                # Add if not dominated
                if not is_dominated(F, archive_F):
                    # Remove dominated points
                    new_arch_X = []
                    new_arch_F = []
                    for i in range(len(archive_X)):
                        if not (np.all(F <= archive_F[i]) and np.any(F < archive_F[i])):
                            new_arch_X.append(archive_X[i])
                            new_arch_F.append(archive_F[i])
                    new_arch_X.append(X)
                    new_arch_F.append(F)
                    return new_arch_X, new_arch_F
                return archive_X, archive_F

            # Phase 1: Initialization
            n_init = 30
            H = halton(dim, n_init)
            X_init = lo + (hi - lo) * H
            for x in X_init:
                f = problem(x)
                X_all.append(x)
                F_all.append(f)
                archive_X, archive_F = update_archive(x, f, archive_X, archive_F)
            
            used = n_init
            rng = np.random.default_rng(42)

            # Phase 2: Surrogate Optimization
            while used < budget:
                if time.time() > deadline:
                    break
                
                # Fit GPs for each objective using ALL data
                GPs = []
                for j in range(n_obj):
                    Y_j = np.array([f[j] for f in F_all])
                    L_j, alpha_j = fit_gp(np.array(X_all), Y_j, length_scale, sigma_f, sigma_n)
                    GPs.append((L_j, alpha_j))
                
                # Generate candidates
                n_cand = 100
                # Mix of random and local perturbations around archive points
                X_cand = np.random.uniform(lo, hi, size=(n_cand, dim))
                
                # Also add perturbations around best points to exploit
                if len(archive_X) > 0:
                    # Pick 5 random archive points
                    idxs = rng.choice(len(archive_X), size=min(5, len(archive_X)), replace=False)
                    for idx in idxs:
                        x_base = archive_X[idx]
                        noise = 0.1 * (hi - lo) * rng.standard_normal(dim)
                        x_pert = np.clip(x_base + noise, lo, hi)
                        X_cand = np.vstack([X_cand, x_pert])
                
                # Evaluate acquisition
                # Acq = - sum(w_j * (mu_j - k*sigma_j))
                # We want to minimize objectives, so we want low mu.
                # We want to explore, so we want high sigma.
                # So Acq = - (sum mu_j) + lambda * (sum sigma_j)
                # Let's use adaptive weights or just uniform.
                # To cover the front, we can use multiple weight vectors.
                
                best_x = None
                best_acq = -np.inf
                
                # Use 5 weight vectors
                weights = rng.dirichlet(np.ones(n_obj), size=5)
                
                for w in weights:
                    # For this weight vector, compute scalar acq
                    mu_sum = np.zeros(len(X_cand))
                    sigma_sum = np.zeros(len(X_cand))
                    
                    for j in range(n_obj):
                        L_j, alpha_j = GPs[j]
                        mean_j, var_j = predict_gp(np.array(X_all), L_j, alpha_j, X_cand, length_scale, sigma_f)
                        mu_sum += w[j] * mean_j.flatten()
                        sigma_sum += w[j] * np.sqrt(var_j)
                    
                    # Acquisition: Minimize weighted mean, maximize weighted std
                    # Score = - mu_sum + 0.5 * sigma_sum
                    scores = -mu_sum + 0.5 * sigma_sum
                    idx_best = np.argmax(scores)
                    
                    if scores[idx_best] > best_acq:
                        best_acq = scores[idx_best]
                        best_x = X_cand[idx_best]
                
                # Evaluate best_x
                f_new = problem(best_x)
                used += 1
                X_all.append(best_x)
                F_all.append(f_new)
                archive_X, archive_F = update_archive(best_x, f_new, archive_X, archive_F)

            return archive_X
