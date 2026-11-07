import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        # 1. Initial Exploration Phase
        # Use a deterministic, space-filling Halton sequence to ensure coverage
        # of the search space [-5, 5]^5. This is more robust than pure random
        # sampling for finding the basin of attraction in the first few steps.
        
        def halton(dim, n):
            """Generate n points in [0, 1]^dim using the Halton sequence."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # Start with primes 2, 3, 5, 7, 11...
                f = 1.0
                i = 1
                for _ in range(n):
                    f /= base
                    i += 1
                    while i % base == 0:
                        i //= base
                        f *= base
                    points[_, d] = f
            return points

        # We use 20 initial points. This is a small fraction of the 500 budget
        # but provides a much better starting point for the surrogate model
        # than the 10 random points in the original, and covers the space
        # more evenly.
        n_init = 20
        if budget <= n_init:
            x = np.random.uniform(-5, 5, dim)
            f = func(x)
            return x, f

        # Map Halton points from [0, 1]^dim to [-5, 5]^dim
        H = halton(dim, n_init)
        X_init = H * 10.0 - 5.0
        
        # Ensure we also evaluate the center and a few other key points
        # to handle cases where the minimum is near the center.
        key_points = [np.zeros(dim)]
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = -4.0
            p2 = np.zeros(dim); p2[i] = 4.0
            key_points.extend([p1, p2])
        
        # Combine and deduplicate (simple check)
        all_init_points = []
        for x in X_init:
            all_init_points.append(x)
        for x in key_points:
            # Check if close to existing points
            if not any(np.linalg.norm(x - y) < 0.1 for y in all_init_points):
                all_init_points.append(x)
        
        # Cap at n_init if we added too many key points, otherwise use all
        # Actually, let's just use the first n_init from the combined list
        # to keep the budget predictable.
        X_init = np.array(all_init_points[:n_init])
        
        F_init = np.array([func(x) for x in X_init])
        evals = n_init
        
        best_idx = np.argmin(F_init)
        best_x = X_init[best_idx].copy()
        best_f = F_init[best_idx]
        
        # 2. Surrogate Model Phase (Expected Improvement)
        # We use a simple Gaussian Process-like surrogate. Since we can't
        # use sklearn, we implement a basic RBF kernel model.
        # Model: f(x) ~ N(mu(x), sigma(x))
        # For a small number of points, we can use a simple weighted average
        # with distance-based weights (inverse distance weighting with uncertainty).
        
        # We will maintain a list of evaluated points.
        X_eval = X_init.copy()
        F_eval = F_init.copy()
        
        # We use a simple heuristic for the next point:
        # Maximize Expected Improvement (EI)
        # EI(x) = (best_f - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
        # Where Z = (best_f - mu(x)) / sigma(x)
        # 
        # To keep it fast and robust without full GP math (which requires
        # matrix inversion that can be unstable or slow in a loop),
        # we use a simpler "Distance to Best" + "Uncertainty" heuristic.
        # However, the prompt asks for a kriging surrogate. Let's implement
        # a basic GP with RBF kernel.
        
        def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
            """Compute RBF kernel matrix between X1 and X2."""
            # X1: (n, d), X2: (m, d)
            # dist^2 = sum(X1^2) + sum(X2^2) - 2 X1 X2^T
            sq1 = np.sum(X1**2, axis=1)[:, None]
            sq2 = np.sum(X2**2, axis=1)[None, :]
            dist_sq = sq1 + sq2 - 2 * np.dot(X1, X2.T)
            dist_sq = np.maximum(dist_sq, 0)  # Numerical stability
            return variance * np.exp(-0.5 * dist_sq / (length_scale**2))

        # Hyperparameters for the GP
        length_scale = 1.5
        variance = 1.0
        noise_var = 1e-6
        
        # Precompute kernel matrix for initial points
        K = rbf_kernel(X_eval, X_eval, length_scale, variance)
        K += noise_var * np.eye(len(X_eval))
        
        try:
            L = np.linalg.cholesky(K)
            K_inv_F = np.linalg.solve(L.T, np.linalg.solve(L, F_eval))
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            # Fallback to diagonal if Cholesky fails
            L = None
            K_inv_F = F_eval / (np.diag(K) + noise_var)
            L_inv = None

        # Sampling grid for EI maximization
        # We use a coarse grid to find the candidate with max EI,
        # then evaluate it.
        n_grid = 1000
        # Generate grid points in the box
        # To save time, we use a subset of Halton points for the grid search
        # or a random sample. Let's use random sampling for the EI maximization
        # as it's fast and effective in 5D for a coarse search.
        
        while evals < budget:
            if time.time() > deadline:
                break
            
            # If we have very few points, just sample randomly
            if len(X_eval) < 5:
                x_next = np.random.uniform(-5, 5, dim)
            else:
                # Compute EI on a set of candidate points
                # We generate 200 candidate points
                n_cand = 200
                X_cand = np.random.uniform(-5, 5, (n_cand, dim))
                
                # Compute mean and variance
                # mu(x) = K(x, X_train) @ K_inv_F
                # var(x) = K(x,x) - K(x, X_train) @ K_inv @ K(x, X_train)^T
                # Using Cholesky: K_inv = L_inv.T @ L_inv
                # var(x) = K(x,x) - sum((L_inv @ k)^2) where k = K(x, X_train)
                
                K_cand = rbf_kernel(X_cand, X_eval, length_scale, variance)
                
                if L is not None:
                    mu = K_cand @ K_inv_F
                    # Compute variance
                    # K_cand is (n_cand, n_train)
                    # L_inv is (n_train, n_train)
                    # We need (L_inv @ K_cand.T) which is (n_train, n_cand)
                    # Then var = 1 - sum((L_inv @ K_cand.T)**2, axis=0)
                    # Note: K(x,x) is variance (1.0) for RBF
                    K_cand_T = K_cand.T
                    L_inv_K = L_inv @ K_cand_T
                    var = 1.0 - np.sum(L_inv_K**2, axis=0)
                else:
                    # Fallback: simple distance-based uncertainty
                    dists = np.linalg.norm(X_cand[:, None, :] - X_eval[None, :, :], axis=2)
                    mu = np.zeros(n_cand)
                    var = np.zeros(n_cand)
                    # This fallback is poor, so we should avoid it.
                    # If L is None, we restart with a new random point.
                    x_next = np.random.uniform(-5, 5, dim)
                    f_next = func(x_next)
                    evals += 1
                    if f_next < best_f:
                        best_f = f_next
                        best_x = x_next.copy()
                    X_eval = np.vstack([X_eval, x_next])
                    F_eval = np.append(F_eval, f_next)
                    continue

                var = np.maximum(var, 1e-10)
                std = np.sqrt(var)
                
                # Expected Improvement
                # best_f is the current minimum
                # Z = (best_f - mu) / std
                Z = (best_f - mu) / std
                # Phi and phi are standard normal CDF and PDF
                # We can use scipy.stats.norm if available, but to be safe
                # we implement a simple approximation or use a lookup.
                # Actually, scipy is available.
                from scipy.stats import norm
                EI = (best_f - mu) * norm.cdf(Z) + std * norm.pdf(Z)
                
                # Select the point with max EI
                idx_max_ei = np.argmax(EI)
                x_next = X_cand[idx_max_ei]
            
            # Evaluate the next point
            f_next = func(x_next)
            evals += 1
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
            
            # Update the surrogate model
            X_eval = np.vstack([X_eval, x_next])
            F_eval = np.append(F_eval, f_next)
            
            # Recompute Cholesky and inverse
            # This is O(n^3) but n is small (<= 500).
            # To speed up, we can use the Sherman-Morrison-Woodbury update,
            # but for simplicity and robustness, we recompute.
            # However, 500^3 is 125 million ops, which is fine in Python/NumPy.
            
            K = rbf_kernel(X_eval, X_eval, length_scale, variance)
            K += noise_var * np.eye(len(X_eval))
            
            try:
                L = np.linalg.cholesky(K)
                K_inv_F = np.linalg.solve(L.T, np.linalg.solve(L, F_eval))
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset the model and use random search
                # for the rest of the budget to ensure we finish.
                while evals < budget:
                    if time.time() > deadline:
                        break
                    x_next = np.random.uniform(-5, 5, dim)
                    f_next = func(x_next)
                    evals += 1
                    if f_next < best_f:
                        best_f = f_next
                        best_x = x_next.copy()
                break

        # 3. Local Refinement
        # Use remaining budget for coordinate descent to fine-tune the best point.
        if evals < budget and time.time() < deadline:
            step = 0.5
            x = best_x.copy()
            f = best_f
            no_improve = 0
            max_no_improve = 10
            
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
                    if step < 1e-5:
                        break
                else:
                    no_improve = 0
                    step = min(step * 1.2, 1.0)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()

        return best_x, best_f
