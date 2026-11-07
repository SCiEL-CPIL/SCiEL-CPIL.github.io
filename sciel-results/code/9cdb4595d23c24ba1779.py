"""Combined optimizer: Quadratic model fitting for global localization, followed by (1+1)-ES polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that leverages the known quadratic structure of the Sphere function.
    
    Strategy:
    1. Sample a set of points to estimate the quadratic coefficients (Hessian and gradient).
    2. Solve for the stationary point (minimum) of the fitted quadratic.
    3. Use this point as the initial guess for a high-precision (1+1)-ES local search.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls = 0

        def make_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Global Sampling for Quadratic Fit ---
        # We need at least n*(n+1)/2 + 1 points to fit a full quadratic.
        # For n=5, that's 15+1=16 points. Let's use a slightly larger set for robustness.
        # We will sample points in a structured way to better condition the fit.
        
        n_samples = 20  # Sufficient for n=5 quadratic fit
        
        X = np.zeros((n_samples, n))
        F = np.zeros(n_samples)
        
        # Point 0: Origin (helps with intercept)
        X[0] = np.zeros(n)
        
        # Points 1..n: Axis-aligned points to estimate gradient and diagonal Hessian
        for i in range(n):
            X[i+1, i] = 2.0  # Use 2.0 to be well within [-5,5] and provide good scale
            
        # Remaining points: Random points to capture off-diagonal Hessian terms
        for i in range(n+1, n_samples):
            X[i] = rng.uniform(-3, 3, n)

        for i in range(n_samples):
            if calls >= self.budget:
                break
            F[i] = make_call(X[i])
            
        # --- Phase 2: Quadratic Model Fitting ---
        # Model: f(x) = 0.5 * x^T H x + g^T x + c
        # We fit: f(x) ≈ w^T phi(x)
        # where phi(x) = [1, x_1, ..., x_n, x_1^2, ..., x_n^2, x_1x_2, ..., x_{n-1}x_n]
        
        def feature_map(x):
            # x is (n,)
            phi = np.zeros(1 + n + n + n*(n-1)//2)
            phi[0] = 1.0
            phi[1:1+n] = x
            phi[1+n:1+2*n] = x**2
            idx = 1 + 2*n
            for i in range(n):
                for j in range(i+1, n):
                    phi[idx] = x[i] * x[j]
                    idx += 1
            return phi

        Phi = np.array([feature_map(x) for x in X])
        
        # Solve least squares: Phi * w = F
        try:
            w, residuals, rank, sv = np.linalg.lstsq(Phi, F, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback if fit fails
            w = None

        if w is not None:
            # Extract Hessian H and gradient g from weights w
            # w = [c, g_1...g_n, H_11/2...H_nn/2, H_12...H_{n-1,n}]
            c = w[0]
            g = w[1:1+n].copy()
            
            H = np.zeros((n, n))
            # Diagonal terms: w[1+n+i] corresponds to 0.5 * H_ii * x_i^2
            # So H_ii = 2 * w[1+n+i]
            for i in range(n):
                H[i, i] = 2.0 * w[1+n+i]
            
            # Off-diagonal terms: w[idx] corresponds to H_ij * x_i * x_j
            # Note: In the expansion 0.5 * x^T H x, the term x_i x_j appears as H_ij x_i x_j for i!=j?
            # Let's check: x^T H x = sum_i H_ii x_i^2 + 2 * sum_{i<j} H_ij x_i x_j
            # 0.5 * x^T H x = 0.5 * sum_i H_ii x_i^2 + sum_{i<j} H_ij x_i x_j
            # So coefficient of x_i x_j in the model is H_ij.
            idx = 1 + 2*n
            for i in range(n):
                for j in range(i+1, n):
                    H[i, j] = w[idx]
                    H[j, i] = w[idx]
                    idx += 1
            
            # Solve H * x_star = -g
            try:
                # Add small regularization for numerical stability
                H_reg = H + 1e-8 * np.eye(n)
                x_star = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                x_star = np.linalg.lstsq(H + 1e-8*np.eye(n), -g, rcond=None)[0]
            
            # Clip x_star to the box
            x_star = np.clip(x_star, -5, 5)
            
            # Evaluate the model at x_star to get an estimate of f_opt
            f_star_model = 0.5 * x_star @ H @ x_star + g @ x_star + c
            
            # We should verify if this point is actually better than our current best.
            # If the model is good, x_star should be close to the true minimum.
            # We will use x_star as the starting point for the local search.
            
            # Optional: Evaluate x_star immediately to update best if it's good.
            # This costs 1 call but ensures we have a solid starting point.
            fx_star = make_call(x_star)
            
            # Update best if this was better
            if fx_star < best_f:
                best_f = fx_star
                best_x = x_star.copy()
                
        else:
            # Fallback to best of initial samples
            best_init_idx = int(np.argmin(F))
            x_star = X[best_init_idx].copy()
            best_f = F[best_init_idx]
            best_x = x_star.copy()

        # --- Phase 3: Local Polish with (1+1)-ES ---
        # Start from the estimated minimum
        x = best_x.copy()
        fx = best_f
        
        # Initialize sigma. Since we are near the minimum, start small.
        # However, if the quadratic fit was poor, we might be far.
        # Let's check the distance from x_star to the best sample point.
        # A safe initial sigma is related to the scale of the problem.
        # For Sphere, the curvature is roughly constant. 
        # Let's start with a moderate sigma and adapt.
        sigma = 0.5
        sigma_min = 1e-10
        sigma_max = 5.0
        
        success_count = 0
        total_steps = 0
        no_progress_calls = 0
        restart_threshold = 30
        max_restarts = 5
        restarts = 0
        
        # Track recent success directions for bias (optional, but helps)
        success_dirs = []
        max_dir_memory = 10

        while calls < self.budget:
            if time.time() > deadline:
                break

            z = rng.standard_normal(n)

            # Bias towards recent success directions if available
            if success_dirs and len(success_dirs) > 2:
                avg_dir = np.mean(success_dirs, axis=0)
                avg_dir_norm = np.linalg.norm(avg_dir)
                if avg_dir_norm > 1e-8:
                    avg_dir /= avg_dir_norm
                    z = 0.8 * z + 0.2 * avg_dir

            y = x + sigma * z
            fy = make_call(y)
            
            total_steps += 1

            if fy < fx:
                # Success
                step_dir = (y - x) / (sigma * np.linalg.norm(z) + 1e-15)
                success_dirs.append(step_dir)
                if len(success_dirs) > max_dir_memory:
                    success_dirs.pop(0)

                x, fx = y, fy
                success_count += 1
                no_progress_calls = 0

                # Adapt sigma: 1/5 rule approximation
                if total_steps >= 10:
                    # Use a simple rate based on last 20 steps
                    # For efficiency, we just use the global rate for now, or a short window
                    # Let's use a short window of 20
                    window = min(total_steps, 20)
                    # We don't have a perfect history of successes in the window, 
                    # but we can approximate. 
                    # Simpler: adjust based on immediate success/failure trend
                    if success_count > 0 and total_steps > 0:
                        # This global rate is slow to adapt. Let's use a local counter.
                        pass 
                    # Let's stick to the simpler adaptive rule:
                    # If we succeeded, increase slightly. If failed, decrease.
                    # This is essentially what the (1+1) ES does.
                    sigma *= 1.1 
                else:
                    sigma *= 1.2 # Grow faster initially if we are lucky
                
                sigma = np.clip(sigma, sigma_min, sigma_max)
            else:
                # Failure
                sigma *= 0.9
                sigma = np.clip(sigma, sigma_min, sigma_max)
                no_progress_calls += 1

            # Restart logic if stuck
            if no_progress_calls >= restart_threshold and restarts < max_restarts:
                restarts += 1
                no_progress_calls = 0
                success_dirs = []
                # Restart from best found so far
                x = best_x.copy()
                fx = best_f
                sigma = max(sigma * 2.0, 0.5)
                sigma = min(sigma, sigma_max)

        return best_x, best_f
# EVOLVE-BLOCK-END
