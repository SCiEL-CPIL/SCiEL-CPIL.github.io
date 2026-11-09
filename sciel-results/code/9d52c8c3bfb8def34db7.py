import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        eval_count = 0
        
        # Phase 1: Random sampling to get initial spread
        n_random = min(20, budget // 10)
        rng = np.random.default_rng(42)
        X_samples = []
        F_samples = []
        
        for i in range(n_random):
            if eval_count >= budget or time.time() > deadline - 5:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            eval_count += 1
            X_samples.append(x)
            F_samples.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if not X_samples:
            x = np.zeros(dim)
            fx = func(x)
            eval_count += 1
            best_f = fx
            best_x = x.copy()
            return best_x, best_f
            
        X_samples = np.array(X_samples)
        F_samples = np.array(F_samples)
        
        # Phase 2: Gaussian Process Surrogate Optimization
        # We implement a simple GP from scratch to avoid dependency issues if sklearn/scipy.gaussian_process is slow or missing,
        # but scipy is available. Let's use a simple RBF kernel GP.
        
        def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
            # X1: (n, d), X2: (m, d)
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return variance * np.exp(-0.5 * sq_dists / (length_scale**2))
        
        def predict_gp(X_test, X_train, y_train, length_scale=1.0, variance=1.0, noise=1e-8):
            K = rbf_kernel(X_train, X_train, length_scale, variance) + noise * np.eye(len(X_train))
            K_star = rbf_kernel(X_test, X_train, length_scale, variance)
            try:
                alpha = np.linalg.solve(K, y_train)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]
            mean = K_star @ alpha
            # We don't strictly need variance for EI if we just use mean, but EI is better
            K_star_star = rbf_kernel(X_test, X_test, length_scale, variance)
            var = np.diag(K_star_star - K_star @ np.linalg.solve(K, K_star.T))
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            return mean, std
        
        def expected_improvement(mean, std, best_f, xi=0.01):
            if std < 1e-8:
                return 0.0
            z = (best_f - mean - xi) / std
            # Normal CDF and PDF
            from scipy.stats import norm
            ei = (best_f - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
            return ei
        
        from scipy.stats import norm
        
        # Adaptive length scale
        # Estimate initial length scale from data spread
        if len(X_samples) > 1:
            dists = np.sqrt(np.sum((X_samples[:, None, :] - X_samples[None, :, :])**2, axis=2))
            median_dist = np.median(dists[dists > 0])
            length_scale = max(median_dist / 2.0, 0.1)
        else:
            length_scale = 1.0
        
        variance = np.var(F_samples) + 1e-8
        noise = 1e-8 * variance
        
        # Local search around best point to refine
        # We will alternate between GP-based global search and local refinement
        
        # First, do a few local refinements on the best point found so far
        def local_search(x_start, f_start, max_evals, sigma_init=0.5):
            x = x_start.copy()
            f = f_start
            sigma = sigma_init
            for _ in range(max_evals):
                if eval_count >= budget or time.time() > deadline - 5:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                eval_count += 1
                if fy < f:
                    x = y
                    f = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.7
            return x, f
        
        # Refine current best
        if eval_count < budget - 10 and time.time() < deadline - 5:
            best_x, best_f = local_search(best_x, best_f, 10, sigma_init=0.5)
            
        # Add refined point to GP
        X_samples = np.vstack([X_samples, [best_x]])
        F_samples = np.append(F_samples, [best_f])
        
        # Main loop: GP guided search
        # We will pick points using EI or Mean, and then locally refine them
        n_gp_iters = min(30, (budget - eval_count) // 5)
        
        for i in range(n_gp_iters):
            if eval_count >= budget - 10 or time.time() > deadline - 10:
                break
                
            # Generate candidate points
            # Mix of random and grid-like points for exploration
            n_candidates = 100
            X_cand = np.random.uniform(-5, 5, (n_candidates, dim))
            
            # Also include points near the current best
            for j in range(10):
                X_cand[j] = np.clip(best_x + rng.standard_normal(dim) * 0.5, -5, 5)
            
            # Predict
            mean, std = predict_gp(X_cand, X_samples, F_samples, length_scale, variance, noise)
            
            # Calculate EI
            ei = expected_improvement(mean, std, best_f, xi=0.01)
            
            # Select best candidate
            idx = np.argmax(ei)
            x_new = X_cand[idx].copy()
            
            # Evaluate
            fx_new = func(x_new)
            eval_count += 1
            
            # Update GP
            X_samples = np.vstack([X_samples, [x_new]])
            F_samples = np.append(F_samples, [fx_new])
            
            # Local search around this new point
            if fx_new < best_f + 0.1: # Only refine if promising
                if eval_count < budget - 5 and time.time() < deadline - 5:
                    x_ref, f_ref = local_search(x_new, fx_new, 5, sigma_init=0.2)
                    if f_ref < best_f:
                        best_f = f_ref
                        best_x = x_ref.copy()
                        
                        # Update GP with refined point
                        X_samples = np.vstack([X_samples, [best_x]])
                        F_samples = np.append(F_samples, [best_f])
                        
            # Adapt length scale
            # If we are improving a lot, decrease length scale (more local)
            # If stuck, increase length scale (more global)
            if i % 5 == 4:
                # Check recent improvement
                recent_f = F_samples[-10:]
                if len(recent_f) >= 5:
                    if F_samples[-1] < F_samples[-5]:
                        length_scale *= 0.8
                    else:
                        length_scale *= 1.2
                length_scale = np.clip(length_scale, 0.1, 5.0)
        
        # Final polish with local search on the global best
        if eval_count < budget and time.time() < deadline - 2:
            remaining = budget - eval_count
            # Use remaining budget for intensive local search
            # Use a simple CMA-ES like or (1+1)-ES with many restarts from best
            # Or just simple hill climbing with adaptive step
            x_curr = best_x.copy()
            f_curr = best_f
            sigma = 0.1
            while eval_count < budget and time.time() < deadline - 2:
                y = np.clip(x_curr + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                eval_count += 1
                if fy < f_curr:
                    x_curr = y
                    f_curr = fy
                    sigma *= 1.1
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x_curr.copy()
                else:
                    sigma *= 0.9
                    if sigma < 1e-4:
                        # Try a few random restarts from best_x
                        for _ in range(3):
                            if eval_count >= budget or time.time() > deadline - 2:
                                break
                            y = np.clip(best_x + rng.standard_normal(dim) * 0.5, -5, 5)
                            fy = func(y)
                            eval_count += 1
                            if fy < f_curr:
                                x_curr = y
                                f_curr = fy
                                sigma = 0.2
                                if f_curr < best_f:
                                    best_f = f_curr
                                    best_x = x_curr.copy()
                        break
                        
        return best_x, best_f
