import scipy
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= budget:
                raise StopIteration("Budget exceeded")
            
            x = np.asarray(x)
            f = func(x)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Global Search with (1+1)-ES and Restarts
        # Use 70% of budget for global exploration - leave more room for local refinement
        global_budget = int(budget * 0.70)
        
        # Initial point
        x = rng.uniform(-5, 5, dim)
        try:
            fx = safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f
            
        sigma = 1.0
        stagnation = 0
        max_stagnation = 25
        min_sigma = 0.005
        max_sigma = 10.0

        for _ in range(global_budget):
            if calls >= budget or time.time() > deadline:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break
                
            if fy < fx:
                x, fx = y, fy
                # Successful step: increase sigma slightly
                sigma = min(sigma * 1.08, max_sigma)
                stagnation = 0
            else:
                # Failed step: decrease sigma
                sigma = max(sigma * 0.92, min_sigma)
                stagnation += 1
                
                # If stagnating, restart with better strategy
                if stagnation > max_stagnation:
                    # Restart from a diverse point: mix of best and random
                    if best_x is not None:
                        # 40% chance to restart near best, 60% from diverse random
                        if rng.random() < 0.4:
                            # Perturb best with moderate spread
                            x = best_x + 2.0 * rng.standard_normal(dim)
                            x = np.clip(x, -5, 5)
                        else:
                            # Diverse restart: pure random or Latin hypercube-inspired
                            if rng.random() < 0.7:
                                x = rng.uniform(-5, 5, dim)
                            else:
                                # Structured random: use different random seed pattern
                                x = 5.0 * (2.0 * rng.random(dim) - 1.0)
                            x = np.clip(x, -5, 5)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    else:
                        x = rng.uniform(-5, 5, dim)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    
                    # Reset parameters for fresh start
                    sigma = 3.0
                    stagnation = 0
                    max_stagnation = max(10, max_stagnation // 2)

        # Phase 2: Bayesian Optimization with Gaussian Process surrogate
        # Uses remaining budget to iteratively maximize expected improvement
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 20:
                # Collect existing data points from all calls made so far
                # We need to track all evaluated points, but since we don't have that,
                # we'll use a fresh BO approach with the best point as seed
                
                try:
                    # Initialize with best point and some random points around it
                    X_data = []
                    y_data = []
                    
                    # Add best point
                    X_data.append(best_x.copy())
                    y_data.append(best_f)
                    
                    # Add a few random points near best to bootstrap the GP
                    n_bootstrap = min(10, remaining_calls // 5)
                    for _ in range(n_bootstrap):
                        if calls >= budget or time.time() > deadline:
                            break
                        x_new = best_x + 0.5 * rng.standard_normal(dim)
                        x_new = np.clip(x_new, -5, 5)
                        try:
                            f_new = safe_func(x_new)
                        except (TimeoutError, StopIteration):
                            break
                        X_data.append(x_new.copy())
                        y_data.append(f_new)
                    
                    X_data = np.array(X_data)
                    y_data = np.array(y_data)
                    
                    # Normalize y for numerical stability
                    y_mean = np.mean(y_data)
                    y_std = np.std(y_data) + 1e-10
                    y_norm = (y_data - y_mean) / y_std
                    
                    # GP hyperparameters (RBF kernel)
                    length_scale = 1.0
                    signal_variance = 1.0
                    noise_variance = 1e-6
                    
                    def gp_predict(X_new, X_train, y_train, length_scale, signal_var, noise_var):
                        """Gaussian Process prediction with RBF kernel"""
                        # Compute kernel matrix between training points
                        X_train_scaled = X_train / length_scale
                        diff_train = X_train_scaled[:, np.newaxis, :] - X_train_scaled[np.newaxis, :, :]
                        K_train = signal_var * np.exp(-0.5 * np.sum(diff_train ** 2, axis=-1))
                        K_train += noise_var * np.eye(len(X_train))
                        
                        # Cholesky decomposition for stability
                        try:
                            L = np.linalg.cholesky(K_train)
                        except np.linalg.LinAlgError:
                            # Add jitter if Cholesky fails
                            K_train += 1e-6 * np.eye(len(X_train))
                            L = np.linalg.cholesky(K_train)
                        
                        # Compute alpha
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                        
                        # Predict at new points
                        X_new_scaled = X_new / length_scale
                        K_new = signal_var * np.exp(-0.5 * np.sum((X_new_scaled[:, np.newaxis, :] - X_train_scaled[np.newaxis, :, :]) ** 2, axis=-1))
                        
                        mu = K_new @ alpha
                        v = np.linalg.solve(L, K_new.T)
                        var = signal_var - np.sum(v ** 2, axis=0)
                        var = np.maximum(var, 1e-10)
                        
                        return mu, np.sqrt(var)
                    
                    # Expected Improvement acquisition function
                    def expected_improvement(X_eval, y_best, mu, sigma):
                        """Compute EI for each point in X_eval"""
                        sigma = np.maximum(sigma, 1e-8)
                        improvement = y_best - mu
                        z = improvement / sigma
                        EI = improvement * norm.cdf(z) + sigma * norm.pdf(z)
                        return EI
                    
                    # Iterative BO loop
                    bo_iterations = 0
                    max_bo_iterations = min(remaining_calls - len(X_data), 100)
                    
                    while bo_iterations < max_bo_iterations:
                        if calls >= budget or time.time() > deadline:
                            break
                        
                        # Generate candidate points for acquisition
                        n_candidates = 200
                        X_candidates = rng.uniform(-5, 5, (n_candidates, dim))
                        
                        # Predict mean and variance
                        mu, sigma = gp_predict(X_candidates, X_data, y_norm, length_scale, signal_variance, noise_variance)
                        
                        # Compute EI
                        y_best_norm = np.min(y_norm)
                        EI = expected_improvement(X_candidates, y_best_norm, mu, sigma)
                        
                        # Select point with maximum EI
                        idx_best = np.argmax(EI)
                        x_new = X_candidates[idx_best]
                        
                        # Evaluate
                        try:
                            f_new = safe_func(x_new)
                        except (TimeoutError, StopIteration):
                            break
                        
                        # Update data
                        X_data = np.vstack([X_data, x_new])
                        y_data = np.append(y_data, f_new)
                        y_norm = (y_data - y_mean) / y_std
                        
                        # Simple hyperparameter adaptation based on progress
                        if bo_iterations % 10 == 0 and bo_iterations > 0:
                            # Adjust length scale based on spread of data
                            y_range = np.max(y_data) - np.min(y_data)
                            if y_range < 1e-3:
                                length_scale = max(0.5, length_scale * 0.8)
                            elif y_range > 10:
                                length_scale = min(5.0, length_scale * 1.2)
                        
                        bo_iterations += 1
                        
                        # If we're making good progress, continue; if stuck, break
                        if bo_iterations > 5:
                            recent_improvements = np.diff(y_data[-6:])
                            if np.all(recent_improvements > -1e-4):
                                # Not making progress, break
                                break
                except (TimeoutError, StopIteration, Exception):
                    pass
        
        # Phase 3: Final Polishing with adaptive (1+1)-ES using success-rate-based sigma adaptation
        # Sometimes BO gets stuck in a local minimum or plateaus
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 10:
                x = best_x.copy()
                fx = best_f  # best_f is already the value at best_x
                
                # Start with moderate sigma and adapt using success-rate-based rule
                sigma = 0.15
                success_count = 0
                total_count = 0
                window_size = 15
                target_success_rate = 0.33
                min_sigma = 0.001
                max_sigma = 1.0
                improvement_threshold = 1e-12
                
                for _ in range(remaining_calls):
                    if calls >= budget or time.time() > deadline:
                        break
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    try:
                        fy = safe_func(y)
                    except (TimeoutError, StopIteration):
                        break
                    
                    total_count += 1
                    if fy < fx - improvement_threshold:
                        x, fx = y, fy
                        success_count += 1
                    else:
                        pass
                    
                    # Adapt sigma based on success rate over recent window
                    if total_count >= window_size:
                        success_rate = success_count / total_count
                        if success_rate > target_success_rate:
                            sigma = min(sigma * 1.1, max_sigma)
                        elif success_rate < target_success_rate * 0.5:
                            sigma = max(sigma * 0.8, min_sigma)
                        else:
                            sigma = max(sigma * 0.95, min_sigma)
                        
                        # Reset counters for next window
                        success_count = 0
                        total_count = 0
                    
                    # If sigma becomes very small and no progress, try a kick
                    if sigma < 0.005 and total_count == 0:
                        # Try a few random kicks to escape potential plateau
                        for _kick in range(3):
                            if calls >= budget or time.time() > deadline:
                                break
                            kick_x = x + 0.5 * rng.standard_normal(dim)
                            kick_x = np.clip(kick_x, -5, 5)
                            try:
                                kick_f = safe_func(kick_x)
                            except (TimeoutError, StopIteration):
                                break
                            if kick_f < fx:
                                x, fx = kick_x, kick_f
                                sigma = 0.1
                                break

        return best_x, best_f
