import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        time_limit = 280.0

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_left():
            return start_time + time_limit - time.time()

        # Phase 1: Fit quadratic model to estimate Sphere minimizer
        # For f(x) = ||x - z||^2 + c, we need 2n+1 points to fit the model
        # f(x) = x^T A x + b^T x + c where A is symmetric positive definite
        # Minimizer is z = -0.5 * A^{-1} b
        
        # Use deterministic design: origin + 2n axis points
        design_points = [np.zeros(n)]
        step_sizes = [2.0, 4.0]  # Try multiple scales
        all_design = []
        
        for scale in step_sizes:
            for i in range(n):
                for sign in [1, -1]:
                    p = np.zeros(n)
                    p[i] = sign * scale
                    all_design.append(p)
        
        # Evaluate design points
        design_values = []
        for p in all_design:
            if calls >= budget or time_left() <= 0:
                break
            p_clipped = np.clip(p, -5, 5)
            fx = evaluate(p_clipped)
            design_values.append((p_clipped, fx))
        
        # Fit quadratic model using least squares
        # f(x) ≈ x^T A x + b^T x + c
        # We can use the values to estimate the gradient and Hessian
        # For Sphere: f(x) = ||x-z||^2 + c = x^T x - 2z^T x + ||z||^2 + c
        # So A = I, b = -2z, c = ||z||^2 + c_offset
        
        # Use finite differences to estimate gradient at best point
        # Then use Newton's method with estimated Hessian
        
        # First, find a good starting point from the design
        if design_values:
            design_values.sort(key=lambda t: t[1])
            x_current = design_values[0][0].copy()
        else:
            x_current = np.zeros(n)
            evaluate(x_current)
        
        # Phase 2: Estimate gradient and Hessian using finite differences
        # Use central differences for gradient, and estimate Hessian from second differences
        
        def estimate_gradient(x0, h=0.01):
            grad = np.zeros(n)
            for i in range(n):
                x_plus = x0.copy()
                x_minus = x0.copy()
                x_plus[i] += h
                x_minus[i] -= h
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                if calls >= budget or time_left() <= 0:
                    return None
                fp = evaluate(x_plus)
                if calls >= budget or time_left() <= 0:
                    return None
                fm = evaluate(x_minus)
                grad[i] = (fp - fm) / (2 * h)
            return grad
        
        def estimate_hessian(x0, h=0.01):
            H = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    if calls >= budget or time_left() <= 0:
                        break
                    x_pp = x0.copy()
                    x_pm = x0.copy()
                    x_mp = x0.copy()
                    x_mm = x0.copy()
                    x_pp[i] += h
                    x_pp[j] += h
                    x_pm[i] += h
                    x_pm[j] -= h
                    x_mp[i] -= h
                    x_mp[j] += h
                    x_mm[i] -= h
                    x_mm[j] -= h
                    
                    x_pp = np.clip(x_pp, -5, 5)
                    x_pm = np.clip(x_pm, -5, 5)
                    x_mp = np.clip(x_mp, -5, 5)
                    x_mm = np.clip(x_mm, -5, 5)
                    
                    if calls >= budget or time_left() <= 0:
                        break
                    fpp = evaluate(x_pp)
                    if calls >= budget or time_left() <= 0:
                        break
                    fpm = evaluate(x_pm)
                    if calls >= budget or time_left() <= 0:
                        break
                    fmp = evaluate(x_mp)
                    if calls >= budget or time_left() <= 0:
                        break
                    fmm = evaluate(x_mm)
                    
                    H[i, j] = (fpp - fpm - fmp + fmm) / (4 * h * h)
                    H[j, i] = H[i, j]
            return H
        
        # Phase 3: Newton refinement with gradient/Hessian estimation
        # Start from current best and iterate
        
        x_newton = x_current.copy()
        h_fd = 0.05  # Finite difference step
        
        for iteration in range(30):
            if calls >= budget or time_left() <= 1.0:
                break
            
            # Estimate gradient
            grad = estimate_gradient(x_newton, h_fd)
            if grad is None:
                break
            
            # Check if we're close to optimum
            if np.linalg.norm(grad) < 1e-6:
                break
            
            # Estimate Hessian
            H = estimate_hessian(x_newton, h_fd)
            if H is None:
                break
            
            # Ensure Hessian is positive definite
            try:
                H_inv = np.linalg.inv(H + 1e-8 * np.eye(n))
            except np.linalg.LinAlgError:
                H_inv = np.linalg.pinv(H + 1e-8 * np.eye(n))
            
            # Newton step
            step = H_inv @ grad
            # Limit step size for stability
            step_norm = np.linalg.norm(step)
            if step_norm > 1.0:
                step = step / step_norm
            
            x_candidate = x_newton - step
            x_candidate = np.clip(x_candidate, -5, 5)
            
            if calls >= budget or time_left() <= 0:
                break
            f_candidate = evaluate(x_candidate)
            
            # If candidate is better, accept it
            if f_candidate < best_f:
                x_newton = x_candidate
            else:
                # Try a smaller step (line search)
                for alpha in [0.5, 0.25, 0.1]:
                    x_try = x_newton - alpha * step
                    x_try = np.clip(x_try, -5, 5)
                    if calls >= budget or time_left() <= 0:
                        break
                    f_try = evaluate(x_try)
                    if f_try < best_f:
                        x_newton = x_try
                        break
                else:
                    # No improvement, stop
                    break
        
        # Phase 4: Final CMA-ES refinement with small sigma
        if calls < budget and time_left() > 2.0 and best_x is not None:
            lambda_ = 10
            mu = 5
            
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 0.5) / (n + 2)
            d_sigma = 1 + 2 * max(0, (np.sqrt((mu_eff + 1) / (n + 1)) - 1) / (n + 2)) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            mean = best_x.copy()
            sigma = 0.1
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            while calls < budget and time_left() > 0.5:
                offspring = []
                for i in range(lambda_):
                    if calls >= budget or time_left() <= 0.5:
                        break
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                    
                    z = np.random.standard_normal(n)
                    y = mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    
                    fy = evaluate(y)
                    offspring.append((fy, y))
                
                if len(offspring) < mu:
                    break
                    
                offspring.sort(key=lambda t: t[0])
                selected = [y for _, y in offspring[:mu]]
                
                m = np.zeros(n)
                for i in range(mu):
                    m += weights[i] * selected[i]
                
                if sigma > 1e-10:
                    diff = (m - mean) / sigma
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
                    
                    norm_p_sigma = np.linalg.norm(p_sigma)
                    threshold = (1.4 + 2.0 / (n + 1)) * np.sqrt((mu_eff + 1) / (n + 1)) - 1.0
                    h_sigma = 1.0 if norm_p_sigma < threshold else 0.0
                    
                    p_c = (1 - c_c) * p_c + h_sigma * c_c * diff
                    C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(weights[i] * np.outer(selected[i] - mean, selected[i] - mean) for i in range(mu))
                    
                    try:
                        np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C += 1e-6 * np.eye(n)
                        
                    sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1))
                    sigma = max(sigma, 1e-10)
                    sigma = min(sigma, 5.0)
                    
                mean = m
                mean = np.clip(mean, -5, 5)
                
                if np.linalg.norm(m - mean) < 1e-12:
                    break

        return best_x, best_f
