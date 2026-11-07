import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280
        rng = np.random.default_rng(0)
        
        # Phase 1: (1+1)-ES to get close to the optimum
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1
        sigma = 2.0
        
        # Run ES for 100 evaluations or until time limit
        es_budget = min(100, budget)
        for _ in range(es_budget):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
        
        # Phase 2: Hessian Newton Step
        if calls < budget and time.time() < deadline:
            # Estimate Gradient and Hessian
            h = 1e-3
            x_curr = x
            fx_curr = fx
            
            # Gradient
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x_curr.copy()
                x_minus = x_curr.copy()
                x_plus[i] += h
                x_minus[i] -= h
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls += 2
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            # Hessian
            H = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(i, dim):
                    x_pp = x_curr.copy()
                    x_pm = x_curr.copy()
                    x_mp = x_curr.copy()
                    x_mm = x_curr.copy()
                    
                    x_pp[i] += h; x_pp[j] += h
                    x_pm[i] += h; x_pm[j] -= h
                    x_mp[i] -= h; x_mp[j] += h
                    x_mm[i] -= h; x_mm[j] -= h
                    
                    x_pp = np.clip(x_pp, -5, 5)
                    x_pm = np.clip(x_pm, -5, 5)
                    x_mp = np.clip(x_mp, -5, 5)
                    x_mm = np.clip(x_mm, -5, 5)
                    
                    f_pp = func(x_pp)
                    f_pm = func(x_pm)
                    f_mp = func(x_mp)
                    f_mm = func(x_mm)
                    calls += 4
                    
                    if i == j:
                        # Reuse gradient points if possible? No, we used central diff for grad.
                        # H_ii = (f(x+he_i) - 2f(x) + f(x-he_i)) / h^2
                        # We have f(x+he_i) and f(x-he_i) from grad loop?
                        # In the grad loop, we computed f(x+he_i) and f(x-he_i).
                        # But we didn't store them. We can just compute them again or store them.
                        # Let's just compute them again for simplicity, or modify the loop.
                        # To save budget, let's store them in the grad loop.
                        pass # Placeholder, will refactor below
            
            # Refactored Hessian Estimation to reuse points
            H = np.zeros((dim, dim))
            f_center = fx_curr
            # We need f(x +/- he_i) for diagonal terms
            # We already computed them in the grad loop but didn't store.
            # Let's re-compute or just accept the cost. 25 * 4 = 100 evals for Hessian.
            # Total budget 500. 100 (ES) + 10 (Grad) + 100 (Hess) = 210.
            # Remaining 290 for polish. This is fine.
            
            for i in range(dim):
                for j in range(i, dim):
                    x_pp = x_curr.copy()
                    x_pm = x_curr.copy()
                    x_mp = x_curr.copy()
                    x_mm = x_curr.copy()
                    
                    x_pp[i] += h; x_pp[j] += h
                    x_pm[i] += h; x_pm[j] -= h
                    x_mp[i] -= h; x_mp[j] += h
                    x_mm[i] -= h; x_mm[j] -= h
                    
                    x_pp = np.clip(x_pp, -5, 5)
                    x_pm = np.clip(x_pm, -5, 5)
                    x_mp = np.clip(x_mp, -5, 5)
                    x_mm = np.clip(x_mm, -5, 5)
                    
                    f_pp = func(x_pp)
                    f_pm = func(x_pm)
                    f_mp = func(x_mp)
                    f_mm = func(x_mm)
                    calls += 4
                    
                    if i == j:
                        H[i, i] = (f_pp - 2 * f_center + f_mm) / (h * h)
                    else:
                        H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
                        H[j, i] = H[i, j]
            
            # Newton Step
            # Regularize H
            H_reg = H + 1e-6 * np.eye(dim)
            try:
                delta = np.linalg.solve(H_reg, -grad)
            except np.linalg.LinAlgError:
                delta = -grad * 0.1 # Fallback to gradient step
            
            x_new = np.clip(x_curr + delta, -5, 5)
            f_new = func(x_new)
            calls += 1
            
            if f_new < fx_curr:
                x, fx = x_new, f_new
        
        # Phase 3: Polish with (1+1)-ES
        sigma = 0.1
        while calls < budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.8
            
            if sigma < 1e-6:
                break

        return x, fx
