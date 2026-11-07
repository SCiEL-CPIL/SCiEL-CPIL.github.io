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
        time_limit = 290.0

        best_x = None
        best_f = float('inf')
        calls = 0

        # 1. Initial Exploration
        np.random.seed(42)
        init_points = [np.zeros(n)]
        for _ in range(10):
            init_points.append(np.random.uniform(-5, 5, n))
        
        for x in init_points:
            if time.time() - start_time > time_limit or calls >= budget:
                break
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        # 2. Newton's Method with Finite Difference Hessian
        x = best_x.copy()
        f = best_f
        
        h = 1e-3
        c1 = 1e-4
        rho = 0.5

        while calls < budget:
            if time.time() - start_time > time_limit:
                break
            
            # Estimate Gradient and Hessian
            g = np.zeros(n)
            H = np.zeros((n, n))
            
            f0 = f
            for i in range(n):
                if calls >= budget: break
                
                # Central difference for Gradient
                x_plus = x.copy()
                x_plus[i] += h
                fx_plus = func(x_plus)
                calls += 1
                if fx_plus < best_f:
                    best_f = fx_plus
                    best_x = x_plus.copy()
                
                if calls >= budget: break
                x_minus = x.copy()
                x_minus[i] -= h
                fx_minus = func(x_minus)
                calls += 1
                if fx_minus < best_f:
                    best_f = fx_minus
                    best_x = x_minus.copy()
                
                g[i] = (fx_plus - fx_minus) / (2 * h)
                
                # Second derivative for Hessian diagonal
                H[i, i] = (fx_plus - 2 * f0 + fx_minus) / (h ** 2)
                
                # Cross derivatives for Hessian off-diagonal
                for j in range(i + 1, n):
                    if calls >= budget: break
                    x_ij_plus_plus = x.copy()
                    x_ij_plus_plus[i] += h
                    x_ij_plus_plus[j] += h
                    fx_ij_pp = func(x_ij_plus_plus)
                    calls += 1
                    if fx_ij_pp < best_f:
                        best_f = fx_ij_pp
                        best_x = x_ij_plus_plus.copy()
                    
                    if calls >= budget: break
                    x_ij_plus_minus = x.copy()
                    x_ij_plus_minus[i] += h
                    x_ij_plus_minus[j] -= h
                    fx_ij_pm = func(x_ij_plus_minus)
                    calls += 1
                    if fx_ij_pm < best_f:
                        best_f = fx_ij_pm
                        best_x = x_ij_plus_minus.copy()
                    
                    if calls >= budget: break
                    x_ij_minus_plus = x.copy()
                    x_ij_minus_plus[i] -= h
                    x_ij_minus_plus[j] += h
                    fx_ij_mp = func(x_ij_minus_plus)
                    calls += 1
                    if fx_ij_mp < best_f:
                        best_f = fx_ij_mp
                        best_x = x_ij_minus_plus.copy()
                    
                    if calls >= budget: break
                    x_ij_minus_minus = x.copy()
                    x_ij_minus_minus[i] -= h
                    x_ij_minus_minus[j] -= h
                    fx_ij_mm = func(x_ij_minus_minus)
                    calls += 1
                    if fx_ij_mm < best_f:
                        best_f = fx_ij_mm
                        best_x = x_ij_minus_minus.copy()
                    
                    H[i, j] = (fx_ij_pp - fx_ij_pm - fx_ij_mp + fx_ij_mm) / (4 * h ** 2)
                    H[j, i] = H[i, j]

            if calls >= budget:
                break

            # Regularize Hessian if not positive definite
            eigvals, eigvecs = np.linalg.eigh(H)
            if np.min(eigvals) < 1e-6:
                H = eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T

            try:
                p = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                p = -g

            # Line Search
            alpha = 1.0
            slope = g @ p
            if slope >= 0:
                p = -p
                slope = -slope
                if slope <= 0:
                    break
            
            f_new = f
            x_new = x.copy()
            
            for _ in range(20):
                if calls >= budget: break
                x_trial = x + alpha * p
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls += 1
                
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                
                if f_trial <= f + c1 * alpha * slope:
                    f_new = f_trial
                    x_new = x_trial
                    break
                else:
                    alpha *= rho
            
            if calls >= budget:
                break

            x = x_new
            f = f_new

            if alpha < 1e-6 or np.linalg.norm(g) < 1e-8:
                break

        return best_x, best_f
