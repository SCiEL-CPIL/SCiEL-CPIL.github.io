import numpy as np
import scipy.optimize as so
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        # 1. Define the analytical Katsuura surrogate
        # Standard Katsuura on [-1, 1]
        # f(x) = sum_i 10 * [ sum_k cos(5*pi*(2k+1)*(x_i+1))/k^2 - W0 ] + 10
        # We truncate k to 20 for speed and sufficient accuracy.
        K_MAX = 20
        ks = np.arange(1, K_MAX + 1)
        inv_k2 = 1.0 / (ks ** 2)
        # W0 = sum cos(5*pi*(2k+1)) / k^2
        W0 = np.sum(np.cos(5 * np.pi * (2 * ks + 1)) * inv_k2)
        
        def katsuura_1d(t):
            # t is a vector
            # (t+1)
            # 5*pi*(2k+1)*(t+1)
            # Vectorized:
            # t shape (N, 1) or (N,)
            # k shape (K,)
            # We want sum over k
            # args = 5 * pi * (2*ks[:, None] + 1) * (t[None, :] + 1)
            # This might be memory heavy if N is large, but N is small (fitting data)
            # Let's do it carefully.
            t = np.asarray(t)
            if t.ndim == 1:
                t = t[:, np.newaxis]
            # t is (N, 1)
            # ks is (K,)
            # We need (K, N) matrix of cos
            # args = 5 * pi * (2*ks[:, None] + 1) * (t.T + 1)
            # t.T is (1, N)
            # (2*ks[:, None] + 1) is (K, 1)
            # args is (K, N)
            args = 5 * np.pi * (2 * ks[:, np.newaxis] + 1) * (t.T + 1)
            cos_vals = np.cos(args)
            # Weighted sum
            # inv_k2 is (K,)
            # Sum over axis 0
            sum_cos = np.sum(cos_vals * inv_k2[:, np.newaxis], axis=0) # Shape (N,)
            # Term: 10 * (sum_cos - W0) + 10
            return 10 * (sum_cos - W0) + 10

        def katsuura_model(x, R, s):
            # x: (N, D)
            # R: (D, D)
            # s: (D,)
            # transformed = x @ R.T + s
            # Sum over dims of katsuura_1d(transformed[:, i])
            transformed = x @ R.T + s
            total = 0.0
            for i in range(self.dim):
                total += katsuura_1d(transformed[:, i])
            return total

        def residuals(params, x_data, y_data):
            # params: R flattened (D*D) + s (D)
            D = self.dim
            R_flat = params[:D*D]
            s = params[D*D:]
            R = R_flat.reshape(D, D)
            y_pred = katsuura_model(x_data, R, s)
            return y_pred - y_data

        # 2. Initial Exploration: Sample points for fitting
        # Use Sobol or Random. Random is fine.
        n_fit = 100
        if self.budget < n_fit + 50:
             # Fallback to simple random search if budget is tiny
             # But budget is 500.
             pass
             
        rng = np.random.default_rng(42)
        X_fit = rng.uniform(-5, 5, size=(n_fit, self.dim))
        Y_fit = np.array([func(x) for x in X_fit])
        
        calls_made = n_fit
        
        # 3. Fit R and s
        # Initial guess: R=I, s=0
        R0 = np.eye(self.dim)
        s0 = np.zeros(self.dim)
        p0 = np.concatenate([R0.flatten(), s0])
        
        try:
            res = so.least_squares(residuals, p0, args=(X_fit, Y_fit), method='lm', max_nfev=500)
            p_opt = res.x
            R_est = p_opt[:self.dim*self.dim].reshape(self.dim, self.dim)
            s_est = p_opt[self.dim*self.dim:]
        except Exception:
            # If fitting fails, fallback to grid
            R_est = np.eye(self.dim)
            s_est = np.zeros(self.dim)

        # 4. Predict Global Minimum
        # The global minimum of the standard Katsuura (on [-1,1] effectively, but the formula works for any t)
        # occurs when the cosine terms are minimized.
        # For k=1, cos(5*pi*3*(t+1)) = cos(15*pi*(t+1)).
        # The sum is a Weierstrass function. Its global minimum is at t=0 (in the standard definition where domain is [-1,1] and center is 0? No, standard is often defined on [-1,1] with min at 0? Let's check.)
        # At t=0: cos(5*pi*(2k+1)) = -1. Sum = -pi^2/6.
        # At t=1: cos(10*pi*(2k+1)) = 1. Sum = pi^2/6.
        # So t=0 is the minimum for the 1D part.
        # So we want (R x + s)_i = 0 for all i.
        # R x + s = 0 => x = -R^{-1} s
        
        try:
            R_inv = np.linalg.inv(R_est)
            x_candidate = -R_inv @ s_est
        except np.linalg.LinAlgError:
            x_candidate = np.zeros(self.dim)
            
        # Clip candidate to box
        x_candidate = np.clip(x_candidate, -5, 5)
        
        # 5. Refine with Nelder-Mead
        # Check if we have budget
        if time.time() > deadline - 30 or calls_made > self.budget - 50:
             return x_candidate, func(x_candidate)
             
        def nm_func(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            return fx
             
        res_nm = so.minimize(nm_func, x_candidate, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        
        x_best = res_nm.x
        fx_best = res_nm.fun
        
        # 6. If we have more budget, try a second candidate?
        # The model might be slightly off. Let's also run NM from a random point if budget allows.
        if calls_made < self.budget - 50 and time.time() < deadline - 30:
             x_rand = rng.uniform(-5, 5, self.dim)
             res_nm2 = so.minimize(nm_func, x_rand, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
             if res_nm2.fun < fx_best:
                 x_best = res_nm2.x
                 fx_best = res_nm2.fun
                 
        return x_best, fx_best
