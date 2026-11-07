import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, n, dim):
        """Generate Halton sequence points in [0, 1]^dim."""
        points = []
        for i in range(1, n + 1):
            point = []
            for base in range(2, 2 + dim):
                f = 1.0
                r = 0.0
                i_temp = i
                while i_temp > 0:
                    f /= base
                    r += f * (i_temp % base)
                    i_temp //= base
                point.append(r)
            points.append(point)
        return np.array(points)

    def _rbf_kernel(self, X1, X2, l):
        """Compute RBF kernel matrix between X1 and X2."""
        X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
        X2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
        dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-dist_sq / (2 * l**2))

    def _fit_gp(self, X, y, l=1.0, sigma_f=1.0, jitter=1e-6):
        """Fit GP and return parameters for prediction."""
        N = len(X)
        K = self._rbf_kernel(X, X, l)
        K += jitter * np.eye(N)
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(N)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return None, None, None, None

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        return L, alpha, l, sigma_f

    def _gp_predict(self, X_new, X_train, y_train, L, alpha, l, sigma_f):
        """Predict mean and variance for X_new."""
        K_s = self._rbf_kernel(X_train, X_new, l)
        
        mu = K_s.T @ alpha
        
        v = np.linalg.solve(L, K_s)
        var = sigma_f**2 - np.sum(K_s * v, axis=0)
        var = np.maximum(var, 1e-8)
        
        return mu, var

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        X_data = []
        y_data = []

        def track_call(x):
            nonlocal best_x, best_f, calls_used
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            calls_used += 1
            X_data.append(x_clipped)
            y_data.append(f)
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Halton Initialization
        n_halton = 20
        halton_pts = self._halton_sequence(n_halton, d)
        for pt in halton_pts:
            if calls_used >= total_budget or time.time() > deadline:
                break
            x = -5.0 + 10.0 * pt
            track_call(x)

        if best_x is None:
            track_call(np.zeros(d))

        # Phase 2: Bayesian Optimization
        bo_budget = total_budget - 50 - calls_used
        max_gp_points = 100
        
        l = 1.0
        if len(y_data) > 1:
            sigma_f = np.std(y_data) + 1e-6
        else:
            sigma_f = 1.0

        for _ in range(bo_budget):
            if calls_used >= total_budget - 50 or time.time() > deadline:
                break
            
            # Subsample for GP
            if len(X_data) > max_gp_points:
                # Use the last max_gp_points to capture recent trends
                # Or random? Random is better for global view.
                # Let's use random subsampling for robustness
                indices = np.random.choice(len(X_data), max_gp_points, replace=False)
                X_sub = np.array(X_data)[indices]
                y_sub = np.array(y_data)[indices]
            else:
                X_sub = np.array(X_data)
                y_sub = np.array(y_data)
            
            gp_params = self._fit_gp(X_sub, y_sub, l=l, sigma_f=sigma_f)
            if gp_params is None or gp_params[0] is None:
                x_new = np.random.uniform(-5, 5, d)
                track_call(x_new)
                continue
            
            L, alpha, l_fit, sigma_f_fit = gp_params
            
            def ei_function(x):
                x = np.clip(x, -5.0, 5.0)
                x_new = x.reshape(1, -1)
                mu, var = self._gp_predict(x_new, X_sub, y_sub, L, alpha, l_fit, sigma_f_fit)
                mu = mu[0]
                sigma = np.sqrt(var[0])
                
                if sigma < 1e-8:
                    return 0.0
                    
                z = (best_f - mu) / sigma
                ei = (best_f - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return ei
            
            best_ei_val = -np.inf
            best_ei_x = None
            
            start_points = [best_x.copy()]
            for _ in range(5):
                start_points.append(np.random.uniform(-5, 5, d))
                
            for sp in start_points:
                try:
                    res = minimize(
                        lambda x: -ei_function(x),
                        sp,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6}
                    )
                    ei_val = -res.fun
                    if ei_val > best_ei_val:
                        best_ei_val = ei_val
                        best_ei_x = res.x
                except Exception:
                    pass
            
            if best_ei_x is None:
                best_ei_x = np.random.uniform(-5, 5, d)
            
            track_call(best_ei_x)

        # Phase 3: Final Polish
        remaining_budget = total_budget - calls_used
        if remaining_budget > 0 and best_x is not None:
            def polish_func(x):
                if calls_used >= total_budget or time.time() > deadline:
                    return np.inf
                calls_used += 1
                return func(np.clip(x, -5.0, 5.0))
            
            try:
                res = minimize(
                    polish_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
