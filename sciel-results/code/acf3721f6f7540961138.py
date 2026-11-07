import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import t as student_t

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None
        self.X_data = []
        self.y_data = []

    def __call__(self, func):
        self.deadline = time.time() + 285
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf
        self.X_data = []
        self.y_data = []

        def wrapped_func(x):
            if self.calls >= self.budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            self.X_data.append(x.copy())
            self.y_data.append(f)
            return f

        # Phase 1: Initial sampling with diverse starts (use ~150 calls)
        initial_points = []
        # Deterministic initial points for reproducibility
        initial_points.append(np.zeros(self.dim))
        for _ in range(4):
            initial_points.append(np.random.uniform(-4, 4, self.dim))
        # Add some structured points
        for axis in range(self.dim):
            for sign in [-3, 3]:
                p = np.zeros(self.dim)
                p[axis] = sign
                initial_points.append(p)
        
        for x0 in initial_points[:min(len(initial_points), 10)]:
            if self.calls >= 150 or time.time() > self.deadline - 10:
                break
            wrapped_func(x0)
        
        # Phase 2: Bayesian Optimization with GP surrogate and Student-t EI
        # Use remaining budget for BO iterations
        bo_budget = self.budget - self.calls
        if bo_budget > 10 and time.time() < self.deadline - 15:
            self._bayesian_optimize(wrapped_func, bo_budget)
        
        # Phase 3: Final polish with Nelder-Mead around best point
        if self.calls < self.budget and time.time() < self.deadline - 10:
            try:
                polish_budget = self.budget - self.calls
                maxiter = max(1, polish_budget // 3)
                res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-10,
                                        'initial_simplex': self._make_simplex(self.best_x)})
                if self.best_f > res.fun:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass

        return self.best_x, self.best_f

    def _make_simplex(self, x0):
        """Create a small simplex around x0 for the polish pass."""
        simplex = [x0.copy()]
        for i in range(self.dim):
            v = x0.copy()
            step = 0.01
            v[i] += step
            simplex.append(v)
        return np.array(simplex)

    def _gp_predict(self, X_test):
        """Gaussian Process prediction using RBF kernel."""
        X_train = np.array(self.X_data)
        y_train = np.array(self.y_data)
        
        # Normalize y for numerical stability
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) + 1e-8
        y_norm = (y_train - y_mean) / y_std
        
        # RBF kernel
        def rbf_kernel(X1, X2, length_scales, variance):
            # Compute squared distances
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            # Scale by length scales
            for i in range(self.dim):
                sq_dist /= length_scales[i] ** 2
            K = variance * np.exp(-0.5 * sq_dist)
            return K
        
        # Hyperparameters
        length_scales = np.ones(self.dim) * 0.5
        variance = 1.0
        noise = 1e-4
        
        # Train kernel matrix
        K = rbf_kernel(X_train, X_train, length_scales, variance)
        K += noise * np.eye(len(X_train))
        
        # Test kernel matrix
        K_star = rbf_kernel(X_test, X_train, length_scales, variance)
        
        # Cholesky decomposition for stability
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-6 * np.eye(len(X_train))
            L = np.linalg.cholesky(K)
        
        # Solve for alpha
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        
        # Predict mean and variance
        mean = K_star @ alpha
        var = variance - np.sum(K_star @ np.linalg.solve(L, K_star.T), axis=1)
        var = np.maximum(var, 1e-8)
        
        # Denormalize
        mean = mean * y_std + y_mean
        var = var * y_std ** 2
        
        return mean, var

    def _student_t_ei(self, mu, sigma, f_best, nu=3):
        """Student-t Expected Improvement acquisition function."""
        sigma = np.maximum(sigma, 1e-8)
        improvement = f_best - mu
        z = improvement / sigma
        # Student-t CDF and PDF
        t_cdf = student_t.cdf(z, df=nu)
        t_pdf = student_t.pdf(z, df=nu)
        ei = improvement * t_cdf + sigma * t_pdf
        return ei

    def _bayesian_optimize(self, eval_func, budget):
        """Bayesian Optimization loop with GP and Student-t EI."""
        max_iter = min(budget // 2, 100)  # Limit iterations to avoid timeout
        
        for it in range(max_iter):
            if self.calls >= self.budget or time.time() > self.deadline - 10:
                break
            
            # Generate candidate points
            n_candidates = 200
            candidates = np.random.uniform(-5, 5, (n_candidates, self.dim))
            
            # Predict mean and variance
            mu, var = self._gp_predict(candidates)
            sigma = np.sqrt(var)
            
            # Compute EI
            ei = self._student_t_ei(mu, sigma, self.best_f, nu=3)
            
            # Select best candidate
            idx = np.argmax(ei)
            x_next = candidates[idx]
            
            # Evaluate
            f_next = eval_func(x_next)
            
            # If we got a valid result, continue
            if np.isinf(f_next):
                break
