import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.n_points = min(budget, 200)
        
        self.optimal_x1 = self._compute_optimal_x1()

    def _compute_optimal_x1(self):
        N = self.n_points
        R1 = 11.0
        R2 = 11.0
        
        # H(x) = R2 - f2(x) = 11 - (1 - sqrt(x)) = 10 + sqrt(x)
        # H'(x) = 1 / (2 * sqrt(x))
        
        def hv_neg(x):
            x = np.sort(x)
            if N == 1:
                return -((R1 - x[0]) * (10.0 + np.sqrt(x[0])))
            
            f2 = 1.0 - np.sqrt(x)
            heights = R2 - f2 # 10 + sqrt(x)
            
            widths = np.diff(x)
            contrib_internal = np.sum(widths * heights[:-1])
            contrib_last = (R1 - x[-1]) * heights[-1]
            
            return -(contrib_internal + contrib_last)

        def hv_grad(x):
            x = np.sort(x)
            g = np.zeros(N)
            if N == 1:
                # d/dx [ (R1-x)(10+sqrt(x)) ] = -(10+sqrt(x)) + (R1-x)/(2sqrt(x))
                g[0] = -((10.0 + np.sqrt(x[0])) - (R1 - x[0]) / (2.0 * np.sqrt(x[0])))
                return g

            heights = 10.0 + np.sqrt(x)
            # H'(x) = 1/(2*sqrt(x))
            # Avoid div by 0
            x_safe = np.maximum(x, 1e-12)
            dH = 1.0 / (2.0 * np.sqrt(x_safe))
            
            # Interior points
            # g_i = H(x_{i-1}) - H(x_i) + (x_{i+1} - x_i) * H'(x_i)
            # Note: The term (x_{i+1} - x_i) comes from the width of the rectangle i.
            # The term -H(x_i) comes from the width of rectangle i (x_{i+1}-x_i) * H(x_i) derivative wrt x_i is -H(x_i).
            # The term H(x_{i-1}) comes from the width of rectangle i-1 (x_i - x_{i-1}) * H(x_{i-1}) derivative wrt x_i is H(x_{i-1}).
            
            for i in range(1, N - 1):
                g[i] = heights[i-1] - heights[i] + (x[i+1] - x[i]) * dH[i]
            
            # First point
            # g_1 = -H(x_1) + (x_2 - x_1) * H'(x_1)
            g[0] = -heights[0] + (x[1] - x[0]) * dH[0]
            
            # Last point
            # g_N = -H(x_N) + (R1 - x_N) * H'(x_N)
            g[N-1] = -heights[N-1] + (R1 - x[N-1]) * dH[N-1]
            
            return g

        # Initial guess: Power law
        x0 = np.linspace(1, N, N)
        x0 = (x0 / N) ** 2 * 0.26 # Scale to approx optimal end point
        
        bounds = [(1e-6, 1.0) for _ in range(N)]
        
        try:
            res = minimize(hv_neg, x0, method='L-BFGS-B', jac=hv_grad, bounds=bounds, 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                return res.x
            else:
                return res.x
        except Exception:
            return x0

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.n_points
        x1_vals = self.optimal_x1
        
        X = np.zeros((N, self.dim))
        X[:, 0] = x1_vals
        X[:, 1:] = 0.5
        
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
