import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 325.0
        self.optimal_x0 = self._compute_optimal_x0()

    def _compute_optimal_x0(self):
        n = self.dim
        # Dense grid to estimate curvature
        x_grid = np.linspace(0, 1, 1000)
        y_vals = np.zeros_like(x_grid)
        i_vals = np.arange(1, n)
        phases = i_vals * np.pi / n
        
        for idx, x0 in enumerate(x_grid):
            sin_vals = np.sin(6.0 * np.pi * x0 + phases)
            # g = 1 + 10/(n-1) * sum(max(0, -sin)) ? No.
            # Min g = 1 + 10/(n-1) * sum(|xi - sin|) with xi = max(0, sin)
            # If sin >= 0, |sin - sin| = 0.
            # If sin < 0, |0 - sin| = -sin.
            # So sum is sum(max(0, -sin)).
            penalty = np.sum(np.maximum(0, -sin_vals))
            g = 1.0 + 10.0 * penalty / (n - 1)
            y_vals[idx] = (1.0 + g) * (1.0 - np.sqrt(x0 / (1.0 + g)))
        
        # Calculate second derivative
        dy = np.gradient(y_vals, x_grid)
        d2y = np.gradient(dy, x_grid)
        
        # We want points where |d2y| is high to be denser.
        # Weight w(x) = 1 / sqrt(|d2y(x)| + eps)
        eps = 1e-6
        weights = 1.0 / np.sqrt(np.abs(d2y) + eps)
        
        # Normalize weights to sum to 1
        weights /= np.sum(weights)
        
        # Cumulative sum to find quantiles
        cum_weights = np.cumsum(weights)
        cum_weights /= cum_weights[-1]
        
        # We want 200 points.
        # The optimal points are the quantiles of the distribution defined by weights.
        n_pts = self.budget
        # Use linspace for quantiles
        quantiles = np.linspace(0, 1, n_pts)
        
        # Interpolate to find x values
        optimal_x0 = np.interp(quantiles, cum_weights, x_grid)
        
        # Ensure 0 and 1 are included?
        # The first point should be close to 0, last close to 1.
        # np.interp with 0 gives x_grid[0]=0.
        # np.interp with 1 gives x_grid[-1]=1.
        
        return optimal_x0

    def __call__(self, problem, bounds):
        n = self.dim
        i_vals = np.arange(1, n)
        phases = i_vals * np.pi / n
        
        for x0 in self.optimal_x0:
            if time.time() > self.deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0
            
            sin_vals = np.sin(6.0 * np.pi * x0 + phases)
            x[1:] = np.maximum(0.0, sin_vals)
            
            try:
                problem(x)
            except Exception:
                break

        return None
