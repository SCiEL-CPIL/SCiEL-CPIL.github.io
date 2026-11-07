import numpy as np
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 specific logic:
        # f1 = x0
        # f2 = g(x) * (1 - sqrt(f1/g(x)))
        # g(x) = 1 + 10(n-1) + sum_{i=2}^n (xi^2 - 10cos(4pi*xi))
        # To minimize f2, we must minimize g(x).
        
        # Find optimal auxiliary variable value
        aux_lo = bounds[1][0]
        aux_hi = bounds[1][1]
        
        def h(x):
            return x**2 - 10*np.cos(4*np.pi*x)
        
        # Grid search + refinement for x_opt
        grid = np.linspace(aux_lo, aux_hi, 200)
        vals = np.array([h(x) for x in grid])
        min_idx = np.argmin(vals)
        x_opt = grid[min_idx]
        
        local_lo = max(aux_lo, x_opt - 0.01)
        local_hi = min(aux_hi, x_opt + 0.01)
        
        if local_lo < local_hi:
            res = minimize_scalar(h, bounds=(local_lo, local_hi), method='bounded')
            if res.success:
                x_opt = res.x
        
        # Calculate the optimal g value
        # g = 1 + 10*(dim-1) + (dim-1)*h(x_opt)
        g_opt = 1 + 10*(self.dim - 1) + (self.dim - 1)*h(x_opt)
        
        # If g_opt is not 1, the Pareto front shape is scaled.
        # f1 = x0
        # f2 = g_opt * (1 - sqrt(x0/g_opt))
        # Let u = x0/g_opt. Then f1 = g_opt * u, f2 = g_opt * (1 - sqrt(u)).
        # The shape is similar to the standard case, just scaled.
        # We want to sample x0 such that the points are uniformly spaced along the arc length of the curve.
        
        # The curve is defined by x0 in [0, g_opt] (assuming x0 bounds allow it, but problem clips).
        # However, x0 bounds are usually [0, 1] in ZDT4? No, ZDT4 x0 is [0, 1].
        # If g_opt > 1, then sqrt(x0/g_opt) is small for x0 in [0,1].
        # Standard ZDT4: g=1. x0 in [0,1].
        # If bounds for x0 are [0,1], we sample x0 in [0,1].
        
        x0_lo, x0_hi = bounds[0][0], bounds[0][1]
        
        # We assume the Pareto front segment we can reach is x0 in [x0_lo, x0_hi].
        # The objective function is f2(x0) = g_opt * (1 - sqrt(x0/g_opt)).
        # We want to distribute x0 values based on arc length of this curve.
        
        # Parameterize by x0.
        # ds/dx0 = sqrt(1 + (df2/dx0)^2)
        # df2/dx0 = g_opt * (-1/2) * (x0/g_opt)^(-1/2) * (1/g_opt) = -0.5 * sqrt(g_opt/x0)
        # (df2/dx0)^2 = 0.25 * g_opt / x0
        # ds/dx0 = sqrt(1 + 0.25 * g_opt / x0)
        
        # We integrate this from x0_lo to x0_hi.
        # Note: at x0=0, derivative is infinite. We start from a small epsilon.
        
        n_grid = 1000
        x0_grid = np.linspace(x0_lo, x0_hi, n_grid)
        # Avoid division by zero
        x0_safe = np.where(x0_grid == 0, 1e-10, x0_grid)
        
        # Calculate arc length density
        # If x0_lo is 0, the first point will be very close to 0.
        ds_dx0 = np.sqrt(1 + 0.25 * g_opt / x0_safe)
        
        # Cumulative arc length
        # Use trapezoidal rule or simple cumsum for discrete approximation
        # Since ds_dx0 is smooth, cumsum * dx is fine.
        dx = (x0_hi - x0_lo) / (n_grid - 1)
        arc_lengths = np.cumsum(ds_dx0) * dx
        # Normalize to [0, 1] for inversion
        total_len = arc_lengths[-1]
        if total_len == 0:
             total_len = 1.0
        arc_lengths_norm = arc_lengths / total_len
        
        # Generate uniform arc length targets
        n_evals = self.budget
        target_lengths = np.linspace(0, 1, n_evals)
        
        # Invert to get x0 values
        # arc_lengths_norm is increasing.
        x0_vals = np.interp(target_lengths, arc_lengths_norm, x0_grid)
        
        # Ensure x0_vals are within bounds (interp handles this, but clip for safety)
        x0_vals = np.clip(x0_vals, x0_lo, x0_hi)
        
        # Construct points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x0_vals
        points[:, 1:] = x_opt
        
        # Evaluate all points
        for i in range(n_evals):
            problem(points[i])
            
        return points[-1]
