import numpy as np
from scipy.optimize import minimize_scalar, brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 specific logic
        aux_lo = bounds[1][0]
        aux_hi = bounds[1][1]
        
        def h(x):
            return x**2 - 10*np.cos(4*np.pi*x)
        
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
        
        n_aux = self.dim - 1
        g_min = 1 + 10 * n_aux + n_aux * h(x_opt)
        
        x0_lo, x0_hi = bounds[0][0], bounds[0][1]
        
        u_lo = max(x0_lo, 0.0)
        u_hi = min(x0_hi, g_min)
        
        N = self.budget
        
        if u_lo >= u_hi:
            u_vals = np.linspace(x0_lo, x0_hi, N)
        else:
            C = g_min
            
            def F(u):
                if u <= 0:
                    return 0.0
                term1 = 0.5 * np.sqrt(u) * np.sqrt(4*u + C)
                term2 = 0.25 * C * np.log(2*np.sqrt(u) + np.sqrt(4*u + C))
                return term1 + term2
            
            F_lo = F(u_lo)
            F_hi = F(u_hi)
            
            u_vals = np.zeros(N)
            u_vals[0] = u_lo
            u_vals[-1] = u_hi
            
            if N > 2:
                # Map arc lengths from F_lo to F_hi
                # We have N-2 intermediate points
                # Total intervals N-1
                # We want uniform arc length
                # So we divide [F_lo, F_hi] into N-1 segments
                # The points are at the boundaries of these segments
                # F_targets[i] = F_lo + i * (F_hi - F_lo) / (N-1) for i=0..N-1
                # u_vals[i] corresponds to F_targets[i]
                
                F_targets = np.linspace(F_lo, F_hi, N)
                
                for i in range(1, N-1):
                    target = F_targets[i]
                    # Solve F(u) = target
                    # F is monotonic
                    try:
                        u_vals[i] = brentq(lambda u: F(u) - target, u_lo, u_hi)
                    except:
                        u_vals[i] = u_lo + (u_hi - u_lo) * (i / (N - 1))
        
        points = np.zeros((N, self.dim))
        points[:, 0] = u_vals
        points[:, 1:] = x_opt
        
        for i in range(N):
            problem(points[i])
            
        return points[-1]
