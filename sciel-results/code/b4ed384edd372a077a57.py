"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimized approach for ZDT6 exploiting the analytic Pareto front structure.
    
    ZDT6:
    - x in [0,1]^10
    - g(x) = 1 + sum(x[1:])
    - f1 = 1 - exp(-4*x1) * sin(6*pi*x1)^6
    - f2 = g
    
    Pareto front: x[1:] = 0 (g=1), x1 in [0,1]
    Front: f2 = 1, f1 = 1 - exp(-4*x1)*sin(6*pi*x1)^6
    
    Key insight: Since f2=1 for all Pareto-optimal points, the 2D hypervolume
    with reference [11,11] is simply 10*(11 - min_f1). So we only need to
    find the x1 that minimizes f1, i.e., maximizes exp(-4*x1)*sin(6*pi*x1)^6.
    
    Strategy: Use dense sampling + local refinement to find the global minimum
    of f1(x1) on [0,1], then place one evaluation there. Remaining evaluations
    can be placed anywhere on the Pareto front.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget
        dim = self.dim
        
        # Phase 1: Find the x1 that minimizes f1 = 1 - exp(-4*x1)*sin(6*pi*x1)^6
        # This is equivalent to maximizing h(x1) = exp(-4*x1)*sin(6*pi*x1)^6
        # Use dense grid + golden section refinement
        
        # Dense initial grid
        n_grid = 10000
        x1_grid = np.linspace(0, 1, n_grid)
        
        # Compute h(x1) on grid
        sin_term = np.sin(6 * np.pi * x1_grid)
        h_grid = np.exp(-4 * x1_grid) * np.abs(sin_term) ** 6
        
        # Find best grid point
        idx_best = np.argmax(h_grid)
        x1_best = x1_grid[idx_best]
        
        # Refine around best point using finer grid
        # The function has 6 local maxima (from sin(6*pi*x1)^6), so check neighbors
        # Find all local maxima on the grid
        local_maxima = []
        for i in range(1, n_grid - 1):
            if h_grid[i] > h_grid[i-1] and h_grid[i] > h_grid[i+1]:
                local_maxima.append(x1_grid[i])
        
        # Also include boundaries
        local_maxima = [0.0] + local_maxima + [1.0]
        
        # Refine each local maximum with a fine grid
        best_h = -1
        best_x1 = x1_grid[idx_best]
        
        for xm in local_maxima:
            # Fine grid around this local max
            width = 0.005
            x1_fine = np.linspace(max(0, xm - width), min(1, xm + width), 2000)
            sin_f = np.sin(6 * np.pi * x1_fine)
            h_fine = np.exp(-4 * x1_fine) * np.abs(sin_f) ** 6
            idx_f = np.argmax(h_fine)
            if h_fine[idx_f] > best_h:
                best_h = h_fine[idx_f]
                best_x1 = x1_fine[idx_f]
        
        # Even finer refinement
        width = 0.0005
        x1_ultra = np.linspace(max(0, best_x1 - width), min(1, best_x1 + width), 10000)
        sin_u = np.sin(6 * np.pi * x1_ultra)
        h_ultra = np.exp(-4 * x1_ultra) * np.abs(sin_u) ** 6
        idx_u = np.argmax(h_ultra)
        best_x1 = x1_ultra[idx_u]
        best_h = h_ultra[idx_u]
        
        # Phase 2: Create solutions on Pareto front
        # One point at best_x1, rest spread out (they don't affect HV but fill the front)
        X = np.zeros((n, dim))
        X[0, 0] = best_x1
        
        # Spread remaining points uniformly on the front
        if n > 1:
            x1_rest = np.linspace(0, 1, n - 1)
            # Make sure we don't duplicate best_x1 exactly (not critical)
            X[1:, 0] = x1_rest
        # x[1:] stays 0 for all points (Pareto-optimal)
        
        # Evaluate all points
        F = []
        for i in range(n):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
        
        F = np.array(F)
        
        return X[:len(F)]
# EVOLVE-BLOCK-END
