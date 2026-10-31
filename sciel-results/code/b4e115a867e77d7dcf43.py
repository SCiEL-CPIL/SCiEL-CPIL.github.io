import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    best_score = np.inf
    best_points = None
    
    # Pre-compute indices for pairs
    pairs = list(itertools.combinations(range(N), 2))
    
    def score_and_grad(x):
        """
        x: shape (N, 2)
        Returns: score (scalar), gradient (shape (N, 2))
        Score = (D_max / D_min)^2
        """
        # Flatten x for distance calculation
        # Distances squared
        # Use broadcasting for efficiency
        # x[:, np.newaxis, :] - x[np.newaxis, :, :]
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal (zero distance)
        mask = ~np.eye(N, dtype=bool)
        dists_sq[~mask] = 0 # Just in case, though we pick pairs
        
        # Get distances for valid pairs
        d_sq = dists_sq[mask]
        
        if len(d_sq) == 0:
            return 1e10, np.zeros_like(x)
            
        d_max_sq = np.max(d_sq)
        d_min_sq = np.min(d_sq)
        
        if d_min_sq < 1e-12:
            return 1e10, np.zeros_like(x)
            
        score = d_max_sq / d_min_sq
        
        # Calculate Gradient
        # We need indices of max and min
        # d_sq is a flattened array. We need to map back to (i,j)
        # It's easier to work with the full matrix
        max_idx = np.unravel_index(np.argmax(dists_sq), dists_sq.shape)
        min_idx = np.unravel_index(np.argmin(dists_sq[mask]), dists_sq.shape)
        
        # Ensure we picked a valid min (not diagonal)
        # argmin on masked array might be tricky if 0s are on diagonal.
        # Let's find min manually among valid pairs to be safe
        valid_dists = dists_sq[mask]
        min_val = np.min(valid_dists)
        # Find indices in the original matrix corresponding to this min
        # There might be multiple. Pick the first one.
        min_mask = (dists_sq == min_val) & mask
        min_indices = np.argwhere(min_mask)
        if len(min_indices) == 0:
             return score, np.zeros_like(x)
        min_i, min_j = min_indices[0]
        
        max_mask = (dists_sq == d_max_sq)
        max_indices = np.argwhere(max_mask)
        if len(max_indices) == 0:
             return score, np.zeros_like(x)
        max_i, max_j = max_indices[0]
        
        grad = np.zeros_like(x)
        
        # Gradient of d_max_sq w.r.t x
        # d(d^2)/dx_i = 2(x_i - x_j)
        # d(d^2)/dx_j = 2(x_j - x_i)
        g_max = np.zeros_like(x)
        g_max[max_i] += 2 * (x[max_i] - x[max_j])
        g_max[max_j] += 2 * (x[max_j] - x[max_i])
        
        # Gradient of d_min_sq w.r.t x
        g_min = np.zeros_like(x)
        g_min[min_i] += 2 * (x[min_i] - x[min_j])
        g_min[min_j] += 2 * (x[min_j] - x[min_i])
        
        # Score = A / B, where A = d_max_sq, B = d_min_sq
        # d(A/B) = (B dA - A dB) / B^2
        # dA = g_max, dB = g_min
        grad = (d_min_sq * g_max - d_max_sq * g_min) / (d_min_sq ** 2)
        
        return score, grad

    def optimize_from(start):
        nonlocal best_score, best_points
        if time.time() > deadline:
            return
        
        try:
            res = minimize(
                score_and_grad, 
                start, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res.success or res.fun < best_score:
                # Verify distinctness
                pts = res.x
                # Check min distance
                d_sq = np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2)
                mask = ~np.eye(N, dtype=bool)
                if np.min(d_sq[mask]) > 1e-8:
                    if res.fun < best_score:
                        best_score = res.fun
                        best_points = pts.copy()
        except Exception:
            pass

    # Initial configurations
    inits = []
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    poly = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(poly)
    
    # 2. Grid (4x4)
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([i, j])
    inits.append(np.array(grid_pts, dtype=float))
    
    # 3. Random
    np.random.seed(42)
    inits.append(np.random.rand(N, 2) * 10 - 5)
    
    # 4. Two Clusters
    c1 = np.random.rand(N//2, 2) * 2 - 1
    c2 = np.random.rand(N//2, 2) * 2 - 1 + [5, 5]
    inits.append(np.vstack([c1, c2]))
    
    # 5. Circle with perturbation
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    poly2 = np.column_stack([np.cos(angles), np.sin(angles)]) + np.random.normal(0, 0.1, (N, 2))
    inits.append(poly2)
    
    # 6. Square lattice with perturbation
    sq = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    sq += np.random.normal(0, 0.1, (N, 2))
    inits.append(sq)
    
    # Run optimization
    for init in inits:
        if time.time() > deadline:
            break
        optimize_from(init)
        
        # If we have time, try more random starts
        while time.time() < deadline - 10:
            rand_init = np.random.rand(N, 2) * 10 - 5
            optimize_from(rand_init)
            
    if best_points is None:
        # Fallback to grid
        return np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
        
    return best_points
