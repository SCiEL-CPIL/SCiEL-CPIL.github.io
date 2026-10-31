"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def objective(params):
        p = params.reshape(N, 2)
        # Calculate pairwise squared distances to avoid sqrt where possible, 
        # but we need actual distances for ratio.
        # diff shape (N, N, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Exclude diagonal
        # Create mask for i != j
        mask = ~np.eye(N, dtype=bool)
        dists_sq_off = dists_sq[mask]
        
        if len(dists_sq_off) == 0:
            return 1e10
            
        d_max_sq = np.max(dists_sq_off)
        if d_max_sq < 1e-12:
            return 1e10
            
        d_min_sq = np.min(dists_sq_off)
        
        # Ratio squared is (d_max / d_min)^2 = d_max_sq / d_min_sq
        # We want to minimize this.
        if d_min_sq < 1e-12:
            return 1e10
            
        return d_max_sq / d_min_sq

    # Initial guesses
    inits = []
    
    # 1. Grid (4x4)
    side = int(np.ceil(N ** 0.5))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    inits.append(np.array(grid, dtype=float))
    
    # 2. Regular Polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    poly = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(poly)
    
    # 3. Hexagonal Cluster Approximation
    # 1 center, 6 ring 1, 9 ring 2
    pts = [(0,0)]
    for i in range(6):
        a = i * np.pi / 3
        pts.append((np.cos(a), np.sin(a)))
    for i in range(9):
        a = i * 2 * np.pi / 9 + np.pi/9
        pts.append((2*np.cos(a), 2*np.sin(a)))
    inits.append(np.array(pts, dtype=float))
    
    # 4. Random inits
    np.random.seed(42)
    for _ in range(10):
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        rand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(rand_pts)
        
    best_score = float('inf')
    best_points = inits[0]
    
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Nelder-Mead
        try:
            # maxiter can be higher, but keep it reasonable to fit in time
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(N, 2)
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
