import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def objective(pts_flat):
        # pts_flat: 1D array of size N*D
        pts = pts_flat.reshape(N, D)
        # Calculate pairwise squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        d_min_sq = np.min(d_sq)
        d_max_sq = np.max(d_sq)
        
        if d_min_sq < 1e-12:
            return 1e10
        ratio_sq = d_max_sq / d_min_sq
        return ratio_sq

    def hexagonal_init():
        # Generate hexagonal grid points
        # v1 = (1, 0), v2 = (0.5, sqrt(3)/2)
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3)/2])
        pts = []
        # Generate points in a 5x5 grid of hex cells centered at 0
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        # Sort by distance from origin
        dists = np.sum(pts**2, axis=1)
        sorted_idx = np.argsort(dists)
        # Take first 16
        return pts[sorted_idx[:16]]

    def square_init():
        # 4x4 grid
        side = int(np.ceil(np.sqrt(N)))
        # Use float arrays to avoid integer casting errors during centering
        x = np.arange(side, dtype=float)
        y = np.arange(side, dtype=float)
        grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        # Center it
        grid -= grid.mean(axis=0)
        return grid[:N]

    def random_init():
        # Random points in unit square, centered
        pts = np.random.rand(N, 2) - 0.5
        return pts

    best_score = np.inf
    best_pts = hexagonal_init()
    
    initial_guesses = [
        hexagonal_init(),
        square_init(),
        random_init(),
        random_init()
    ]
    
    for init_pts in initial_guesses:
        if time.time() > deadline - 10:
            break
        
        init_flat = init_pts.flatten()
        
        # Optimization
        # Using Nelder-Mead or Powell
        # Powell is generally robust for this type of problem
        res = minimize(objective, init_flat, method='Powell', 
                       options={'maxiter': 1000, 'disp': False})
        
        if res.fun < best_score:
            best_score = res.fun
            best_pts = res.x.reshape(N, D)
        
        # Also try Nelder-Mead on the best found so far? Or just separate runs.
        # Let's try a quick Nelder-Mead refinement on the best point found
        if time.time() > deadline - 5:
            break
            
        best_flat = best_pts.flatten()
        res2 = minimize(objective, best_flat, method='Nelder-Mead', 
                        options={'maxiter': 500, 'disp': False})
        if res2.fun < best_score:
            best_score = res2.fun
            best_pts = res2.x.reshape(N, D)

    return best_pts
