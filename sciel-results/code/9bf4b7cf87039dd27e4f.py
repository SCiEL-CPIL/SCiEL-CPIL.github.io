import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def objective(pts):
        # pts: (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        d_min_sq = np.min(d_sq)
        d_max_sq = np.max(d_sq)
        
        if d_min_sq < 1e-12:
            return 1e10
        return d_max_sq / d_min_sq

    def hexagonal_init():
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3)/2])
        pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        dists = np.sum(pts**2, axis=1)
        sorted_idx = np.argsort(dists)
        return pts[sorted_idx[:N]]

    def square_init():
        side = int(np.ceil(np.sqrt(N)))
        grid = np.array(np.meshgrid(range(side), range(side))).T.reshape(-1, 2)
        grid -= grid.mean(axis=0)
        return grid[:N]

    def random_init():
        pts = np.random.rand(N, 2) - 0.5
        return pts

    best_score = np.inf
    best_pts = hexagonal_init()
    
    # Evaluate initial hexagonal
    best_score = objective(best_pts)
    
    initial_guesses = [
        hexagonal_init(),
        square_init(),
        random_init(),
        random_init()
    ]
    
    for init_pts in initial_guesses:
        if time.time() > deadline - 10:
            break
            
        # Optimization
        res = minimize(objective, init_pts, method='Powell', 
                       options={'maxiter': 1000, 'disp': False})
        
        if res.fun < best_score:
            best_score = res.fun
            best_pts = res.x
            
        if time.time() > deadline - 5:
            break
            
        # Refine best found
        res2 = minimize(objective, best_pts, method='Nelder-Mead', 
                        options={'maxiter': 500, 'disp': False})
        if res2.fun < best_score:
            best_score = res2.fun
            best_pts = res2.x

    return best_pts
