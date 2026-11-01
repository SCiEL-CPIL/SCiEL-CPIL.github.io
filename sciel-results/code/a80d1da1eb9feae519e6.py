import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 14
D = 3
DEADLINE = 130.0

def _exact_score(pts):
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _smooth_obj(x, tau):
    pts = x.reshape(N, D)
    # Center and normalize
    pts = pts - np.mean(pts, axis=0)
    norm_sq = np.sum(pts**2)
    if norm_sq < 1e-6:
        return 1e6
    scale = np.sqrt(N / norm_sq)
    pts = pts * scale
    
    d2 = cdist(pts, pts) ** 2
    np.fill_diagonal(d2, 0)
    
    # Soft max
    max_d2 = np.max(d2)
    lse_max = max_d2 + tau * np.log(np.sum(np.exp((d2 - max_d2) / tau)))
    
    # Soft min
    min_d2 = np.min(d2)
    lse_min = min_d2 - tau * np.log(np.sum(np.exp((min_d2 - d2) / tau)))
    
    if lse_min <= 0:
        return 1e6
        
    return np.log(lse_max) - np.log(lse_min)

def construct():
    t0 = time.time()
    best_score = np.inf
    best_pts = None
    
    # Pre-compute grid points for initialization
    side = 3
    grid = list(np.meshgrid(np.arange(side), np.arange(side), np.arange(side)))
    grid = np.array([c for r in zip(*grid) for c in r]).reshape(-1, 3)
    grid_pts = grid[:N].astype(float)
    grid_pts = grid_pts - np.mean(grid_pts, axis=0)
    grid_norms = np.linalg.norm(grid_pts, axis=1)
    grid_pts *= (np.sqrt(N) / grid_norms)[:, None]

    num_restarts = 100 # High number, will be cut by time
    
    for i in range(num_restarts):
        if time.time() - t0 > DEADLINE - 15:
            break
            
        # Generate initial points
        if i % 3 == 0:
            # Random on sphere
            pts = np.random.randn(N, D)
            norms = np.linalg.norm(pts, axis=1)
            pts /= norms[:, None]
            pts *= np.sqrt(N)
        elif i % 3 == 1:
            # Perturbed grid
            pts = grid_pts + np.random.randn(N, D) * 0.2
            pts = pts - np.mean(pts, axis=0)
            norms = np.linalg.norm(pts, axis=1)
            pts *= (np.sqrt(N) / norms)[:, None]
        else:
            # Another random
            pts = np.random.rand(N, D) * 2 - 1
            pts = pts - np.mean(pts, axis=0)
            norms = np.linalg.norm(pts, axis=1)
            pts *= (np.sqrt(N) / norms)[:, None]

        x0 = pts.flatten()
        
        # Anneal tau
        taus = [0.5, 0.2, 0.1, 0.05, 0.02]
        current_x = x0
        
        for tau in taus:
            if time.time() - t0 > DEADLINE - 5:
                break
                
            try:
                res = minimize(
                    _smooth_obj, 
                    current_x, 
                    args=(tau,), 
                    method='L-BFGS-B', 
                    options={'maxiter': 50, 'ftol': 1e-8}
                )
                if res.fun < 1e5: # Valid result
                    current_x = res.x
            except:
                pass
            
        # Polish with Nelder-Mead on exact objective
        final_pts = current_x.reshape(N, D)
        final_pts = final_pts - np.mean(final_pts, axis=0)
        norms = np.linalg.norm(final_pts, axis=1)
        final_pts *= (np.sqrt(N) / norms)[:, None]
        
        try:
            res2 = minimize(
                _exact_score, 
                final_pts.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_pts = res2.x.reshape(N, D)
                # Normalize best_pts
                best_pts = best_pts - np.mean(best_pts, axis=0)
                norms = np.linalg.norm(best_pts, axis=1)
                best_pts *= (np.sqrt(N) / norms)[:, None]
        except:
            # Fallback to L-BFGS result
            score = _exact_score(final_pts)
            if score < best_score:
                best_score = score
                best_pts = final_pts

    if best_pts is None:
        return grid_pts

    return best_pts
