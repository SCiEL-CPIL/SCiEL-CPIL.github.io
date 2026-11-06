import numpy as np
from scipy.optimize import differential_evolution
import math

def construct():
    n = 8
    
    def objective(params):
        pts = params.reshape(n, 2)
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        
        iu = np.triu_indices(n, k=1)
        d_pairs = dists[iu]
        
        d_pairs = np.maximum(d_pairs, 1e-10)
        log_sum = np.sum(np.log(d_pairs))
        
        max_d = np.max(dists)
        if max_d < 1e-10:
            return 0.0
        max_d = np.maximum(max_d, 1e-10)
        log_max_d = np.log(max_d)
        
        val = 2.0 * log_sum - n * (n - 1) * log_max_d
        return -val

    bounds = [(-1.0, 1.0)] * (2 * n)
    
    try:
        res = differential_evolution(objective, bounds, seed=42, maxiter=1000, popsize=30, tol=1e-8, polish=True)
        
        if res.success:
            pts = res.x.reshape(n, 2)
            
            diff = pts[:, np.newaxis] - pts[np.newaxis, :]
            dists = np.linalg.norm(diff, axis=2)
            max_d = np.max(dists)
            
            if max_d > 1e-6:
                pts_scaled = pts * (2.0 / max_d)
                
                diff_s = pts_scaled[:, np.newaxis] - pts_scaled[np.newaxis, :]
                dists_s = np.linalg.norm(diff_s, axis=2)
                
                d_pairs_s = dists_s[np.triu_indices(n, k=1)]
                log_delta = 2.0 * np.sum(np.log(np.maximum(d_pairs_s, 1e-10)))
                
                min_d = np.min(dists_s[~np.eye(n, dtype=bool)])
                if min_d > 1e-6:
                    return pts_scaled.tolist()
                
    except Exception:
        pass

    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts_oct = np.array([np.cos(angles), np.sin(angles)]).T
    return pts_oct.tolist()
