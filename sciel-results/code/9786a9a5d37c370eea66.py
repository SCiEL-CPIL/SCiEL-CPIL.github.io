import numpy as np
from scipy.optimize import minimize
import time

N = 50

# Precompute triple indices
_idx = np.arange(N)
_ii, _jj, _kk = np.meshgrid(_idx, _idx, _idx, indexing='ij')
_mask = (_ii < _jj) & (_jj < _kk)
_II, _JJ, _KK = _ii[_mask], _jj[_mask], _kk[_mask]

def _fibonacci_spiral(phase_offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + phase_offset
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_smooth(points):
    a = points[_II]
    b = points[_JJ]
    c = points[_KK]
    
    n_dir = np.cross(b - a, c - a)
    n_norm = np.linalg.norm(n_dir, axis=1)
    valid = n_norm > 1e-10
    if not np.any(valid):
        return np.pi
    
    n_dir = n_dir[valid] / n_norm[valid, None]
    a_v = a[valid]
    
    dots_a = np.sum(n_dir * a_v, axis=1)
    signs = np.where(dots_a >= 0, 1.0, -1.0)
    n_dir = n_dir * signs[:, None]
    
    dots = np.sum(n_dir * a_v, axis=1)
    dists = np.arccos(np.clip(dots, -1.0, 1.0))
    
    m = np.max(dists)
    log_sum_exp = m + np.log(np.sum(np.exp(dists - m)))
    covering = log_sum_exp - np.log(len(dists))
    
    return covering

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    best_points = None
    best_radius = np.pi
    
    phase_offsets = [0.0, 1.0, 2.0]
    
    for phase in phase_offsets:
        if time.time() > deadline - 30.0:
            break
        
        x0 = _fibonacci_spiral(phase)
        
        def objective(x):
            pts = x.reshape(-1, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts = pts / norms
            return _covering_radius_smooth(pts)
        
        try:
            result = minimize(objective, x0.flatten(), method='L-BFGS-B',
                           options={'maxiter': 80, 'ftol': 1e-13, 'gtol': 1e-9, 'maxfun': 8000})
        except Exception:
            continue
        
        pts = result.x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-10)
        
        r = _covering_radius_smooth(pts)
        if r < best_radius:
            best_radius = r
            best_points = pts
        
        # Polish
        if time.time() < deadline - 15.0:
            try:
                result2 = minimize(objective, pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 40, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 4000})
                pts2 = result2.x.reshape(-1, 3)
                norms2 = np.linalg.norm(pts2, axis=1, keepdims=True)
                pts2 = pts2 / np.maximum(norms2, 1e-10)
                r2 = _covering_radius_smooth(pts2)
                if r2 < best_radius:
                    best_radius = r2
                    best_points = pts2
            except Exception:
                pass
    
    if best_points is None:
        best_points = _fibonacci_spiral(0.0)
    
    return best_points
