import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 560
    n = 14

    def objective(x):
        pts = x.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1)
        # Handle zero norm
        mask = norms < 1e-12
        if np.any(mask):
            norms[mask] = 1.0
            pts[mask] = 0.0
            pts[mask, 0] = 1.0
        pts = pts / norms[:, np.newaxis]
        
        d = pdist(pts)
        if len(d) == 0:
            return 0.0
        d_min = float(d.min())
        d_max = float(d.max())
        if d_max < 1e-12:
            return 0.0
        ratio = d_min / d_max
        return -(ratio ** 2)

    # Initial candidates
    candidates = []

    # 1. Icosahedron + Poles
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, 1, t], [1, 1, t], [-1, -1, t], [1, -1, t],
        [1, t, -1], [1, t, 1], [-1, t, -1], [-1, t, 1],
        [t, -1, -1], [t, 1, -1], [t, -1, 1], [t, 1, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    candidates.append(np.vstack([ico, poles]).flatten())

    # 2. Fibonacci Sphere
    k = np.arange(n)
    phi = np.arccos(1 - 2 * (k + 0.5) / n)
    golden = np.pi * (1 + np.sqrt(5))
    fib = np.column_stack([
        np.sin(phi) * np.cos(golden * k),
        np.sin(phi) * np.sin(golden * k),
        np.cos(phi)
    ])
    candidates.append(fib.flatten())

    # 3. Random Sphere Points
    np.random.seed(12345)
    for _ in range(30):
        pts = np.random.randn(n, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())

    best_val = -np.inf
    best_pts = None

    # Optimize each candidate
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 60:
            break
        
        try:
            # Powell is robust for this
            res = minimize(objective, cand, method='Powell',
                           options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11})
            val = float(res.fun)
            if val > best_val: # Note: objective is negative, so higher (less negative) is better
                # Wait, objective returns -ratio^2.
                # We want to minimize -ratio^2, which is maximizing ratio^2.
                # So we want the MINIMUM value of res.fun (most negative).
                if val < best_val:
                    best_val = val
                    best_pts = res.x
        except Exception:
            pass

    # Polish the best result
    if best_pts is not None and time.time() < deadline - 30:
        try:
            res2 = minimize(objective, best_pts, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if float(res2.fun) < best_val:
                best_val = float(res2.fun)
                best_pts = res2.x
        except Exception:
            pass

    # Final cleanup and normalization
    if best_pts is None:
        best_pts = candidates[0]
        
    pts = best_pts.reshape(n, 3)
    norms = np.linalg.norm(pts, axis=1)
    mask = norms < 1e-12
    if np.any(mask):
        norms[mask] = 1.0
        pts[mask] = 0.0
        pts[mask, 0] = 1.0
    pts = pts / norms[:, np.newaxis]
    
    return pts
