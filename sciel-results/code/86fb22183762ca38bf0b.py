import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    start = time.time()
    DEADLINE = start + 605

    n = 14
    d = 3

    def score(pts):
        d = pdist(pts)
        if d.max() <= 0:
            return 0.0
        return (d.min() / d.max()) ** 2

    def coulomb_obj(x):
        pts = x.reshape((n, d))
        d = pdist(pts)
        # Avoid division by zero
        d = np.where(d < 1e-10, 1e-10, d)
        return np.sum(1.0 / d)

    def neg_metric(x):
        pts = x.reshape((n, d))
        # Center
        pts = pts - np.mean(pts, axis=0)
        s = score(pts)
        return -s

    def make_nm(x0):
        r = minimize(neg_metric, x0, method='Nelder-Mead',
                     options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        return r.x

    def make_coulomb(x0):
        r = minimize(coulomb_obj, x0, method='L-BFGS-B',
                     options={'maxiter': 1000, 'ftol': 1e-10})
        return r.x

    best_pts = None
    best_score = -1.0
    rng = np.random.RandomState(42)

    # Strategy 1: Multi-start Coulomb + NM
    # We will run several Coulomb optimizations from random starts, then polish with NM.
    num_starts = 50
    
    for i in range(num_starts):
        if time.time() > DEADLINE - 60:
            break
            
        # Random start
        x0 = rng.randn(n * d)
        
        # Coulomb polish to spread points
        try:
            x_coulomb = make_coulomb(x0)
        except Exception:
            x_coulomb = x0
        
        # NM polish on the specific metric
        try:
            x_nm = make_nm(x_coulomb)
        except Exception:
            x_nm = x_coulomb
        
        pts = x_nm.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        s = score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Strategy 2: Specific structural seeds
    # Seed 1: Cube (8) + Octahedron (6)
    # Cube: (±1, ±1, ±1)
    # Octahedron: (±2, 0, 0), (0, ±2, 0), (0, 0, ±2)
    # We can optimize the relative scale and offset if we allow general movement, 
    # but let's just feed this as a start for NM.
    c = 1.0
    o = 2.0
    cube = np.array([[c, c, c], [c, c, -c], [c, -c, c], [c, -c, -c],
                     [-c, c, c], [-c, c, -c], [-c, -c, c], [-c, -c, -c]])
    octa = np.array([[o, 0, 0], [-o, 0, 0], [0, o, 0], [0, -o, 0], [0, 0, o], [0, 0, -o]])
    x_seed = np.vstack([cube, octa]).flatten()
    
    if time.time() < DEADLINE - 30:
        try:
            x_nm = make_nm(x_seed)
            pts = x_nm.reshape((n, d))
            pts = pts - np.mean(pts, axis=0)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass

    # Seed 2: Two rings of 7? (D7 symmetry)
    # Or Icosahedron (12) + 2 poles? (Current program did this, but let's try a distorted version)
    # Let's try a "distorted icosahedron" start.
    phi = (1 + np.sqrt(5)) / 2
    I12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    I12 = I12 / np.linalg.norm(I12, axis=1, keepdims=True)
    poles = np.array([[0, 0, 1.5], [0, 0, -1.5]])
    x_seed2 = np.vstack([I12, poles]).flatten()
    
    if time.time() < DEADLINE - 30:
        try:
            x_nm = make_nm(x_seed2)
            pts = x_nm.reshape((n, d))
            pts = pts - np.mean(pts, axis=0)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback
        best_pts = np.zeros((n, d))
        for i in range(n):
            best_pts[i] = rng.randn(d)
        best_pts = best_pts - np.mean(best_pts, axis=0)

    # Final check
    if not np.all(np.isfinite(best_pts)):
        best_pts = np.zeros((n, d))
        for i in range(n):
            best_pts[i] = rng.randn(d)
        best_pts = best_pts - np.mean(best_pts, axis=0)

    return best_pts
