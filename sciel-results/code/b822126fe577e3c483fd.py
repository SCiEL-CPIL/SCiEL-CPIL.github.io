import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import cKDTree

N = 50

def _compute_radius(p):
    """Compute the valid radius for a configuration."""
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(p)):
        return 0.0
    # Boundary constraint
    b = 0.5 - np.max(np.abs(p), axis=1)
    min_b = np.min(b)
    # Pairwise constraint
    diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists) / 2.0
    return float(min(min_b, min_d))

def _rot_matrix(rx, ry, rz):
    """Build rotation matrix from Euler angles (XYZ convention)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

def _fcc_seed():
    """Generate a compact FCC-like seed cluster of N points."""
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                pts.append([float(i), float(j), float(k)])
                pts.append([i+0.5, j+0.5, float(k)])
                pts.append([i+0.5, float(j), k+0.5])
                pts.append([float(i), j+0.5, k+0.5])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1)
    idx = np.argsort(d)
    return pts[idx[:N]]

_SEED = _fcc_seed()
# Precompute unit vectors from centroid for breathing
_SEED_NORM = np.linalg.norm(_SEED, axis=1, keepdims=True)
_SEED_NORM = np.where(_SEED_NORM < 1e-12, 1.0, _SEED_NORM)
_SEED_DIRS = _SEED / _SEED_NORM

def _decode(params):
    """Decode parameter vector to point cloud.
    params: [rx, ry, rz, tx, ty, tz, scale, breathe_0..breathe_{N-1}]
    """
    rx, ry, rz = params[0], params[1], params[2]
    t = params[3:6]
    scale = params[6]
    breathe = params[7:7+N]
    
    R = _rot_matrix(rx, ry, rz)
    # Apply breathing: move each point along its radial direction
    p = _SEED + _SEED_DIRS * breathe[:, np.newaxis]
    # Rotate and scale
    p = (p @ R.T) * scale + t
    return p

def _neg_radius(params):
    p = _decode(params)
    return -_compute_radius(p)

def _local_search_from(p0, time_budget=10.0):
    """Run Nelder-Mead local search from p0 with time budget."""
    start = time.time()
    best_p = p0.copy()
    best_r = _compute_radius(p0)
    
    # Convert to parameter space
    # Estimate initial params: assume identity rotation, fit scale and translation
    # Simple approach: use centroid and RMS
    centroid = p0.mean(axis=0)
    # Estimate scale from RMS distance
    rms = np.sqrt(np.mean(np.sum((p0 - centroid)**2, axis=1)))
    seed_rms = np.sqrt(np.mean(np.sum(_SEED**2, axis=1)))
    scale0 = rms / seed_rms if seed_rms > 0 else 1.0
    
    # Initial params
    p_params = np.zeros(6 + 1 + N)
    p_params[3:6] = centroid
    p_params[6] = scale0
    # Breathing: try to fit
    # For now start at zero breathing
    
    for attempt in range(3):
        if time.time() - start > time_budget:
            break
        res = minimize(_neg_radius, p_params, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8,
                                'adaptive': True})
        p_new = _decode(res.x)
        r_new = _compute_radius(p_new)
        if r_new > best_r:
            best_r = r_new
            best_p = p_new
            p_params = res.x.copy()
        else:
            # Perturb and retry
            p_params = p_params + np.random.normal(0, 0.05, size=p_params.shape)
    
    return best_p, best_r

def _metropolis_mc(p0, time_budget=30.0, temp=0.02, n_iter=5000):
    """Metropolis Monte Carlo to escape local minima."""
    start = time.time()
    p = p0.copy()
    best_p = p.copy()
    best_r = _compute_radius(p)
    
    for it in range(n_iter):
        if time.time() - start > time_budget:
            break
        # Random perturbation
        p_new = p + np.random.normal(0, temp, size=(N, 3))
        # Keep in cube
        p_new = np.clip(p_new, -0.49, 0.49)
        r_new = _compute_radius(p_new)
        if r_new > best_r:
            best_r = r_new
            best_p = p_new.copy()
        # Accept if better or with Metropolis probability
        if r_new >= _compute_radius(p) or np.random.rand() < np.exp((r_new - _compute_radius(p)) / 0.01):
            p = p_new
            # Cool down slowly
            temp *= 0.9999
    
    return best_p, best_r

def construct():
    start_time = time.time()
    time_limit = 140.0  # 10s margin before 150s kill
    
    # Phase 1: Global search in parameter space using differential evolution
    # params: [rx, ry, rz, tx, ty, tz, scale, breathe_0..breathe_{N-1}]
    # Bounds
    bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
              (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3),
              (0.1, 0.8)]
    bounds += [(-0.1, 0.1)] * N  # breathing bounds
    
    best_p = None
    best_r = 0.0
    
    # Run differential evolution with time limit
    de_time_budget = 40.0
    if time.time() - start_time < de_time_budget:
        def de_wrapper(x):
            if time.time() - start_time > de_time_budget:
                return 1.0  # Large penalty to stop
            return _neg_radius(x)
        
        try:
            res = differential_evolution(de_wrapper, bounds, maxiter=200, 
                                         tol=1e-6, popsize=15, seed=42,
                                         polish=True, init='sobol')
            p_de = _decode(res.x)
            r_de = _compute_radius(p_de)
            if r_de > best_r:
                best_r = r_de
                best_p = p_de
        except Exception:
            pass
    
    # If DE didn't find anything, use seed
    if best_p is None:
        # Use scaled FCC seed
        p_seed = _SEED * 0.3
        best_p = p_seed
        best_r = _compute_radius(p_seed)
    
    # Phase 2: Local search refinement
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5.0:
        p_refined, r_refined = _local_search_from(best_p, time_budget=min(remaining * 0.4, 15.0))
        if r_refined > best_r:
            best_r = r_refined
            best_p = p_refined
    
    # Phase 3: Metropolis MC to escape local minima
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10.0:
        p_mc, r_mc = _metropolis_mc(best_p, time_budget=min(remaining * 0.5, 30.0))
        if r_mc > best_r:
            best_r = r_mc
            best_p = p_mc
    
    # Phase 4: Final local polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3.0:
        p_final, r_final = _local_search_from(best_p, time_budget=min(remaining * 0.6, 10.0))
        if r_final > best_r:
            best_r = r_final
            best_p = p_final
    
    # Final validation
    final_r = _compute_radius(best_p)
    if final_r > best_r:
        best_r = final_r
    
    # Ensure valid output
    if not np.all(np.isfinite(best_p)) or best_p.shape != (N, 3):
        # Fallback to simple grid
        xs = np.linspace(-0.4, 0.4, 5)
        ys = np.linspace(-0.4, 0.4, 5)
        zs = np.linspace(-0.4, 0.4, 2)
        grid = np.array([[x, y, z] for x in xs for y in ys for z in zs])
        best_p = grid[:N]
    
    return best_p
