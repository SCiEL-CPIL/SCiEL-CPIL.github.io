import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3
DEADLINE = 145.0  # safety margin from 150s

def _pairwise_dists(points):
    """Compute all pairwise squared distances."""
    # points: (N, D)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=-1)
    # Zero out diagonal
    np.fill_diagonal(d2, 0)
    return d2

def _objective(x):
    """Objective: (max_dist / min_dist)^2, with points centered."""
    pts = x.reshape(N, D)
    # Center to remove translation invariance
    pts = pts - pts.mean(axis=0)
    d2 = _pairwise_dists(pts)
    # Get non-zero distances
    mask = d2 > 1e-12
    if not np.any(mask):
        return 1e10
    min_d2 = np.min(d2[mask])
    max_d2 = np.max(d2[mask])
    if min_d2 < 1e-12:
        return 1e10
    ratio2 = max_d2 / min_d2
    return ratio2

def _score_from_points(points):
    """Compute the actual score for the evaluator."""
    d2 = _pairwise_dists(points)
    mask = d2 > 1e-12
    if not np.any(mask):
        return 1e10
    min_d2 = np.min(d2[mask])
    max_d2 = np.max(d2[mask])
    if min_d2 < 1e-12:
        return 1e10
    return max_d2 / min_d2

def _make_icosahedron():
    """Return 12 icosahedron vertices."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,  phi, 0], [1,  phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1,  phi], [0, 1,  phi], [0, -1, -phi], [0, 1, -phi],
        [ phi, 0, -1], [ phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _initial_configs():
    """Generate several initial configurations."""
    configs = []
    
    # 1. Icosahedron + 2 points along z-axis
    ico = _make_icosahedron()
    # Find the 5-fold axis (z-axis in this representation)
    # Add 2 points at ±z with some radius
    for r in [0.5, 0.8, 1.2]:
        extra = np.array([[0, 0, r], [0, 0, -r]])
        cfg = np.vstack([ico, extra])
        configs.append(cfg)
    
    # 2. Random on sphere
    for seed in [42, 123, 456, 789, 1011]:
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, D)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        configs.append(pts)
    
    # 3. Random in ball
    for seed in [111, 222, 333]:
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, D) * 0.5
        configs.append(pts)
    
    # 4. Grid-based perturbed
    rng = np.random.RandomState(42)
    grid = np.array([[i % 2, (i // 2) % 2, i // 4] for i in range(14)], dtype=float)
    grid += rng.randn(14, 3) * 0.1
    configs.append(grid)
    
    return configs

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = np.inf
    
    # Try initial configurations
    configs = _initial_configs()
    
    for i, cfg in enumerate(configs):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        x0 = cfg.flatten()
        
        # Run Nelder-Mead
        try:
            res = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if res.success or res.fun < best_score:
                pts = res.x.reshape(N, D)
                score = _score_from_points(pts)
                if score < best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() - start_time > DEADLINE - 60:
            break
    
    # If we still have time, do more restarts with random perturbations
    if best_points is not None and time.time() - start_time < DEADLINE - 20:
        for restart in range(20):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            rng = np.random.RandomState(restart * 1000 + 7)
            x0 = best_points.flatten() + rng.randn(N * D) * 0.01
            try:
                res = minimize(
                    _objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_score:
                    pts = res.x.reshape(N, D)
                    score = _score_from_points(pts)
                    if score < best_score:
                        best_score = score
                        best_points = pts.copy()
            except Exception:
                pass
    
    # Fallback: return icosahedron + 2 points if nothing worked
    if best_points is None:
        ico = _make_icosahedron()
        extra = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        best_points = np.vstack([ico, extra])
    
    return best_points
