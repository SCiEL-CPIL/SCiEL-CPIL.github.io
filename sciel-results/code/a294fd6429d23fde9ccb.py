"""Optimize 14 points in 3D to minimize (max_pairwise_dist / min_pairwise_dist)^2."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 14
D = 3
DEADLINE = 120.0  # evaluator kills at 150s; keep safety margin

def _score(pts):
    """Compute the raw score: (max pairwise dist / min pairwise dist)^2."""
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-12:
        return np.inf
    return (mx / mn) ** 2

def _objective(x):
    """
    Objective for optimization. We use a log-ratio formulation to make
    the landscape smoother. We also add a small regularization to keep
    points from collapsing.
    """
    pts = x.reshape(N, D)
    # Center to reduce translation freedom (helps convergence)
    pts = pts - pts.mean(axis=0, keepdims=True)
    
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    
    if mn < 1e-6:
        return 1e10
    
    # Use log ratio to smooth the objective
    val = np.log(mx) - np.log(mn)
    
    # Small penalty for very small min distance to avoid degeneracy
    if mn < 0.1:
        val += (0.1 - mn) ** 2 * 100
    
    return val

def _objective_grad(x):
    """Gradient of the log-ratio objective (approximate via finite differences or analytic)."""
    # For simplicity and robustness, use numerical gradient
    eps = 1e-7
    g = np.zeros_like(x)
    f0 = _objective(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        g[i] = (_objective(xp) - f0) / eps
    return g

def _random_init(scale=1.0):
    """Generate a random initial configuration."""
    return np.random.randn(N, D) * scale

def _icosahedron_init():
    """
    Generate points based on icosahedron vertices + some perturbation.
    Icosahedron has 12 vertices; we need 14, so add 2 more near center or on faces.
    This gives a good starting geometry for spherical codes.
    """
    t = (1 + np.sqrt(5)) / 2  # golden ratio
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Add 2 more points: place them at the poles of a different axis or on faces
    # For simplicity, add two points near the center but slightly offset
    extra = np.array([[0, 0, 0], [0.1, 0.1, 0.1]], dtype=float)
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True) * 0.9
    
    pts = np.vstack([verts, extra])
    return pts

def _optimize_from_start(start_pts, max_iter=500):
    """Run L-BFGS-B optimization from a given start."""
    x0 = start_pts.flatten()
    res = minimize(
        _objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return res.x.reshape(N, D), res.fun

def construct():
    global _t0
    _t0 = time.time()
    
    best_pts = None
    best_score = np.inf
    
    # Strategy: multiple random restarts + icosahedron-based starts
    # with annealing-like approach: start with smoother objective, then refine
    
    np.random.seed(42)
    
    starts = []
    
    # 1. Icosahedron-based start
    starts.append(_icosahedron_init())
    
    # 2. Several random starts with different scales
    for scale in [0.5, 1.0, 2.0]:
        starts.append(_random_init(scale))
    
    # 3. Fibonacci sphere (good distribution on sphere)
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + np.sqrt(5)) * i
    fib_pts = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    starts.append(fib_pts)
    
    # 4. Random perturbation of icosahedron
    ico = _icosahedron_init()
    starts.append(ico + np.random.randn(N, D) * 0.1)
    
    for start in starts:
        if time.time() - _t0 > DEADLINE - 10:
            break
        
        try:
            opt_pts, obj_val = _optimize_from_start(start, max_iter=300)
            sc = _score(opt_pts)
            if sc < best_score:
                best_score = sc
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # If we have time, do a second round with perturbed best
    if best_pts is not None and time.time() - _t0 < DEADLINE - 15:
        for _ in range(3):
            if time.time() - _t0 > DEADLINE - 10:
                break
            perturbed = best_pts + np.random.randn(N, D) * 0.05
            try:
                opt_pts, obj_val = _optimize_from_start(perturbed, max_iter=200)
                sc = _score(opt_pts)
                if sc < best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
            except Exception:
                continue
    
    if best_pts is None:
        # Fallback: return icosahedron-based points
        best_pts = _icosahedron_init()
    
    # Ensure distinct points
    d = cdist(best_pts, best_pts)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-9:
        # Add tiny perturbation to any coincident points
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-9:
                    best_pts[j] += np.random.randn(D) * 1e-6
    
    return best_pts.astype(float)
