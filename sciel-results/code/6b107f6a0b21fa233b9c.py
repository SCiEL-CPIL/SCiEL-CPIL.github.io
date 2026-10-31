"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
import itertools

N = 16
D = 2
DEADLINE = 150.0  # wall-clock limit; return best before this
START_TIME = time.time()


def _time_left():
    return max(0.0, DEADLINE - (time.time() - START_TIME))


def _score(pts):
    """Return (max_dist / min_dist)^2. Returns inf if any pair is too close."""
    if pts.shape[0] != N:
        return np.inf
    # Check distinctness
    d = pdist(pts)
    if d.size == 0:
        return np.inf
    dmin = d.min()
    if dmin < 1e-12:
        return np.inf
    dmax = d.max()
    return (dmax / dmin) ** 2


def _objective_flat(x):
    """Objective for flat parameter vector: minimize (max/min)^2."""
    pts = x.reshape(N, D)
    return _score(pts)


def _hexagonal_init():
    """Generate a hexagonal packing of 16 points."""
    # Hexagonal lattice: rows offset
    # For 16 points, use a 4x4 hexagonal arrangement
    pts = []
    # Use a compact hexagonal cluster
    # Center at origin, add rings
    # Ring 0: 1 point (center)
    # Ring 1: 6 points
    # Ring 2: 12 points -> total 19, too many
    # Let's use a different approach: 4x4 grid with hexagonal offsets
    
    # Actually, let's use a known good configuration:
    # 4x4 grid with alternating row offsets
    side = 2.0
    pts = []
    for i in range(4):
        for j in range(4):
            x = j * side + (0.5 * side if i % 2 == 1 else 0)
            y = i * side * np.sqrt(3) / 2
            pts.append([x, y])
    return np.array(pts, dtype=float)


def _square_init():
    """Generate a 4x4 square grid."""
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([float(j), float(i)])
    return np.array(pts, dtype=float)


def _ring_init(n_rings=4):
    """Generate points on concentric rings."""
    pts = []
    pts.append([0.0, 0.0])
    remaining = N - 1
    ring_idx = 1
    while remaining > 0 and ring_idx <= 10:
        # Number of points on this ring
        n_on_ring = min(remaining, 6 * ring_idx)
        for k in range(n_on_ring):
            angle = 2 * np.pi * k / n_on_ring
            r = ring_idx * 1.0
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        remaining -= n_on_ring
        ring_idx += 1
    return np.array(pts[:N], dtype=float)


def _random_init(seed=None):
    """Generate random points in a disk."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = rng.uniform(0, 3, N)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def _polish(pts):
    """Local polish using Nelder-Mead on the ratio objective."""
    if _time_left() < 5.0:
        return pts
    
    x0 = pts.flatten()
    try:
        res = minimize(
            _objective_flat,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        if res.fun < _score(pts):
            return res.x.reshape(N, D)
    except Exception:
        pass
    return pts


def _powell_polish(pts):
    """Local polish using Powell's method."""
    if _time_left() < 5.0:
        return pts
    
    x0 = pts.flatten()
    try:
        res = minimize(
            _objective_flat,
            x0,
            method='Powell',
            options={
                'maxiter': 3000,
                'xtol': 1e-10,
                'ftol': 1e-10
            }
        )
        if res.fun < _score(pts):
            return res.x.reshape(N, D)
    except Exception:
        pass
    return pts


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_pts = None
    best_score = np.inf
    
    # Initial configurations
    inits = [
        _hexagonal_init(),
        _square_init(),
        _ring_init(),
        _random_init(42),
        _random_init(123),
        _random_init(777),
    ]
    
    # Evaluate initial scores
    for init in inits:
        if _time_left() < 10.0:
            break
        s = _score(init)
        if s < best_score:
            best_score = s
            best_pts = init.copy()
    
    # Multi-start local optimization with Nelder-Mead
    for init in inits:
        if _time_left() < 15.0:
            break
        polished = _polish(init)
        s = _score(polished)
        if s < best_score:
            best_score = s
            best_pts = polished.copy()
    
    # Multi-start local optimization with Powell
    for init in inits:
        if _time_left() < 20.0:
            break
        polished = _powell_polish(init)
        s = _score(polished)
        if s < best_score:
            best_score = s
            best_pts = polished.copy()
    
    # Try differential evolution for global search if time permits
    if _time_left() > 30.0:
        try:
            # Bounds: points in [-5, 5] x [-5, 5]
            bounds = [(-5, 5)] * (N * D)
            
            def de_objective(x):
                if _time_left() < 5.0:
                    return np.inf
                return _objective_flat(x)
            
            de_res = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
                popsize=10,
                init='latinhypercube',
                polish=True,
                updating='deferred',
                workers=1
            )
            
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_pts = de_res.x.reshape(N, D).copy()
        except Exception:
            pass
    
    # Final polish on the best result
    if best_pts is not None and _time_left() > 10.0:
        polished = _polish(best_pts)
        s = _score(polished)
        if s < best_score:
            best_score = s
            best_pts = polished.copy()
        
        if _time_left() > 10.0:
            polished2 = _powell_polish(best_pts)
            s2 = _score(polished2)
            if s2 < best_score:
                best_score = s2
                best_pts = polished2.copy()
    
    if best_pts is None:
        # Fallback: return a simple grid
        best_pts = _square_init()
    
    return best_pts


# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
