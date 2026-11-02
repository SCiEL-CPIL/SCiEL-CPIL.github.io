"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 115.0  # seconds, safety margin before 120s kill
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _score(points):
    """Compute the packing score: min(min pairwise dist/2, min (1 - |c_i|))."""
    n = len(points)
    if n == 0:
        return 0.0
    # Check validity
    if not np.all(np.isfinite(points)):
        return 0.0
    radii = np.hypot(points[:, 0], points[:, 1])
    if np.any(radii > 1.0):
        return 0.0
    # Minimum distance to boundary
    min_boundary = 1.0 - np.max(radii)
    # Minimum pairwise distance
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(points[i, 0] - points[j, 0], points[i, 1] - points[j, 1])
            if d < min_pair:
                min_pair = d
    min_pair_half = min_pair / 2.0
    return min(min_pair_half, min_boundary)


def _hex_init():
    """Generate a hexagonal lattice arrangement clipped to unit disk."""
    # Try multiple spacings to find one that gives at least N points
    best_pts = None
    for spacing in np.linspace(0.15, 0.35, 40):
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        r = np.hypot(pts[:, 0], pts[:, 1])
        # Ensure points are inside with margin
        margin = spacing * 0.5
        mask = r <= 1.0 - margin
        pts = pts[mask]
        if len(pts) >= N:
            # Sort by radius to prefer center
            order = np.argsort(r[mask])
            pts = pts[order[:N]]
            if best_pts is None or _score(pts) > _score(best_pts):
                best_pts = pts.copy()
    if best_pts is not None:
        return best_pts
    # Fallback: random
    return _random_init()


def _ring_init():
    """Generate a ring-based arrangement."""
    # Place points on concentric rings
    # Center + rings
    pts = [np.array([0.0, 0.0])]
    # Ring 1: 6 points
    r1 = 0.25
    for k in range(6):
        angle = 2 * np.pi * k / 6
        pts.append(np.array([r1 * np.cos(angle), r1 * np.sin(angle)]))
    # Ring 2: 12 points
    r2 = 0.5
    for k in range(12):
        angle = 2 * np.pi * k / 12
        pts.append(np.array([r2 * np.cos(angle), r2 * np.sin(angle)]))
    # Ring 3: 18 points
    r3 = 0.75
    for k in range(18):
        angle = 2 * np.pi * k / 18
        pts.append(np.array([r3 * np.cos(angle), r3 * np.sin(angle)]))
    # Ring 4: 15 points
    r4 = 0.95
    for k in range(15):
        angle = 2 * np.pi * k / 15
        pts.append(np.array([r4 * np.cos(angle), r4 * np.sin(angle)]))
    
    pts = np.array(pts[:N])
    return pts


def _random_init():
    """Random initialization inside unit disk."""
    pts = []
    for _ in range(N):
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x * x + y * y <= 1.0:
                pts.append([x, y])
                break
    return np.array(pts)


def _objective(x, N_local):
    """
    Objective: maximize the minimum of (pairwise dist/2, boundary margin).
    We minimize the negative of that.
    """
    pts = x.reshape(-1, 2)
    # Boundary constraint penalty
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_min = 1.0 - np.max(radii)
    
    # Pairwise distances
    min_pair = np.inf
    for i in range(N_local):
        for j in range(i + 1, N_local):
            d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
            if d < min_pair:
                min_pair = d
    min_pair_half = min_pair / 2.0
    
    score = min(min_pair_half, boundary_min)
    return -score


def _gradient(x, N_local):
    """
    Numerical gradient of the objective.
    """
    pts = x.reshape(-1, 2)
    eps = 1e-7
    grad = np.zeros_like(x)
    base = _objective(x, N_local)
    for i in range(len(x)):
        x_pert = x.copy()
        x_pert[i] += eps
        f_pert = _objective(x_pert, N_local)
        grad[i] = (f_pert - base) / eps
    return grad


def _polish(points, max_iter=2000, tol=1e-10):
    """
    Local optimization to polish the packing.
    Uses gradient-based approach to maximize the minimum clearance.
    """
    pts = points.copy().astype(float)
    x = pts.flatten()
    
    best_score = _score(pts)
    best_x = x.copy()
    
    # Use Nelder-Mead for robustness (no gradient needed, handles non-smooth min)
    result = minimize(
        _objective,
        x,
        args=(N,),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
    )
    
    if result.fun < _objective(best_x, N):
        best_x = result.x
        best_score = -result.fun
    
    return best_x.reshape(-1, 2), best_score


def _refine_with_grad(points, max_iter=500):
    """
    Gradient-based refinement using L-BFGS-B with smooth approximation.
    """
    pts = points.copy().astype(float)
    x = pts.flatten()
    
    def smooth_obj(x):
        pts = x.reshape(-1, 2)
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_min = 1.0 - np.max(radii)
        
        # Soft-min of pairwise distances/2
        min_pair = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                if d < min_pair:
                    min_pair = d
        min_pair_half = min_pair / 2.0
        
        score = min(min_pair_half, boundary_min)
        return -score
    
    result = minimize(
        smooth_obj,
        x,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': 1e-8}
    )
    
    if result.fun < smooth_obj(x):
        return result.x.reshape(-1, 2)
    return pts


def construct():
    global _start_time
    _start_time = time.time()
    
    # Try multiple initializations and pick the best
    inits = []
    
    # Hexagonal lattice
    try:
        hex_pts = _hex_init()
        if hex_pts is not None and len(hex_pts) == N:
            inits.append(hex_pts)
    except Exception:
        pass
    
    # Ring-based
    try:
        ring_pts = _ring_init()
        if len(ring_pts) == N:
            inits.append(ring_pts)
    except Exception:
        pass
    
    # Random
    np.random.seed(42)
    for _ in range(3):
        try:
            rand_pts = _random_init()
            if len(rand_pts) == N:
                inits.append(rand_pts)
        except Exception:
            pass
    
    if not inits:
        # Last resort
        return _random_init()
    
    best_pts = None
    best_score = -np.inf
    
    for init_pts in inits:
        if _elapsed() > DEADLINE - 20:
            break
        
        # Polish with Nelder-Mead
        try:
            polished, score = _polish(init_pts, max_iter=1000, tol=1e-9)
            if score > best_score:
                best_score = score
                best_pts = polished
        except Exception:
            pass
        
        if _elapsed() > DEADLINE - 15:
            break
    
    # Final high-precision polish if time allows
    if best_pts is not None and _elapsed() < DEADLINE - 10:
        try:
            polished, score = _polish(best_pts, max_iter=3000, tol=1e-12)
            if score > best_score:
                best_score = score
                best_pts = polished
        except Exception:
            pass
    
    if best_pts is None:
        return inits[0]
    
    return best_pts
# EVOLVE-BLOCK-END
