"""Packing 50 equal circles in a unit circle using a structural optimization approach."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(pts):
    """Compute the score: min of half-min-distance and min margin to boundary."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    norms = np.linalg.norm(pts, axis=1)
    boundary_margin = np.min(1.0 - norms)
    if boundary_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_pair_dist / 2.0, boundary_margin)


def _negative_score(pts):
    """Negative score for minimization."""
    return -_compute_score(pts)


def _hex_lattice_init():
    """Generate a hexagonal lattice initial configuration."""
    # Start with a reasonable spacing
    best_pts = None
    best_score = 0.0
    
    for spacing in np.linspace(0.22, 0.30, 20):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Select N points closest to center for better symmetry
            idx = np.argsort(radius)[:N]
            pts = points[idx]
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    if best_pts is not None:
        return best_pts
    
    # Fallback: random
    return np.random.randn(N, 2) * 0.5


def _ring_init():
    """Initialize with points on concentric rings."""
    pts = np.zeros((N, 2))
    # One point at center
    pts[0] = [0.0, 0.0]
    remaining = N - 1
    ring_radii = [0.25, 0.5, 0.75]
    idx = 1
    for r in ring_radii:
        if idx >= N:
            break
        # How many points can fit on this ring?
        circumference = 2 * np.pi * r
        # Approximate max points: circumference / (2*r_approx)
        # Use spacing ~ 0.24
        max_pts = int(circumference / 0.24)
        take = min(max_pts, remaining)
        if take > 0:
            angles = np.linspace(0, 2 * np.pi, take, endpoint=False)
            for a in angles:
                if idx >= N:
                    break
                pts[idx] = [r * np.cos(a), r * np.sin(a)]
                idx += 1
            remaining -= take
    # Fill any remaining with random
    if idx < N:
        for i in range(idx, N):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.6, 0.95)
            pts[i] = [r * np.cos(angle), r * np.sin(angle)]
    return pts


def _random_init():
    """Random initialization inside the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.95
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(5):
        inits.append(_random_init())
    
    best_pts = None
    best_score = 0.0
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        
        # Try BFGS optimization from this initial point
        try:
            result = minimize(
                _negative_score,
                init,
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-6}
            )
            if result.success or result.fun < -best_score:
                s = _compute_score(result.x)
                if s > best_score:
                    best_score = s
                    best_pts = result.x.copy()
        except Exception:
            pass
    
    # If we have time, do a few more random restarts with shorter optimization
    if best_pts is not None and time.time() < deadline - 60:
        for _ in range(10):
            if time.time() > deadline - 30:
                break
            init = _random_init()
            try:
                result = minimize(
                    _negative_score,
                    init,
                    method='BFGS',
                    options={'maxiter': 100, 'gtol': 1e-5}
                )
                s = _compute_score(result.x)
                if s > best_score:
                    best_score = s
                    best_pts = result.x.copy()
            except Exception:
                pass
    
    # Final refinement if we have time
    if best_pts is not None and time.time() < deadline - 15:
        try:
            result = minimize(
                _negative_score,
                best_pts,
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            s = _compute_score(result.x)
            if s > best_score:
                best_score = s
                best_pts = result.x.copy()
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = inits[0]
    
    return best_pts
# EVOLVE-BLOCK-END
