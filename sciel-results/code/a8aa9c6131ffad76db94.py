"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, Bounds

N = 50
_DEADLINE = 150.0  # wall-clock limit from evaluator


# EVOLVE-BLOCK-START
def _score(points):
    """Return the packing radius for an (N,2) array of centres."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # distance to boundary
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    r_boundary = np.min(dist_boundary)
    if r_boundary < 0:
        return 0.0
    # pairwise distances
    # Use vectorized pairwise distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    return float(min(r_boundary, r_pair))


def _hex_lattice_init():
    """Generate a hexagonal lattice configuration inside the unit disk."""
    # Try to find a good spacing
    best_pts = None
    for scale in np.linspace(0.8, 2.5, 40):
        spacing = 1.0 / scale
        pts = []
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    pts.append((x, y))
        pts = np.array(pts)
        if len(pts) >= N:
            # Pick N points closest to origin for better packing
            r = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            candidate = pts[idx]
            if best_pts is None or _score(candidate) > _score(best_pts):
                best_pts = candidate
    if best_pts is None:
        # Fallback: random
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 0.5 * np.ones(N)
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return best_pts


def _ring_init():
    """Generate a ring-structured initialization."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    # Rings
    ring_radii = [0.15, 0.35, 0.55, 0.75, 0.92]
    counts = [6, 12, 18, 13]  # 1 + 6 + 12 + 18 + 13 = 50
    for r, n in zip(ring_radii, counts):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
    pts = np.array(pts[:N])
    return pts


def _random_init(seed=42):
    """Random initialization inside the unit disk."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = rng.uniform(0, 0.95, N)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def _force_relax(points, max_iter=5000, dt=0.001, repulsion_k=1.0, boundary_k=5.0):
    """
    Force-directed relaxation: repulsive forces between circles and attractive force to boundary.
    Uses simple Euler integration with adaptive step size.
    """
    pts = points.copy().astype(float)
    n = len(pts)
    
    for it in range(max_iter):
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Repulsive forces between pairs
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 1e-10:
                    # Add small random perturbation
                    forces[i] += 0.01 * np.array([1, 0])
                    forces[j] += 0.01 * np.array([-1, 0])
                    continue
                # Repulsion: stronger when closer
                if dist < 0.5:  # Only interact when reasonably close
                    force_mag = repulsion_k * (0.5 - dist) / dist
                    forces[i] += force_mag * diff
                    forces[j] -= force_mag * diff
        
        # Boundary attraction: pull points toward the boundary
        for i in range(n):
            r = np.sqrt(np.sum(pts[i] ** 2))
            if r > 1e-10:
                # Attractive force toward radius 1
                force_mag = boundary_k * (1.0 - r)
                forces[i] += force_mag * pts[i] / r
        
        # Update positions
        pts += dt * forces
        
        # Keep points inside the unit disk
        r = np.sqrt(np.sum(pts ** 2, axis=1))
        mask = r > 0.99
        if np.any(mask):
            pts[mask] = 0.99 * pts[mask] / r[mask]
    
    return pts


def _lbfgs_refine(points):
    """
    Refine using L-BFGS-B to maximize the minimum distance.
    We minimize the negative of the packing radius.
    """
    def neg_radius(x):
        pts = x.reshape(N, 2)
        # Check boundary
        r_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        if np.any(r_boundary < 0):
            return 1e6
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        return -min(r_boundary, r_pair)
    
    x0 = points.flatten()
    # Bounds: each coordinate in [-1, 1]
    bounds = Bounds(-1.0, 1.0)
    
    result = minimize(neg_radius, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
    
    best_pts = result.x.reshape(N, 2)
    # Ensure points are inside the disk
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r[mask]
    
    return best_pts


def _adaptive_beta_relax(points, beta_start=0.1, beta_end=0.01, max_iter=2000):
    """
    Relaxation with adaptive repulsion strength (beta).
    Start with strong repulsion to separate overlapping circles, then reduce.
    """
    pts = points.copy().astype(float)
    n = len(pts)
    
    for it in range(max_iter):
        beta = beta_start * (beta_end / beta_start) ** (it / max_iter)
        
        forces = np.zeros_like(pts)
        
        # Repulsive forces
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 1e-10:
                    forces[i] += 0.01 * np.array([1, 0])
                    forces[j] += 0.01 * np.array([-1, 0])
                    continue
                if dist < 1.0:
                    force_mag = beta * (1.0 - dist) / dist
                    forces[i] += force_mag * diff
                    forces[j] -= force_mag * diff
        
        # Boundary force
        for i in range(n):
            r = np.sqrt(np.sum(pts[i] ** 2))
            if r > 1e-10:
                force_mag = 2.0 * (1.0 - r)
                forces[i] += force_mag * pts[i] / r
        
        # Adaptive step size
        step = 0.01 * (1.0 - it / max_iter) + 0.001
        pts += step * forces
        
        # Clamp to disk
        r = np.sqrt(np.sum(pts ** 2, axis=1))
        mask = r > 0.995
        if np.any(mask):
            pts[mask] = 0.995 * pts[mask] / r[mask]
    
    return pts


def construct():
    start_time = time.time()
    
    # Generate multiple initial configurations
    inits = [
        _hex_lattice_init(),
        _ring_init(),
        _random_init(42),
        _random_init(123),
        _random_init(999),
    ]
    
    # Relax each with force-directed method
    relaxed = []
    for init in inits:
        if time.time() - start_time > 80:
            break
        relaxed.append(_adaptive_beta_relax(init, max_iter=3000))
    
    # Refine the best few with L-BFGS-B
    # Sort by score
    scores = [_score(p) for p in relaxed]
    order = np.argsort(scores)[::-1]
    
    best_pts = relaxed[order[0]]
    best_score = scores[order[0]]
    
    # Try to refine top candidates with L-BFGS-B
    for idx in order[:3]:
        if time.time() - start_time > 120:
            break
        refined = _lbfgs_refine(relaxed[idx])
        s = _score(refined)
        if s > best_score:
            best_pts = refined
            best_score = s
    
    # Final check
    if time.time() - start_time > 130:
        return best_pts
    
    # One more round of refinement on the best
    if time.time() - start_time < 135:
        refined = _lbfgs_refine(best_pts)
        s = _score(refined)
        if s > best_score:
            best_pts = refined
    
    return best_pts
# EVOLVE-BLOCK-END
