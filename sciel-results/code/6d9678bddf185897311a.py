"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _convex_hull_area(pts):
    """Compute convex hull area; returns 0 if degenerate."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area(pts):
    """Compute minimum area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pj = pts[j]
            for k in range(j + 1, n):
                pk = pts[k]
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                cross = abs((pj[0] - pi[0]) * (pk[1] - pi[1]) -
                            (pj[1] - pi[1]) * (pk[0] - pi[0]))
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return -1.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area


def _make_ring_config(n_rings, radii, angles_per_ring, rotation_offsets=None):
    """Create points on concentric rings."""
    pts = []
    for r_idx, radius in enumerate(radii):
        n_angles = angles_per_ring[r_idx]
        offset = 0.0
        if rotation_offsets is not None and r_idx < len(rotation_offsets):
            offset = rotation_offsets[r_idx]
        for a_idx in range(n_angles):
            angle = offset + 2.0 * np.pi * a_idx / n_angles
            pts.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(pts)


def _voronoi_relax(pts, n_iter=50, step_size=0.1):
    """Move each point toward centroid of its Delaunay-triangulation neighborhood."""
    pts = pts.copy()
    for _ in range(n_iter):
        if _time_remaining() < 5.0:
            break
        try:
            tri = Delaunay(pts)
        except Exception:
            break
        new_pts = pts.copy()
        for i in range(len(pts)):
            # Find neighbors via Delaunay simplices containing point i
            neighbors = set()
            for simplex in tri.simplices:
                if i in simplex:
                    for s in simplex:
                        if s != i:
                            neighbors.add(s)
            if len(neighbors) > 0:
                neighbor_pts = pts[list(neighbors)]
                centroid = neighbor_pts.mean(axis=0)
                # Move toward centroid with damping
                new_pts[i] = pts[i] + step_size * (centroid - pts[i])
        # Normalize: keep within unit circle
        max_r = np.max(np.linalg.norm(new_pts, axis=1))
        if max_r > 1.0:
            new_pts /= max_r
        pts = new_pts
    return pts


def _optimize_local(pts, max_evals=2000):
    """Nelder-Mead optimization on flattened point coordinates."""
    flat = pts.flatten()
    best_x = flat.copy()
    best_score = _score(pts)

    def objective(x):
        if _time_remaining() < 3.0:
            return 1e10  # signal to stop
        p = x.reshape(-1, 2)
        # Keep points within a reasonable bound
        p = np.clip(p, -2.0, 2.0)
        s = _score(p)
        if s < 0:
            return 1e6
        return -s  # minimize negative score

    n_iter = 0
    result = minimize(objective, flat, method='Nelder-Mead',
                      options={'maxiter': max_evals, 'xatol': 1e-6, 'fatol': 1e-8,
                               'adaptive': True})
    if result.x is not None:
        p = result.x.reshape(-1, 2)
        s = _score(p)
        if s > best_score:
            best_x = result.x.copy()
            best_score = s
    return best_x.reshape(-1, 2), best_score


def _random_perturb(pts, sigma=0.05):
    """Add small random perturbation."""
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    return pts + rng.normal(0, sigma, size=pts.shape)


def construct():
    """Construct 14 points maximizing min triangle area / convex hull area."""
    best_pts = None
    best_score = -1.0

    # Strategy 1: Ring-based configurations
    # Try various ring splits: (1, 6, 7), (1, 5, 8), (2, 6, 6), (1, 4, 9), (3, 5, 6)
    ring_configs = [
        # (radii, angles_per_ring, rotation_offsets)
        ([0.0, 0.55, 1.0], [1, 6, 7], [0.0, 0.0, np.pi / 7.0]),
        ([0.0, 0.6, 1.0], [1, 5, 8], [0.0, 0.0, np.pi / 8.0]),
        ([0.0, 0.5, 1.0], [2, 6, 6], [0.0, 0.0, np.pi / 6.0]),
        ([0.0, 0.65, 1.0], [1, 4, 9], [0.0, 0.0, np.pi / 9.0]),
        ([0.0, 0.5, 0.8, 1.0], [1, 4, 5, 4], [0.0, 0.0, np.pi / 5.0, np.pi / 4.0]),
        ([0.0, 0.45, 0.75, 1.0], [1, 5, 5, 3], [0.0, 0.0, np.pi / 5.0, np.pi / 3.0]),
    ]

    for radii, angles, offsets in ring_configs:
        if _time_remaining() < 10.0:
            break
        pts = _make_ring_config(len(radii), radii, angles, offsets)
        # Voronoi relaxation
        pts_relaxed = _voronoi_relax(pts, n_iter=30, step_size=0.08)
        s = _score(pts_relaxed)
        if s > best_score:
            best_score = s
            best_pts = pts_relaxed.copy()

        # Local optimization from this starting point
        if _time_remaining() > 15.0:
            opt_pts, opt_score = _optimize_local(pts_relaxed, max_evals=1500)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()

    # Strategy 2: Try some random initializations with optimization
    if _time_remaining() > 20.0:
        rng = np.random.RandomState(42)
        for attempt in range(3):
            if _time_remaining() < 15.0:
                break
            # Random points in unit disk
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(rng.uniform(0.1, 1.0, N))
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            pts_relaxed = _voronoi_relax(pts, n_iter=20, step_size=0.05)
            opt_pts, opt_score = _optimize_local(pts_relaxed, max_evals=1000)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()

    # Final: ensure we have a valid result
    if best_pts is None or _score(best_pts) <= 0:
        # Fallback: regular 14-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_pts
# EVOLVE-BLOCK-END
