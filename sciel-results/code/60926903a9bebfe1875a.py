"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

# Precompute all index triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I1, _I2, _I3 = _TRIPLES[:, 0], _TRIPLES[:, 1], _TRIPLES[:, 2]


def _min_triangle_area(pts):
    """Compute the minimum triangle area over all C(N,3) triples."""
    p1 = pts[_I1]
    p2 = pts[_I2]
    p3 = pts[_I3]
    # Cross product (p2-p1) x (p3-p1)
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross).min()


def _hull_area(pts):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return _min_triangle_area(pts) / ha


def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)


def _random_initial(rng, n=N):
    """Generate random points in unit square."""
    return np.array([rng.random(2) for _ in range(n)])


def _grid_initial(n=N):
    """Generate points on a regular grid (13 points in a 4x4 grid minus one corner)."""
    # 4x4 grid has 16 points, remove 3 to get 13
    # Or use a more natural arrangement
    # Let's use a hexagonal-like packing or just a perturbed grid
    # For 13 points, a 4x4 grid minus 3 corners might work
    xs = np.linspace(0.1, 0.9, 4)
    ys = np.linspace(0.1, 0.9, 4)
    pts = []
    for x in xs:
        for y in ys:
            pts.append([x, y])
    # Remove 3 points to get 13
    # Remove corners to make it more convex-friendly
    pts = pts[:13]
    return np.array(pts)


def _hexagonal_initial(n=N):
    """Generate points in a hexagonal lattice pattern."""
    pts = []
    # Center point
    pts.append([0.5, 0.5])
    # First ring (6 points)
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        pts.append([0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)])
    # Second ring (6 points)
    for angle in np.linspace(np.pi / 6, 2 * np.pi + np.pi / 6, 6, endpoint=False):
        pts.append([0.5 + 0.55 * np.cos(angle), 0.5 + 0.55 * np.sin(angle)])
    # We have 1 + 6 + 6 = 13 points
    return np.array(pts[:n])


def _optimize_from(start_pts, deadline):
    """Optimize from a starting point using L-BFGS-B."""
    best_pts = start_pts.copy()
    best_score = _score(best_pts)
    
    # Flatten points
    x0 = best_pts.flatten()
    
    # Define objective
    def objective(x):
        if time.time() > deadline:
            return 0.0
        pts = x.reshape(N, 2)
        return -_score(pts)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        if result.success or result.fun < objective(x0):
            new_pts = result.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
    except Exception:
        pass
    
    return best_pts, best_score


def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    rng = np.random.RandomState(42)
    
    best_pts = None
    best_score = 0.0
    
    # Strategy 1: Hexagonal initial
    hex_pts = _hexagonal_initial()
    hex_pts, hex_score = _optimize_from(hex_pts, deadline)
    if hex_score > best_score:
        best_pts = hex_pts
        best_score = hex_score
    
    if time.time() > deadline:
        return best_pts if best_pts is not None else _random_initial(rng)
    
    # Strategy 2: Grid initial
    grid_pts = _grid_initial()
    grid_pts, grid_score = _optimize_from(grid_pts, deadline)
    if grid_score > best_score:
        best_pts = grid_pts
        best_score = grid_score
    
    if time.time() > deadline:
        return best_pts if best_pts is not None else _random_initial(rng)
    
    # Strategy 3: Multiple random restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        rand_pts = _random_initial(rng)
        # Small perturbation to break symmetry
        rand_pts += 0.01 * rng.randn(N, 2)
        rand_pts = np.clip(rand_pts, 0.05, 0.95)
        
        opt_pts, opt_score = _optimize_from(rand_pts, deadline)
        if opt_score > best_score:
            best_pts = opt_pts
            best_score = opt_score
    
    if best_pts is None:
        best_pts = _random_initial(rng)
    
    return best_pts
# EVOLVE-BLOCK-END
