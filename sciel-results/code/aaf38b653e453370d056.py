"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 13

# Precompute all triangle index triples
def _get_triples(n):
    """Return array of shape (n_tri, 3) with all C(n,3) index triples."""
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    return np.stack([i[mask], j[mask], k[mask]], axis=1)

_TRIPLES = _get_triples(N)
N_TRI = _TRIPLES.shape[0]

def _triangle_areas(points):
    """Compute areas of all C(n,3) triangles. points: (n, 2). Returns (n_tri,) array."""
    p_i = points[_TRIPLES[:, 0]]
    p_j = points[_TRIPLES[:, 1]]
    p_k = points[_TRIPLES[:, 2]]
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute min triangle area / hull area. Returns the ratio (higher is better)."""
    points = np.asarray(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _soft_min(areas, tau=0.01):
    """Soft minimum using log-sum-exp (differentiable surrogate for min)."""
    # soft-min = -log(sum(exp(-areas/tau)) * tau)
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum * tau

def _soft_objective(points, tau=0.01):
    """Differentiable surrogate: -soft_min(triangle areas) / hull_area."""
    points = np.asarray(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas(points)
    sm = _soft_min(areas, tau)
    return -sm / hull_area

def _construct_initial(seed):
    """Generate an initial configuration with some structure."""
    rng = np.random.RandomState(seed)
    # Start with points on a circle with some perturbation
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 0.1, N)
    radii = 1.0 + rng.uniform(-0.1, 0.1, N)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points

def _polish(points, deadline, max_iter=2000):
    """Local polish using L-BFGS on the exact objective."""
    start = time.time()
    x0 = points.flatten()
    
    def obj(x):
        if time.time() > deadline - 5:
            return 1e10
        pts = x.reshape(N, 2)
        return _negative_score(pts)
    
    result = minimize(
        obj, x0, method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return result.x.reshape(N, 2)

def _polish_soft(points, deadline, max_iter=1000):
    """Polish using the soft-min surrogate for smoother gradients."""
    start = time.time()
    x0 = points.flatten()
    tau = 0.005
    
    def obj(x):
        if time.time() > deadline - 5:
            return 1e10
        pts = x.reshape(N, 2)
        return _soft_objective(pts, tau)
    
    result = minimize(
        obj, x0, method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return result.x.reshape(N, 2)

def _local_search(points, deadline, n_restarts=5):
    """Perturb-and-restart local search."""
    best = points.copy()
    best_score = _score(best)
    rng = np.random.RandomState(42)
    
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        # Perturb
        noise_scale = 0.05 / (r + 1)
        perturbed = best + rng.normal(0, noise_scale, best.shape)
        
        # Polish
        polished = _polish(perturbed, deadline)
        score = _score(polished)
        
        if score > best_score:
            best = polished
            best_score = score
    
    return best, best_score

def construct():
    """Main entry point: construct 13 points to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Multiple initial configurations with soft-min polish then exact polish
    n_initials = 8
    for i in range(n_initials):
        if time.time() > deadline - 20:
            break
        
        seed = 100 + i
        pts = _construct_initial(seed)
        
        # Soft-min polish (smoother, finds good basin)
        pts_soft = _polish_soft(pts, deadline, max_iter=500)
        
        # Exact polish
        pts_exact = _polish(pts_soft, deadline, max_iter=1000)
        
        score = _score(pts_exact)
        if score > best_score:
            best_score = score
            best_points = pts_exact.copy()
    
    # Phase 2: Local search with restarts from best point
    if best_points is not None and time.time() < deadline - 15:
        best_points, best_score = _local_search(best_points, deadline, n_restarts=10)
    
    # Phase 3: Final high-precision polish
    if best_points is not None and time.time() < deadline - 5:
        final = _polish(best_points, deadline, max_iter=3000)
        final_score = _score(final)
        if final_score > best_score:
            best_points = final
            best_score = final_score
    
    if best_points is None:
        # Fallback: simple regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
