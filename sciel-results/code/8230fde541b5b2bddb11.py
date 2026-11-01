import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    n = len(pts)
    min_area = float('inf')
    # Vectorized: for each pair (i,j), compute cross product with all k
    for i in range(n):
        for j in range(i + 1, n):
            v = pts[j] - pts[i]
            # cross product of v with (pts[k] - pts[i]) for all k
            diffs = pts - pts[i]
            crosses = diffs[:, 0] * v[1] - diffs[:, 1] * v[0]
            areas = 0.5 * np.abs(crosses)
            # exclude i and j themselves (area 0)
            areas[i] = float('inf')
            areas[j] = float('inf')
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # in 2D, volume = area
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _fast_score(pts):
    """Faster score using vectorized triangle area computation."""
    n = len(pts)
    # Compute all triangle areas vectorized
    min_area = float('inf')
    for i in range(n):
        vi = pts[i]
        for j in range(i + 1, n):
            vj = pts[j]
            # Cross product magnitude for triangle (i, j, k)
            # = 0.5 * | (pj - pi) x (pk - pi) |
            dx = vj[0] - vi[0]
            dy = vj[1] - vi[1]
            # For all k: cross = dx*(pk_y - pi_y) - dy*(pk_x - pi_x)
            cross = dx * (pts[:, 1] - vi[1]) - dy * (pts[:, 0] - vi[0])
            areas = 0.5 * np.abs(cross)
            areas[i] = float('inf')
            areas[j] = float('inf')
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import random

N = 14

def _compute_min_tri_area(pts):
    """Vectorized minimum triangle area."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j, 0] - pts[i, 0]
            dy = pts[j, 1] - pts[i, 1]
            cross = dx * (pts[:, 1] - pts[i, 1]) - dy * (pts[:, 0] - pts[i, 0])
            areas = 0.5 * np.abs(cross)
            areas[i] = 1e18
            areas[j] = 1e18
            lm = np.min(areas)
            if lm < min_area:
                min_area = lm
    return min_area

def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0

def _objective(pts):
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    ma = _compute_min_tri_area(pts)
    return ma / ha

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    # Start with regular polygon inscribed in unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _objective(best_pts)
    
    current_pts = best_pts.copy()
    current_score = best_score
    
    # Simulated annealing
    T0 = 0.01
    Tend = 1e-6
    alpha = 0.9995  # cooling rate
    T = T0
    
    step_size = 0.15
    n_iter = 0
    
    while time.time() < deadline:
        # Check every 1000 iterations for time
        if n_iter % 500 == 0 and time.time() > deadline - 5:
            break
        
        # Generate perturbation
        idx = rng.integers(0, N)
        direction = rng.normal(size=2)
        direction /= (np.linalg.norm(direction) + 1e-12)
        magnitude = step_size * np.exp(rng.normal(0, 0.5))
        new_pts = current_pts.copy()
        new_pts[idx] += magnitude * direction
        
        new_score = _objective(new_pts)
        
        if new_score >= current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta > 0:
                accept = True
            else:
                if T > Tend:
                    accept_prob = np.exp(delta / T)
                    accept = rng.random() < accept_prob
                else:
                    accept = False
            
            if accept:
                current_pts = new_pts
                current_score = new_score
        
        # Cool down
        T = max(T * alpha, Tend)
        
        # Occasionally reheat
        if n_iter % 20000 == 19999 and T < 0.001:
            T = 0.005
        
        # Adapt step size
        if n_iter % 10000 == 9999:
            if best_score > 0:
                step_size = max(0.01, min(0.3, step_size * 0.95))
        
        n_iter += 1
    
    return best_pts
# EVOLVE-BLOCK-END
