"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute convex hull area."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _regular_polygon(n, radius=1.0):
    """Regular n-gon centered at origin."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _grid_initial(n):
    """Grid-like initial configuration."""
    # Arrange in roughly square grid
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    pts = []
    for i in range(n):
        r = i // cols
        c = i % cols
        # Add slight perturbation to avoid degeneracy
        x = c + 0.1 * np.random.RandomState(i*7+3).uniform(-0.1, 0.1)
        y = r + 0.1 * np.random.RandomState(i*11+5).uniform(-0.1, 0.1)
        pts.append([x, y])
    return np.array(pts)

def _structured_initial(n, seed=42):
    """Structured initial: mix of polygon and grid."""
    rng = np.random.RandomState(seed)
    # Start with regular polygon, add small perturbations
    pts = _regular_polygon(n, radius=1.0)
    pts = pts + 0.05 * rng.randn(n, 2)
    return pts

def _optimize_from_initial(pts0, deadline):
    """Run Nelder-Mead optimization from initial points."""
    def objective(x):
        if time.time() > deadline:
            return 0.0
        pts = x.reshape(N, 2)
        return -_score(pts)
    
    x0 = pts0.flatten()
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                               'adaptive': True})
        if time.time() > deadline:
            return pts0
        best_pts = res.x.reshape(N, 2)
        return best_pts
    except Exception:
        return pts0

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    
    # Generate diverse initial configurations
    inits = []
    
    # Regular polygon
    inits.append(_regular_polygon(N))
    
    # Grid
    inits.append(_grid_initial(N))
    
    # Structured perturbations with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        inits.append(_structured_initial(N, seed=seed))
    
    # More grid variants
    for seed in [7, 13, 23]:
        rng = np.random.RandomState(seed)
        cols = 4
        rows = 4
        pts = []
        for i in range(N):
            r = i // cols
            c = i % cols
            x = c + rng.uniform(-0.15, 0.15)
            y = r + rng.uniform(-0.15, 0.15)
            pts.append([x, y])
        inits.append(np.array(pts))
    
    # Evaluate all initial configs
    for i, pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
    
    # Optimize from best few initial configs
    if best_pts is not None:
        # Sort inits by score, optimize top ones
        scored_inits = []
        for pts in inits:
            sc = _score(pts)
            scored_inits.append((sc, pts))
        scored_inits.sort(key=lambda x: -x[0])
        
        # Try optimizing top 3 initial configs
        for sc, pts0 in scored_inits[:3]:
            if time.time() > deadline - 15:
                break
            opt_pts = _optimize_from_initial(pts0, deadline)
            if opt_pts is not None:
                opt_score = _score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts.copy()
    
    # Additional local refinement with smaller steps if time allows
    if best_pts is not None and time.time() < deadline - 20:
        def objective2(x):
            if time.time() > deadline:
                return 0.0
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        x0 = best_pts.flatten()
        try:
            res = minimize(objective2, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9,
                                   'adaptive': True, 'initial_simplex_size': 0.01})
            if time.time() < deadline:
                refined_pts = res.x.reshape(N, 2)
                refined_score = _score(refined_pts)
                if refined_score > best_score:
                    best_score = refined_score
                    best_pts = refined_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: regular polygon
        best_pts = _regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
