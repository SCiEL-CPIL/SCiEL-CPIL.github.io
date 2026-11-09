# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-10:
        return 0.0
        
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def _smoothed_objective(params: np.ndarray, beta: float = 200.0) -> float:
    """
    Objective function for minimization.
    We want to maximize A_min / A_hull.
    Equivalent to minimizing -log(A_min) + log(A_hull).
    To handle the non-differentiable min, we use a soft-min approximation:
    -log(A_min) ≈ (1/beta) * log(sum(exp(-beta * A_i)))
    
    So objective = (1/beta) * log(sum(exp(-beta * A_i))) + log(A_hull)
    """
    points = params.reshape(13, 2)
    
    # Check for degeneracy
    if np.any(~np.isfinite(points)):
        return 1e10
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
        if A_hull <= 1e-10:
            return 1e10
    except Exception:
        return 1e10
        
    idx = np.arange(13)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Avoid log(0)
    areas = np.maximum(areas, 1e-12)
    
    # Soft min approximation
    # log(sum(exp(-beta * A))) / beta
    # For numerical stability, subtract max exponent
    exponents = -beta * areas
    max_exp = np.max(exponents)
    soft_min_term = (max_exp + np.log(np.sum(np.exp(exponents - max_exp)))) / beta
    
    # log(A_hull)
    log_hull_term = np.log(A_hull)
    
    return soft_min_term + log_hull_term

def _generate_initializations():
    """
    Generate diverse initial configurations.
    Focus on known good structures for Heilbronn problem.
    """
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    pts = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    inits.append(pts)
    
    # 2. Fejes Toth / Hexagonal lattice approximation
    # This is often the optimal structure for small N.
    # Arrange in a hexagonal grid pattern.
    # 13 points can be arranged as a hexagon of radius 2 (1+6+6=13? No, 1+6+12=19 is too many).
    # A hexagon of radius 1 has 7 points.
    # A hexagon of radius 2 has 19 points.
    # For 13, we can take a subset or use a rectangular grid with hexagonal offset.
    
    # Let's try a "hexagonal packing" in a box.
    # Rows: 3, 4, 4, 2 or similar.
    # Better: Use the known optimal for N=13? 
    # Actually, for N=13, the optimal is often a subset of the triangular lattice.
    # Let's construct a triangular lattice subset.
    
    # Triangular lattice points:
    # y=0: x=0, 1, 2, 3
    # y=sqrt(3)/2: x=0.5, 1.5, 2.5
    # y=sqrt(3): x=0, 1, 2
    # y=sqrt(3)*1.5: x=0.5, 1.5
    # Total: 4+3+2+2 = 11. Need 2 more.
    
    # Let's just use a well-spaced grid and let the optimizer fix it.
    # Grid 4x4 = 16, remove 3 corners?
    # Or just use the hexagonal rows from the seed, but optimized.
    
    # Row-based hexagonal:
    pts = []
    # Row 0: 3 pts
    for x in [0.2, 0.5, 0.8]: pts.append([x, 0.8])
    # Row 1: 4 pts
    for x in [0.1, 0.36, 0.63, 0.9]: pts.append([x, 0.53])
    # Row 2: 4 pts
    for x in [0.1, 0.36, 0.63, 0.9]: pts.append([x, 0.26])
    # Row 3: 2 pts
    for x in [0.3, 0.7]: pts.append([x, 0.0])
    inits.append(np.array(pts))
    
    # 3. Concentric rings
    pts = [[0.5, 0.5]]
    r1 = 0.2
    for i in range(6):
        a = np.pi / 3 * i
        pts.append([0.5 + r1 * np.cos(a), 0.5 + r1 * np.sin(a)])
    r2 = 0.4
    for i in range(6):
        a = np.pi / 3 * i + np.pi / 6
        pts.append([0.5 + r2 * np.cos(a), 0.5 + r2 * np.sin(a)])
    inits.append(np.array(pts))
    
    # 4. Random (for diversity)
    rng = np.random.default_rng(42)
    for _ in range(3):
        pts = rng.random((13, 2)) * 0.8 + 0.1
        inits.append(pts)
        
    return inits

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using Gradient-Based Optimization.
    """
    start_time = time.time()
    deadline = 600  # seconds, with a safety margin
    
    inits = _generate_initializations()
    
    best_points = None
    best_score = -np.inf
    
    # Bounds for points: keep them in [0.01, 0.99] to avoid boundary issues with Hull
    bounds = [(0.01, 0.99)] * 26
    
    for i, init_pts in enumerate(inits):
        if time.time() - start_time > deadline - 10:
            break
            
        current_points = init_pts.copy()
        current_score = _compute_score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # Flatten for optimizer
        x0 = current_points.flatten()
        
        try:
            # Use L-BFGS-B as it handles bounds well
            # maxiter and maxfun need to be tuned based on time
            # Each function evaluation is ~1-2ms.
            # 600s / 0.002s = 300,000 evals possible.
            # But we have multiple inits.
            
            # Allocate time per init
            time_remaining = deadline - 10 - (time.time() - start_time)
            # Be conservative
            max_evals = int(time_remaining * 500) # ~500 evals per second estimate
            
            res = minimize(
                _smoothed_objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 1000, 'maxfun': max_evals, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < _smoothed_objective(x0):
                new_points = res.x.reshape(13, 2)
                # Clip to ensure validity
                new_points = np.clip(new_points, 0.01, 0.99)
                new_score = _compute_score(new_points)
                
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
                    
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    
        except Exception:
            pass
            
    # Fallback
    if best_points is None:
        best_points = inits[0]
        
    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = np.array([
            [0.2, 0.8], [0.5, 0.8], [0.8, 0.8],
            [0.1, 0.53], [0.36, 0.53], [0.63, 0.53], [0.9, 0.53],
            [0.1, 0.26], [0.36, 0.26], [0.63, 0.26], [0.9, 0.26],
            [0.3, 0.0], [0.7, 0.0]
        ])
        
    return best_points

# EVOLVE-BLOCK-END
