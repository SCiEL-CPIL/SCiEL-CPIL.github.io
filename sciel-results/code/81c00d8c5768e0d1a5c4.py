# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRIPLES = _TRIPLES.shape[0]


def _min_triangle_area_fast(points):
    """Fully vectorized min triangle area for N=11 points."""
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    y = np.clip(y, 0, H)
    x_min = y / S3
    x_max = 1 - y / S3
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points


def _bary_to_cart(bary):
    """Convert barycentric coordinates (a, b, c) to Cartesian.
    point = b*(1,0) + c*(0.5, H)
    so x = b + 0.5*c, y = H*c
    """
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert Cartesian to barycentric coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    c = y / H
    b = x - 0.5 * c
    a = 1 - b - c
    return np.column_stack([a, b, c])


def _fps_init(n_points=11, n_seed=3, seed=42):
    """Farthest Point Sampling initialization.
    Starts with n_seed random points, then adds points that maximize the minimum distance to existing points.
    This ensures a well-spread initial configuration.
    """
    rng = np.random.RandomState(seed)
    
    # Start with corners (or near corners) to ensure boundary coverage
    # Barycentric corners: (1,0,0), (0,1,0), (0,0,1)
    # But we need them strictly inside? No, "on or inside".
    # Let's start with 3 points near the corners.
    init_bary = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9]
    ])
    points = _bary_to_cart(init_bary)
    
    # Add more points using FPS
    for _ in range(n_points - n_seed):
        # Calculate min distance from each candidate in a grid to existing points
        # Instead of a grid, we can just pick a random point and check, or use a coarse grid.
        # For speed, let's use a coarse grid of candidates.
        # Grid size ~ 20x20 in barycentric?
        # Simpler: Pick 100 random candidates, evaluate min dist to existing, pick best.
        candidates_bary = rng.dirichlet(np.ones(n_points), size=100) # Wait, dirichlet is for 1 point.
        candidates_bary = np.zeros((100, 3))
        for i in range(100):
            candidates_bary[i] = rng.dirichlet(np.ones(3))
        
        candidates_cart = _bary_to_cart(candidates_bary)
        
        best_dist = -1
        best_point = None
        
        # Vectorized min distance to existing points
        # dist_sq[i, j] = ||candidate_i - point_j||^2
        # We want min_j dist_sq[i, j], then max_i min_j dist_sq[i, j]
        
        # existing points: (3, 2)
        # candidates: (100, 2)
        # diff: (100, 3, 2) - (1, 3, 2) -> (100, 3, 2)
        diff = candidates_cart[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        min_dist_sq = np.min(dist_sq, axis=1)
        
        best_idx = np.argmax(min_dist_sq)
        best_point = candidates_cart[best_idx]
        
        points = np.vstack([points, best_point])
        
    return points


def _simulated_annealing(points, deadline):
    """Simulated Annealing to maximize min triangle area."""
    current_points = points.copy()
    current_score = _min_triangle_area_fast(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    T = 0.05 # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.99
    
    rng = np.random.RandomState(123)
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        # Generate a neighbor: move one random point by a small amount
        i = rng.randint(N)
        # Step size proportional to T
        step = T * 2.0
        delta = rng.randn(2) * step
        new_points = current_points.copy()
        new_points[i] += delta
        new_points = _clip_to_triangle(new_points)
        
        new_score = _min_triangle_area_fast(new_points)
        
        # Delta in objective (we want to maximize score, so minimize -score)
        delta_score = new_score - current_score
        
        if delta_score > 0 or rng.rand() < np.exp(delta_score / T):
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= cooling_rate
        
    return best_points, best_score


def _local_search_combinatorial(points, deadline):
    """Combinatorial local search:
    Identify the minimum area triangle.
    Try to move its vertices to increase the area.
    """
    current_points = points.copy()
    current_score = _min_triangle_area_fast(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    rng = np.random.RandomState(42)
    
    for _ in range(500): # Max iterations
        if time.time() > deadline - 5:
            break
            
        # Find the minimum area triangle
        p1_idx, p2_idx, p3_idx = _TRIPLES[np.argmin(0.5 * np.abs((current_points[_TRIPLES[:,1]] - current_points[_TRIPLES[:,0]])[:,0] * (current_points[_TRIPLES[:,2]] - current_points[_TRIPLES[:,0]])[:,1] - (current_points[_TRIPLES[:,1]] - current_points[_TRIPLES[:,0]])[:,1] * (current_points[_TRIPLES[:,2]] - current_points[_TRIPLES[:,0]])[:,0]))]
        
        # Try to move one of these points
        # We want to increase the area of this triangle.
        # Area = 0.5 * |cross(p2-p1, p3-p1)|
        # Gradient of area w.r.t p1, p2, p3?
        # Or just try small moves in random directions and accept if min area increases.
        
        # Pick a random vertex from the critical triangle
        vertex_idx = rng.choice([p1_idx, p2_idx, p3_idx])
        
        # Try 10 random moves
        for _ in range(10):
            step = 0.01
            delta = rng.randn(2) * step
            new_points = current_points.copy()
            new_points[vertex_idx] += delta
            new_points = _clip_to_triangle(new_points)
            
            new_score = _min_triangle_area_fast(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                break # Accept first improvement
                
    return best_points, best_score


def _objective_cartesian(flat):
    """Objective for Nelder-Mead: minimize negative min triangle area with penalty."""
    points = flat.reshape(N, 2)
    min_area = _min_triangle_area_fast(points)
    
    x = points[:, 0]
    y = points[:, 1]
    eps = 1e-8
    penalty = 0.0
    penalty += np.sum(np.maximum(0, -y - eps)) * 100
    penalty += np.sum(np.maximum(0, y - S3 * x + eps)) * 100
    penalty += np.sum(np.maximum(0, y - S3 * (1 - x) + eps)) * 100
    
    return -min_area + penalty


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_points = None
    best_score = -np.inf
    
    # Generate multiple initializations
    inits = []
    
    # 1. FPS Init
    inits.append(_fps_init(seed=1))
    inits.append(_fps_init(seed=2))
    inits.append(_fps_init(seed=3))
    
    # 2. Structured Init (Centroid based)
    # Simple grid in barycentric
    bary_grid = np.array([
        [0.1, 0.1, 0.8],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.33, 0.33, 0.34],
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
        [0.4, 0.4, 0.2]
    ])
    inits.append(_bary_to_cart(bary_grid))
    
    for init in inits:
        if time.time() > deadline - 30:
            break
            
        init = _clip_to_triangle(init)
        
        # Phase 1: Simulated Annealing
        sa_pts, sa_score = _simulated_annealing(init, deadline)
        
        # Phase 2: Combinatorial Local Search
        ls_pts, ls_score = _local_search_combinatorial(sa_pts, deadline)
        
        # Phase 3: Nelder-Mead Polish
        if time.time() < deadline - 10:
            try:
                res = minimize(
                    _objective_cartesian,
                    ls_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < 0:
                    nm_pts = res.x.reshape(N, 2)
                    nm_pts = _clip_to_triangle(nm_pts)
                    nm_score = _min_triangle_area_fast(nm_pts)
                    if nm_score > ls_score:
                        ls_pts = nm_pts
                        ls_score = nm_score
            except Exception:
                pass
        
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_pts.copy()
            
    if best_points is None:
        best_points = inits[0]
        
    return _clip_to_triangle(best_points)
# EVOLVE-BLOCK-END
