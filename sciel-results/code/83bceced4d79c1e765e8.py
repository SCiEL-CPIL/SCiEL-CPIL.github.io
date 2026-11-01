"""
Packing 200 equal circles in a unit square using L-BFGS-B with Soft-Min Objective.
This approach replaces the non-differentiable min-distance objective with a smooth 
log-sum-exp approximation, allowing gradient-based optimization to effectively 
maximize the minimum distance.
"""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Constants
DEADLINE = 140.0  # Safety margin before 150s kill

def _score(pts):
    """Compute the exact radius for a given set of points."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check bounds
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
        
    # Compute min distance between any two points
    # cdist is efficient for N=200
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Compute min distance to boundary
    # Distance to x=-0.5: 0.5 - x
    # Distance to x=0.5: 0.5 + x
    # Distance to y=-0.5: 0.5 - y
    # Distance to y=0.5: 0.5 + y
    dists_to_boundary = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_boundary_dist = np.min(dists_to_boundary)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _hex_init(seed=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    
    # Estimate spacing
    # Area per hex point ~ s^2 * sqrt(3)/2
    # N * s^2 * sqrt(3)/2 = 1 => s = sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    points = []
    y_step = s * np.sqrt(3) / 2.0
    x_step = s
    
    y = -0.5
    row = 0
    # Iterate while y is within bounds + margin
    while y <= 0.5 + y_step:
        x = -0.5
        # Offset for even/odd rows
        offset = (x_step / 2.0) * (row % 2)
        while x <= 0.5 + x_step:
            px = x + offset
            py = y
            if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5:
                points.append([px, py])
            x += x_step
        y += y_step
        row += 1
    
    pts = np.array(points)
    
    if len(pts) == 0:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N].copy()

    if len(pts) > N:
        # Randomly select N points to break symmetry
        indices = rng.choice(len(pts), N, replace=False)
        pts = pts[indices].copy()
    elif len(pts) < N:
        # Pad with random points
        extra = N - len(pts)
        random_pts = rng.uniform(-0.49, 0.49, size=(extra, 2))
        pts = np.vstack([pts, random_pts])
    
    # Add small jitter to break perfect lattice symmetry which can trap in local minima
    pts += rng.normal(0, 1e-5, size=pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    
    return pts

def _soft_min_neg(x, alpha):
    """
    Objective function for L-BFGS-B.
    x: flat array of size 2N.
    alpha: sharpness of the soft-min.
    Returns: -soft_min(distances)
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()
    
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Numerically stable log-sum-exp for soft min
    # soft_min = -1/alpha * log(sum(exp(-alpha * d)))
    # Use max trick for numerical stability
    min_d = np.min(all_dists)
    # Avoid underflow/overflow
    exp_terms = np.exp(-alpha * (all_dists - min_d))
    log_sum = np.log(np.sum(exp_terms)) + (-alpha * min_d)
    soft_min_val = -log_sum / alpha
    
    # We want to maximize min distance, so minimize -min distance.
    return -soft_min_val

def construct():
    start_time = time.time()
    
    # Initialization
    pts = _hex_init(seed=42)
    current_score = _score(pts)
    best_score = current_score
    best_pts = pts.copy()
    
    # L-BFGS-B Optimization with Multi-Phase Alpha Schedule
    # Alpha controls the sharpness of the soft-min.
    # Low alpha: smooth, global optimization.
    # High alpha: sharp, local optimization.
    
    phases = [
        {"alpha": 5.0, "max_iter": 100, "gtol": 1e-3},
        {"alpha": 15.0, "max_iter": 150, "gtol": 1e-4},
        {"alpha": 50.0, "max_iter": 200, "gtol": 1e-5},
        {"alpha": 150.0, "max_iter": 200, "gtol": 1e-6},
        {"alpha": 500.0, "max_iter": 200, "gtol": 1e-7},
        {"alpha": 1500.0, "max_iter": 200, "gtol": 1e-8},
        {"alpha": 5000.0, "max_iter": 200, "gtol": 1e-9},
    ]
    
    x0 = pts.flatten()
    
    for phase in phases:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        alpha = phase["alpha"]
        
        def obj(x):
            return _soft_min_neg(x, alpha)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                           bounds=[(-0.4999, 0.4999)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"],
                                    'ftol': 1e-12, 'maxls': 50})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            # If optimization fails, keep previous best
            pass
            
        # Update best score after each phase
        pts_candidate = x0.reshape(N, 2)
        score_candidate = _score(pts_candidate)
        if score_candidate > best_score:
            best_score = score_candidate
            best_pts = pts_candidate.copy()
            
    # Final Polish: Local Search
    # L-BFGS might stop at a point where the gradient of the soft-min is small,
    # but the hard-min could be improved by a tiny move.
    # We perform a few iterations of random perturbation and acceptance.
    
    pts = best_pts.copy()
    current_score = best_score
    rng = np.random.default_rng(123)
    
    # Time budget for polish: 10 seconds
    polish_deadline = start_time + DEADLINE - 5
    
    T = 0.001
    step_scale = 0.5
    
    while time.time() < polish_deadline:
        # Pick a random point
        idx = rng.integers(N)
        
        # Propose move
        step_size = T * step_scale
        move = rng.normal(0, step_size, size=2)
        
        new_pts = pts.copy()
        new_pts[idx] += move
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_score = _score(new_pts)
        
        # Acceptance
        if new_score > current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                if delta / T > -50:
                    prob = np.exp(delta / T)
                else:
                    prob = 0.0
            else:
                prob = 1.0
            
            if rng.random() < prob:
                pts = new_pts
                current_score = new_score
        
        # Cool down
        T *= 0.999
        if T < 1e-6:
            T = 1e-6
            
    return best_pts
# EVOLVE-BLOCK-END
