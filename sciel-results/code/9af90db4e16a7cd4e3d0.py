"""
Strategy: Direct optimization of centers and radii using a smooth penalty objective.
The current program optimizes anchors and derives radii via Voronoi inradii, which creates
a non-smooth objective landscape (kinks where the closest neighbor changes).
This program optimizes (x, y, r) directly with a smooth penalty for overlaps and boundary violations.
This allows L-BFGS-B to use gradients effectively to find better packings.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def _penalty_objective(params):
    """
    params: (3N,) array of [x1, y1, r1, x2, y2, r2, ...]
    Returns: -sum(r) + penalty(overlaps) + penalty(boundary)
    We minimize this.
    """
    n = N
    params = params.reshape(n, 3)
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # Objective: maximize sum of radii -> minimize -sum(r)
    obj = -np.sum(r)
    
    # Boundary penalty: keep circles inside [0,1]^2
    # If x - r < 0 or x + r > 1, penalty.
    # Smooth penalty: max(0, violation)^2
    # Left/Bottom
    viol_lb_x = np.maximum(0.0, -(x - r))
    viol_lb_y = np.maximum(0.0, -(y - r))
    # Right/Top
    viol_rt_x = np.maximum(0.0, (x + r) - 1.0)
    viol_rt_y = np.maximum(0.0, (y + r) - 1.0)
    
    bound_pen = np.sum(viol_lb_x**2) + np.sum(viol_lb_y**2) + \
                np.sum(viol_rt_x**2) + np.sum(viol_rt_y**2)
    
    # Overlap penalty
    # For each pair i, j, if dist < r_i + r_j, penalty.
    # Smooth: max(0, r_i + r_j - dist)^2
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            viol = np.maximum(0.0, min_dist - dist)
            overlap_pen += viol**2
            
    # Weights for penalties
    # Large weight to ensure validity, but not so large it dominates sum(r) entirely
    # We want valid solutions. If invalid, score is 0. 
    # So we need strong penalties to guide optimizer to valid region.
    w_bound = 100.0
    w_overlap = 100.0
    
    return obj + w_bound * bound_pen + w_overlap * overlap_pen

def _is_valid(anchors, radii, tol=1e-7):
    """
    Verify validity for the final result.
    """
    n = anchors.shape[0]
    for i in range(n):
        x, y, r = anchors[i, 0], anchors[i, 1], radii[i]
        if r < -tol:
            return False
        if x - r < -tol or 1.0 - x - r < -tol or y - r < -tol or 1.0 - y - r < -tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = anchors[i, 0] - anchors[j, 0]
            dy = anchors[i, 1] - anchors[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _grid_init():
    """
    Initial guess: 5x5 grid with one removed, or 5x6 grid with 2 removed?
    26 circles.
    5x5 = 25. Add 1 in center?
    Let's use a 5x5 grid of points, and add one more near the center.
    Or just 5x5 grid and let optimizer split one?
    Better: 5x5 grid, radius ~ 0.1.
    """
    pts = []
    side = 5
    for i in range(side):
        for j in range(side):
            pts.append([(i + 0.5) / side, (j + 0.5) / side, 0.1])
    # We need 26. Add one in the middle, maybe slightly offset?
    # Or just duplicate one? No, overlap penalty will push them apart.
    # Let's add one at (0.5, 0.5) with small radius, and push the center grid point away?
    # Actually, 5x5 is 25. Let's just make it 26 by adding a point.
    pts.append([0.5, 0.5, 0.05])
    return np.array(pts[:N])

def _random_init(rng):
    """
    Random initialization.
    """
    x = rng.random(N)
    y = rng.random(N)
    r = rng.uniform(0.05, 0.15, N)
    return np.column_stack((x, y, r))

def construct():
    start = time.time()
    deadline = start + 140.0
    
    best_params = None
    best_score = -np.inf
    best_valid_score = 0.0
    best_valid_params = None
    
    # Initializations
    inits = [
        _grid_init(),
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(10):
        inits.append(_random_init(rng))
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (max possible is 0.5, but realistically < 0.2)
        
    for init_p in inits:
        if time.time() > deadline - 20.0:
            break
            
        # L-BFGS-B is good for smooth problems with bounds
        # We use the penalty method, so it's smooth.
        try:
            res = minimize(
                _penalty_objective,
                init_p.ravel(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 5.0:
                 # If we are running out of time, just evaluate current best
                 pass
                 
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.reshape(N, 3)
                
            # Check validity of this result
            anchors = best_params[:, :2]
            radii = best_params[:, 2]
            
            if _is_valid(anchors, radii):
                valid_score = np.sum(radii)
                if valid_score > best_valid_score:
                    best_valid_score = valid_score
                    best_valid_params = best_params.copy()
                    
        except Exception:
            pass

    # If no valid solution found, try to repair the best invalid one?
    # Or just return the best valid one found.
    if best_valid_params is not None:
        return best_valid_params
    
    # Fallback: Grid with small radii
    side = 5
    result = np.zeros((N, 3))
    for i in range(N):
        gx = i % side
        gy = i // side
        result[i, 0] = (gx + 0.5) / side
        result[i, 1] = (gy + 0.5) / side
        result[i, 2] = 0.09 # 0.1 might overlap if close? 1/5 = 0.2 spacing. 0.1 radius touches.
        # 0.09 is safe.
        
    # If we found a best_params that was invalid, maybe we can scale down radii to make it valid?
    if best_params is not None:
        # Try to scale radii of best_params to make it valid
        # This is a heuristic repair.
        anchors = best_params[:, :2]
        radii = best_params[:, 2]
        
        # Check max overlap
        max_viol = 0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(anchors[i] - anchors[j])
                viol = radii[i] + radii[j] - dist
                if viol > max_viol:
                    max_viol = viol
        
        # Also check boundary
        for i in range(N):
            x, y, r = anchors[i, 0], anchors[i, 1], radii[i]
            viol = max(0, r - x, r - y, r - (1-x), r - (1-y))
            if viol > max_viol:
                max_viol = viol
        
        if max_viol < 0.1: # If violation is small, scale down
            # Scale all radii by (1 - max_viol)
            scale = 1.0 - max_viol
            new_radii = radii * scale
            new_params = best_params.copy()
            new_params[:, 2] = new_radii
            
            if _is_valid(anchors, new_radii):
                return new_params
            else:
                # Try smaller scale
                for s in [0.9, 0.8, 0.7]:
                    nr = radii * s
                    if _is_valid(anchors, nr):
                        new_params[:, 2] = nr
                        return new_params
                        
    return result
# EVOLVE-BLOCK-END
