import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
DEADLINE = 145.0  # safety margin before 150s kill

def _penalty_objective(params, penalty=1000.0, edge_mask=None):
    """
    Penalty objective: minimize -sum(r) + penalty*(overlaps + wall violations).
    Uses Delaunay edges for overlap constraints to reduce O(N^2) to O(N).
    """
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    score = -np.sum(r)

    # Wall boundary violations
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Pairwise overlap violations (only for Delaunay edges)
    if edge_mask is None:
        # Fallback to all pairs if no mask provided (should not happen in main loop)
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = np.maximum(0.0, (r[iu[0]] + r[iu[1]]) - dist)
        pair = np.sum(overlap)
    else:
        # edge_mask is a list of (i, j) tuples
        if len(edge_mask) == 0:
            pair = 0.0
        else:
            idx_i = np.array([e[0] for e in edge_mask])
            idx_j = np.array([e[1] for e in edge_mask])
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = np.maximum(0.0, (r[idx_i] + r[idx_j]) - dist)
            pair = np.sum(overlap)

    return score + penalty * (pair + wall)

def _penalty_grad(params, penalty=1000.0, edge_mask=None):
    """
    Analytic gradient of penalty objective.
    """
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    g = np.zeros_like(params)

    # d(-sum r)/dr = -1
    g[2::3] = -1.0

    # Wall contributions
    for i in range(N):
        if r[i] - x[i] > 0:
            g[3*i] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - x[i]) > 0:
            g[3*i] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - y[i] > 0:
            g[3*i+1] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - y[i]) > 0:
            g[3*i+1] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)

    # Pairwise contributions
    if edge_mask is None:
        iu = np.triu_indices(N, k=1)
        idx_i = iu[0]
        idx_j = iu[1]
    else:
        if len(edge_mask) == 0:
            return g
        idx_i = np.array([e[0] for e in edge_mask])
        idx_j = np.array([e[1] for e in edge_mask])

    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-10)
    ov = (r[idx_i] + r[idx_j]) - dist
    active = ov > 0
    
    if np.any(active):
        ax = idx_i[active]
        ay = idx_j[active]
        adx = dx[active] / dist_safe[active]
        ady = dy[active] / dist_safe[active]
        
        # Vectorized update
        # g[3*i] += -penalty * adx
        # g[3*j] += +penalty * adx
        # g[3*i+1] += -penalty * ady
        # g[3*j+1] += +penalty * ady
        # g[3*i+2] += penalty
        # g[3*j+2] += penalty
        
        # Using np.add.at for accumulation if indices repeat (they don't in simple edges, but safe)
        # However, simple loop is fine for O(N) edges
        for k in range(len(ax)):
            i = ax[k]
            j = ay[k]
            g[3*i] += penalty * (-adx[k])
            g[3*j] += penalty * (+adx[k])
            g[3*i+1] += penalty * (-ady[k])
            g[3*j+1] += penalty * (+ady[k])
            g[3*i+2] += penalty * (+1.0)
            g[3*j+2] += penalty * (+1.0)

    return g

def _compute_delaunay_edges(points):
    """Compute unique edges from Delaunay triangulation."""
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails (e.g., degenerate points)
        return [(i, j) for i in range(N) for j in range(i+1, N)]

def _grid_initial(seed):
    """Grid of equal circles with small jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    # 5x5 grid = 25 points. We need 26.
    # Add one point in the center of the grid or a random spot.
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append([0.1 + 0.8*(i+0.5)/5, 0.1 + 0.8*(j+0.5)/5])
    # Add 26th point near center
    pts.append([0.5 + rng.uniform(-0.05, 0.05), 0.5 + rng.uniform(-0.05, 0.05)])
    
    pts = np.array(pts)
    rng.shuffle(pts)
    
    # Estimate initial radii
    radii = np.zeros(N)
    for i in range(N):
        min_dist = 1.0
        for j in range(N):
            if i != j:
                d = np.linalg.norm(pts[i] - pts[j])
                min_dist = min(min_dist, d)
        min_dist = min(min_dist, pts[i][0], 1-pts[i][0], pts[i][1], 1-pts[i][1])
        radii[i] = min_dist * 0.49
        
    params = np.empty(3 * N)
    params[0::3] = pts[:, 0]
    params[1::3] = pts[:, 1]
    params[2::3] = radii
    
    # Small jitter
    params += rng.uniform(-1e-6, 1e-6, size=params.size)
    return params

def _random_initial(seed):
    """Random initialization with jitter."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(0.1, 0.9, size=(N, 2))
    
    radii = np.zeros(N)
    for i in range(N):
        min_dist = 1.0
        for j in range(N):
            if i != j:
                d = np.linalg.norm(pts[i] - pts[j])
                min_dist = min(min_dist, d)
        min_dist = min(min_dist, pts[i][0], 1-pts[i][0], pts[i][1], 1-pts[i][1])
        radii[i] = min_dist * 0.49
        
    params = np.empty(3 * N)
    params[0::3] = pts[:, 0]
    params[1::3] = pts[:, 1]
    params[2::3] = radii
    
    params += rng.uniform(-1e-6, 1e-6, size=params.size)
    return params

def _enforce_validity(params):
    """
    Enforce all constraints: walls and pairwise non-overlap by shrinking radii.
    This is a post-processing step to ensure the returned solution is strictly valid.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps by shrinking the smaller circle
    # Use Delaunay edges for efficiency in checking, but check all pairs for safety in final validation?
    # For shrinking, checking all pairs is O(N^2) but N=26 is small.
    for _ in range(200):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 1e-9
        if not np.any(active):
            break
        
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle by ov (or split if equal)
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        
        # Re-clamp to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p

def _is_valid(params, tol=1e-6):
    """Check if parameters represent a valid packing."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    # Wall check
    if np.any(r > x + tol) or np.any(r > 1 - x + tol) or np.any(r > y + tol) or np.any(r > 1 - y + tol):
        return False
        
    # Pairwise check
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(r[iu[0]] + r[iu[1]] > dist + tol):
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    # Multi-start with different initializations
    # We will use a mix of grid and random starts.
    # The key structural change is using Delaunay edges for the penalty gradient.
    
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100,
             2025, 314, 161, 271, 141, 414, 265, 358, 979, 323]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        # Alternate between grid and random initialization
        if seed % 2 == 0:
            z0 = _grid_initial(seed)
        else:
            z0 = _random_initial(seed)
            
        # Compute initial Delaunay edges
        points = z0.reshape(N, 3)[:, :2]
        edge_mask = _compute_delaunay_edges(points)
        
        # Optimization loop with adaptive penalty
        # Start with moderate penalty, increase if invalid
        penalty = 100.0
        current_params = z0.copy()
        
        # Inner loop: refine the solution
        # We do a few steps of L-BFGS-B, then recompute Delaunay edges, then continue
        for step in range(10):
            if time.time() > deadline - 5:
                break
                
            def obj(z):
                return _penalty_objective(z, penalty, edge_mask)
            
            def grad(z):
                return _penalty_grad(z, penalty, edge_mask)
            
            try:
                res = minimize(obj, current_params, jac=grad, method='L-BFGS-B', 
                               bounds=bounds_list, options={'maxiter': 50, 'ftol': 1e-8})
                current_params = res.x
            except Exception:
                break
            
            # Check validity and update edges
            if _is_valid(current_params, tol=1e-4):
                # If valid, we can try to increase penalty to push for better score
                # Or just accept and move to next seed if we have a good score
                pass
            
            # Recompute Delaunay edges based on new positions
            points = current_params.reshape(N, 3)[:, :2]
            new_edge_mask = _compute_delaunay_edges(points)
            
            # If edges changed significantly, we might want to restart or just continue
            # For simplicity, just update the mask for the next iteration
            edge_mask = new_edge_mask
            
            # Increase penalty slightly to enforce constraints more strictly
            penalty *= 1.5
            
            # Early termination if we have a very good valid solution
            if _is_valid(current_params, tol=1e-6):
                score = np.sum(current_params.reshape(N, 3)[:, 2])
                if score > best_score:
                    best_score = score
                    best_params = current_params.copy()
                # If we are already very good, we can break to try another seed
                # But let's try a few more steps to polish
                if step > 4:
                    break

        # Final check and enforcement for this seed
        if current_params is not None:
            # Enforce validity by shrinking
            valid_params = _enforce_validity(current_params)
            score = np.sum(valid_params[:, 2])
            
            if score > best_score:
                best_score = score
                best_params = valid_params.flatten().copy()

    # Fallback if no valid solution found
    if best_params is None:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        best_params = np.array(cells[:N]).flatten()
        
    return best_params.reshape(N, 3)

# Ensure the function name is exactly construct
# The task requires: construct() -> array of shape (n, 3)

if __name__ == "__main__":
    # Quick test
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Sum of radii: {np.sum(res[:, 2]):.6f}")
    # Validate
    p = res
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    valid = True
    if np.any(r > x) or np.any(r > 1-x) or np.any(r > y) or np.any(r > 1-y):
        valid = False
    iu = np.triu_indices(26, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx*dx + dy*dy)
    if np.any(r[iu[0]] + r[iu[1]] > dist):
        valid = False
    print(f"Valid: {valid}")
