import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

N = 50

def _violation_and_grad(centers, R):
    """
    Compute violation and gradient for a given configuration and target radius R.
    Violation is the sum of overlaps with other circles and walls.
    """
    n = len(centers)
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-12, 1e-12, dists)
    
    # Pairwise violations
    # We want d >= 2R. Violation is max(0, 2R - d).
    # Only consider i < j to avoid double counting and self-interaction.
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    pair_overlaps = np.maximum(0, 2 * R - pair_dists)
    
    # Gradient contribution from pairs
    grad = np.zeros_like(centers)
    # For each overlapping pair, add gradient
    # grad_i += (centers[i] - centers[j]) / d
    # grad_j += (centers[j] - centers[i]) / d
    # We need to map iu indices back to i and j
    i_idx, j_idx = iu
    # Calculate unit vectors
    # diff[i, j] is centers[i] - centers[j]
    diff_pair = diff[i_idx, j_idx]
    dists_pair = dists[i_idx, j_idx]
    
    # Only add gradient where there is overlap
    mask = pair_overlaps > 0
    if np.any(mask):
        # Normalized direction
        dirs = diff_pair[mask] / dists_pair[mask, np.newaxis]
        # Accumulate gradients
        # np.add.at for efficient accumulation
        np.add.at(grad, i_idx[mask], dirs)
        np.add.at(grad, j_idx[mask], -dirs)
        
    # Wall violations
    # dist to left: x + 0.5, right: 0.5 - x, bottom: y + 0.5, top: 0.5 - y
    # We want min(0.5 - |x|, 0.5 - |y|) >= R
    # This is equivalent to:
    # 0.5 - |x| >= R  => |x| <= 0.5 - R
    # 0.5 - |y| >= R  => |y| <= 0.5 - R
    # Violation for x: max(0, |x| - (0.5 - R))
    # Violation for y: max(0, |y| - (0.5 - R))
    
    x = centers[:, 0]
    y = centers[:, 1]
    
    limit = 0.5 - R
    
    vx = np.maximum(0, np.abs(x) - limit)
    vy = np.maximum(0, np.abs(y) - limit)
    
    wall_violation = np.sum(vx) + np.sum(vy)
    
    # Gradient for walls
    # d/dx max(0, |x| - limit) = sign(x) if |x| > limit else 0
    # We want to minimize violation, so move towards center.
    # If x > limit, grad is 1, so we subtract grad from x (L-BFGS minimizes).
    # If x < -limit, grad is -1, so we subtract -1 (add 1) to x.
    
    gx = np.zeros_like(x)
    gy = np.zeros_like(y)
    
    mask_x = vx > 0
    gx[mask_x] = np.sign(x[mask_x])
    
    mask_y = vy > 0
    gy[mask_y] = np.sign(y[mask_y])
    
    grad[:, 0] += gx
    grad[:, 1] += gy
    
    total_violation = np.sum(pair_overlaps) + wall_violation
    
    return total_violation, grad

def _objective_flat(x, R):
    centers = x.reshape(N, 2)
    v, _ = _violation_and_grad(centers, R)
    return v

def _grad_flat(x, R):
    centers = x.reshape(N, 2)
    _, g = _violation_and_grad(centers, R)
    return g.flatten()

def _check_feasibility(centers, R, tol=1e-6):
    """
    Check if centers are feasible for radius R.
    Returns True if all constraints are satisfied within tolerance.
    """
    v, _ = _violation_and_grad(centers, R)
    return v < tol

def _construct_initial_grid(rows, cols, extra=0):
    """
    Construct a grid of points.
    """
    # Adjust rows/cols to get at least N points
    # Then slice to N
    if rows * cols < N:
        # Increase cols
        cols = int(np.ceil(N / rows))
    
    xs = np.linspace(-0.5 + 0.01, 0.5 - 0.01, cols)
    ys = np.linspace(-0.5 + 0.01, 0.5 - 0.01, rows)
    
    grid = np.array([(x, y) for y in ys for x in xs])
    
    # If we have more than N, take the first N
    # If we have less, pad with random points? 
    # Better to ensure rows*cols >= N
    if len(grid) < N:
        # Pad with random
        random_pts = np.random.uniform(-0.4, 0.4, size=(N - len(grid), 2))
        grid = np.vstack([grid, random_pts])
        
    return grid[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    best_radius = 0.0
    best_centers = None
    
    # Binary search bounds
    R_low = 0.06
    R_high = 0.08
    
    # Try to find the maximum feasible R
    # We will do a few iterations of binary search.
    # For each R, we run an optimizer to minimize violation.
    
    # Initialization strategies
    def get_inits():
        inits = []
        # 1. 7x7 grid + 1
        grid_49 = _construct_initial_grid(7, 7)
        # Add one point near center?
        # Actually _construct_initial_grid(7,7) returns 49 points.
        # We need 50. Let's add a random point.
        p50 = np.random.uniform(-0.1, 0.1, size=(1, 2))
        init1 = np.vstack([grid_49, p50])
        inits.append(init1)
        
        # 2. 5x10 grid
        init2 = _construct_initial_grid(5, 10)
        inits.append(init2)
        
        # 3. 10x5 grid
        init3 = _construct_initial_grid(10, 5)
        inits.append(init3)
        
        # 4. Random
        np.random.seed(123)
        init4 = np.random.uniform(-0.45, 0.45, size=(N, 2))
        inits.append(init4)
        
        return inits
    
    inits = get_inits()
    
    # Binary search
    for _ in range(15): # Max 15 iterations
        if time.time() > deadline - 20.0:
            break
            
        R_mid = (R_low + R_high) / 2.0
        
        # Try to find a feasible configuration for R_mid
        found_feasible = False
        best_violation_for_R = np.inf
        best_centers_for_R = None
        
        for init in inits:
            if time.time() > deadline - 25.0:
                break
                
            x0 = init.flatten()
            
            try:
                # Use L-BFGS-B to minimize violation
                res = opt.minimize(
                    _objective_flat,
                    x0,
                    jac=_grad_flat,
                    args=(R_mid,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
                
                v = res.fun
                if v < best_violation_for_R:
                    best_violation_for_R = v
                    best_centers_for_R = res.x.reshape(N, 2)
                    
                if v < 1e-6:
                    found_feasible = True
                    break
                    
            except Exception:
                continue
        
        if found_feasible:
            R_low = R_mid
            if best_centers_for_R is not None:
                # Update best centers if this R is better
                if R_mid > best_radius:
                    best_radius = R_mid
                    best_centers = best_centers_for_R.copy()
        else:
            R_high = R_mid
            
    # If we didn't find a feasible solution for the initial R_low, we might have a problem.
    # But R_low=0.06 should be feasible.
    
    # Final polish: Maximize the minimum distance directly starting from best_centers
    if best_centers is not None and time.time() < deadline - 5.0:
        
        def _min_dist_obj(x):
            centers = x.reshape(N, 2)
            # Compute min pair dist / 2
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            
            # Compute min wall dist
            wall_dists = np.column_stack((
                0.5 - np.abs(centers[:, 0]),
                0.5 - np.abs(centers[:, 1])
            ))
            min_wall = np.min(wall_dists)
            
            return -min(min_pair, min_wall)
            
        def _min_dist_grad(x):
            centers = x.reshape(N, 2)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            min_pair = np.min(dists) / 2.0
            wall_dists = np.column_stack((
                0.5 - np.abs(centers[:, 0]),
                0.5 - np.abs(centers[:, 1])
            ))
            min_wall = np.min(wall_dists)
            
            r = min(min_pair, min_wall)
            
            grad = np.zeros_like(centers)
            
            if min_pair < min_wall:
                # Pair constraint active
                i_min, j_min = np.unravel_index(np.argmin(dists), dists.shape)
                vec = diff[i_min, j_min]
                norm = np.linalg.norm(vec)
                if norm > 1e-12:
                    grad[i_min] += vec / norm
                    grad[j_min] -= vec / norm
            else:
                # Wall constraint active
                i_wall, j_wall = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
                if j_wall == 0:
                    grad[i_wall, 0] = -np.sign(centers[i_wall, 0])
                    if centers[i_wall, 0] == 0: grad[i_wall, 0] = 1.0
                else:
                    grad[i_wall, 1] = -np.sign(centers[i_wall, 1])
                    if centers[i_wall, 1] == 0: grad[i_wall, 1] = 1.0
                    
            return grad.flatten()
            
        try:
            x0 = best_centers.flatten()
            res = opt.minimize(
                _min_dist_obj,
                x0,
                jac=_min_dist_grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_centers = res.x.reshape(N, 2)
            # Calculate true radius
            diff = new_centers[:, np.newaxis, :] - new_centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            wall_dists = np.column_stack((
                0.5 - np.abs(new_centers[:, 0]),
                0.5 - np.abs(new_centers[:, 1])
            ))
            min_wall = np.min(wall_dists)
            new_r = min(min_pair, min_wall)
            
            if new_r > best_radius:
                best_radius = new_r
                best_centers = new_centers
                
        except Exception:
            pass
            
    # Ensure validity
    if best_centers is None:
        # Fallback to a simple grid
        best_centers = _construct_initial_grid(7, 7)
        p50 = np.array([[0.0, 0.0]]) # This will overlap, but it's a fallback
        best_centers = np.vstack([best_centers, p50])
        # Actually, let's just return a valid grid
        best_centers = _construct_initial_grid(5, 10)
        
    # Clip to strictly inside
    eps = 1e-9
    best_centers = np.clip(best_centers, -0.5 + eps, 0.5 - eps)
    
    return best_centers
# EVOLVE-BLOCK-END
