"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import cKDTree
from scipy.optimize import minimize

def _compute_radius(c):
    """Compute the largest common radius for centres c (shape (N, 3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    
    # Wall distances
    abs_c = np.abs(c)
    wall_dists = 0.5 - abs_c
    min_wall = np.min(wall_dists, axis=1)
    min_wall_global = np.min(min_wall)
    
    # Pairwise distances
    # Use KDTree for speed in evaluation, though for N=100 direct is fine.
    # But we use KDTree in the search, so let's keep it consistent or just direct for accuracy.
    # Direct is safer for exact score reporting.
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    return min(min_wall_global, min_dist / 2.0)

def _get_min_clearance(c):
    """
    Fast computation of the minimum clearance (radius) using KDTree.
    Returns (r_min, min_wall_global, min_pair_dist_half)
    """
    # Wall clearance
    abs_c = np.abs(c)
    wall_dists = 0.5 - abs_c
    min_wall_per_point = np.min(wall_dists, axis=1)
    min_wall_global = np.min(min_wall_per_point)
    
    # Pairwise clearance
    tree = cKDTree(c)
    # Query for 2 nearest neighbors for each point
    # dists[0] is self (0), dists[1] is nearest neighbor
    dists, _ = tree.query(c, k=2)
    min_nn_dist = np.min(dists[:, 1])
    
    r_pair = min_nn_dist / 2.0
    r_min = min(min_wall_global, r_pair)
    
    return r_min, min_wall_global, r_pair

def _build_initial_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _build_bcc_init():
    """Body-Centered Cubic initialization."""
    # Approximate BCC structure
    # 4x4x6 grid of cells? 4*4*6 = 96 cells. Each cell 2 points -> 192 points.
    # Let's just do a dense random or grid and let the optimizer fix it.
    # A good structured start is better.
    # 5x5x4 = 100 points in a simple cubic grid scaled to fit.
    nx, ny, nz = 5, 5, 4
    points = []
    a = 1.0 / max(nx, ny, nz)
    # Center the grid
    offset_x = -(nx - 1) * a / 2.0
    offset_y = -(ny - 1) * a / 2.0
    offset_z = -(nz - 1) * a / 2.0
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                points.append([
                    i * a + offset_x,
                    j * a + offset_y,
                    k * a + offset_z
                ])
    return np.array(points[:N])

def _local_search(c, time_limit):
    """
    Fast local search using KDTree.
    Strategy: 
    1. Identify the "bottleneck" constraints (points with lowest clearance).
    2. Propose moves to increase clearance.
    3. Accept if global minimum clearance increases.
    """
    c = c.copy()
    current_r, min_wall, min_pair = _get_min_clearance(c)
    
    # Pre-compute which points are on the bottleneck
    # We need to know which pairs/walls define the current minimum.
    
    max_iters = 20000
    start_time = time.time()
    
    # Adaptive step size
    step_size = 0.005
    
    for i in range(max_iters):
        if time.time() > time_limit:
            break
            
        # Find the points with the lowest clearance
        abs_c = np.abs(c)
        wall_dists = 0.5 - abs_c
        min_wall_per_point = np.min(wall_dists, axis=1)
        
        tree = cKDTree(c)
        dists, _ = tree.query(c, k=2)
        min_nn_dist_per_point = dists[:, 1]
        min_pair_per_point = min_nn_dist_per_point / 2.0
        
        # Clearance per point is min(wall, pair)
        clearance_per_point = np.minimum(min_wall_per_point, min_pair_per_point)
        
        # The global minimum is the min of these.
        # Find indices of points with clearance close to the global min
        # This defines the "active" set of constraints.
        active_mask = clearance_per_point < (current_r + 1e-6)
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) == 0:
            # Should not happen, but safety
            active_indices = [np.argmin(clearance_per_point)]
        
        # Pick a random active point to move
        idx = np.random.choice(active_indices)
        
        # Determine direction to move
        # We want to move away from the closest obstacle (wall or neighbor)
        # Let's check if wall or pair is the limiting factor for this point
        if min_wall_per_point[idx] <= min_pair_per_point[idx]:
            # Wall is limiting. Move away from the closest wall.
            # Find closest wall
            x, y, z = c[idx]
            # Distances to 6 walls
            d_x_min = 0.5 - x
            d_x_max = 0.5 + x
            d_y_min = 0.5 - y
            d_y_max = 0.5 + y
            d_z_min = 0.5 - z
            d_z_max = 0.5 + z
            
            dists_to_walls = np.array([d_x_min, d_x_max, d_y_min, d_y_max, d_z_min, d_z_max])
            min_d = np.min(dists_to_walls)
            min_idx = np.argmin(dists_to_walls)
            
            # Direction vector
            dir_vec = np.zeros(3)
            if min_idx == 0: dir_vec[0] = 1.0  # Move +x
            elif min_idx == 1: dir_vec[0] = -1.0 # Move -x
            elif min_idx == 2: dir_vec[1] = 1.0  # Move +y
            elif min_idx == 3: dir_vec[1] = -1.0 # Move -y
            elif min_idx == 4: dir_vec[2] = 1.0  # Move +z
            elif min_idx == 5: dir_vec[2] = -1.0 # Move -z
        else:
            # Pair is limiting. Move away from nearest neighbor.
            # Find nearest neighbor
            tree = cKDTree(c)
            _, nn_idx = tree.query(c[idx], k=2)
            nn_point = c[nn_idx[1]]
            dir_vec = c[idx] - nn_point
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-9:
                dir_vec = np.random.normal(size=3)
                norm = np.linalg.norm(dir_vec)
            dir_vec = dir_vec / norm
            
        # Propose a move
        # Add some noise to avoid local minima cycling
        noise = np.random.normal(0, 0.1, size=3)
        move_dir = dir_vec + noise
        move_dir = move_dir / (np.linalg.norm(move_dir) + 1e-9)
        
        c_new = c.copy()
        c_new[idx] = c[idx] + move_dir * step_size
        
        # Ensure inside cube (hard constraint for validity, but we optimize radius)
        # If it goes outside, the radius drops to 0 or negative. 
        # Better to clip or reject? 
        # Rejecting is safer for maintaining valid state during search, 
        # but we want to maximize radius. If we clip, we might stick to wall.
        # Let's just check the new radius.
        
        new_r, _, _ = _get_min_clearance(c_new)
        
        if new_r > current_r + 1e-10:
            c = c_new
            current_r = new_r
            # Slightly increase step size if we made progress? 
            # Or decrease to refine?
            # Keep step size constant or adapt.
            # Let's keep it constant for now, maybe scale down slowly.
        else:
            # No improvement. 
            # Reduce step size slightly to allow finer moves?
            # Or try a different point?
            # We already picked a random active point.
            # Let's try a smaller step if we've been stuck?
            # Simple strategy: if no improvement in last 100 steps, reduce step size.
            # For simplicity, let's just keep going.
            pass
            
        # Optional: Occasionally try a larger "kick" to escape local minima
        if i % 500 == 0 and i > 0:
             # Random jitter on all points
             c += np.random.normal(0, 0.001, size=(N, 3))
             c = np.clip(c, -0.5, 0.5)
             # Recalculate current_r
             current_r, _, _ = _get_min_clearance(c)
             if current_r < 0: # Should not happen if clipped
                 current_r = 0

    return c, current_r

def _polish_lbfgs(c, time_limit):
    """
    Final polish using L-BFGS-B with a smooth approximation.
    We use a simpler objective: minimize the sum of squared "violations" 
    or just maximize the min clearance using a soft-min.
    Since N=100, the O(N^2) cost is ~10k ops. 
    L-BFGS-B might take ~10-20 iterations to converge well.
    """
    def obj(x):
        c = x.reshape(N, 3)
        # We want to maximize r.
        # Use a soft-min of the clearances.
        # Clearances: wall_dists (N*6) and pair_dists/2 (N*(N-1)/2)
        
        abs_c = np.abs(c)
        walls = np.stack([
            0.5 - c[:, 0], 0.5 + c[:, 0],
            0.5 - c[:, 1], 0.5 + c[:, 1],
            0.5 - c[:, 2], 0.5 + c[:, 2]
        ], axis=1).flatten()
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pairs = dist.flatten() / 2.0
        pairs = pairs[pairs < np.inf]
        
        all_vals = np.concatenate([walls, pairs])
        
        # Soft min: -1/beta * log(sum(exp(-beta * v)))
        # To maximize min, we minimize -soft_min.
        # Or just minimize soft_min and negate in result? 
        # Let's minimize the negative of the soft min.
        # f(x) = - soft_min(v)
        
        beta = 500.0
        m = np.min(all_vals)
        # Stable log sum exp
        # log(sum(exp(-beta * v))) = -beta * m + log(sum(exp(-beta * (v - m))))
        terms = np.exp(-beta * (all_vals - m))
        log_sum = np.log(np.sum(terms))
        soft_min_val = -m + log_sum / beta
        
        return -soft_min_val # Minimize this to maximize soft_min

    def grad(x):
        c = x.reshape(N, 3)
        abs_c = np.abs(c)
        walls = np.stack([
            0.5 - c[:, 0], 0.5 + c[:, 0],
            0.5 - c[:, 1], 0.5 + c[:, 1],
            0.5 - c[:, 2], 0.5 + c[:, 2]
        ], axis=1).flatten()
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pairs = dist.flatten() / 2.0
        pairs = pairs[pairs < np.inf]
        
        all_vals = np.concatenate([walls, pairs])
        
        beta = 500.0
        m = np.min(all_vals)
        exp_terms = np.exp(-beta * (all_vals - m))
        sum_exp = np.sum(exp_terms)
        p = exp_terms / sum_exp
        
        # Gradient of -soft_min is - grad(soft_min)
        # grad(soft_min) = - sum(p_k * dv_k/dx)
        # So grad(-soft_min) = sum(p_k * dv_k/dx)
        
        grad_x = np.zeros_like(c)
        
        # Wall grads
        p_walls = p[:N*6].reshape(N, 6)
        # dv/dx for walls:
        # 0: 0.5 - x -> -1
        # 1: 0.5 + x -> +1
        # ...
        grad_x[:, 0] += p_walls[:, 0] * (-1) + p_walls[:, 1] * (1)
        grad_x[:, 1] += p_walls[:, 2] * (-1) + p_walls[:, 3] * (1)
        grad_x[:, 2] += p_walls[:, 4] * (-1) + p_walls[:, 5] * (1)
        
        # Pair grads
        p_pairs = p[N*6:]
        P_mat = np.zeros((N, N))
        idx = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                P_mat[i, j] = p_pairs[idx]
                idx += 1
                
        dist_safe = np.where(dist > 1e-8, dist, 1e-8)
        coeff = P_mat / (2.0 * dist_safe)
        weighted_diff = coeff[:, :, np.newaxis] * diff
        grad_x += np.sum(weighted_diff, axis=1)
        
        return grad_x.flatten()

    try:
        res = minimize(obj, c.flatten(), jac=grad, method='L-BFGS-B',
                       options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
        c_polished = res.x.reshape(N, 3)
        c_polished = np.clip(c_polished, -0.5, 0.5)
        return c_polished
    except Exception:
        return c

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # 1. Generate Initializations
    inits = [
        _build_initial_grid(),
        _build_bcc_init(),
        np.random.uniform(-0.45, 0.45, size=(N, 3))
    ]
    
    best_c = None
    best_score = -1.0
    
    # 2. Run Local Search on each init
    # Allocate time: ~40s per init? 3 inits * 40s = 120s.
    # Plus polish time.
    
    for i, c0 in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
            
        time_left_for_ls = min(30.0, deadline - time.time() - 15.0)
        
        # Run Local Search
        c_ls, score_ls = _local_search(c0, time.time() + time_left_for_ls)
        
        # Verify score
        score_ls = _compute_radius(c_ls)
        
        if score_ls > best_score:
            best_score = score_ls
            best_c = c_ls
            
    # 3. Polish the best candidate
    if best_c is not None:
        if time.time() < deadline - 5.0:
            c_polished = _polish_lbfgs(best_c, deadline - 5.0)
            score_polished = _compute_radius(c_polished)
            if score_polished > best_score:
                best_c = c_polished
                best_score = score_polished

    if best_c is None:
        return _build_initial_grid()
        
    return best_c
# EVOLVE-BLOCK-END
