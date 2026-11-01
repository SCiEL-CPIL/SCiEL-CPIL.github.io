"""
Strategy: Log-Sum-Exp soft-minimization with L-BFGS-B.
Instead of discrete constraint resolution, we minimize a smooth surrogate
function that approximates the negative of the minimum clearance.
This allows gradient-based global optimization to escape local minima
more effectively than the local push-pull heuristics.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N_CIRCLES = 200
HALF_SIDE = 0.5
EPS = 1e-8

def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N_CIRCLES:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-12):
        return 0.0

    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers), axis=None)
    if min_dist_to_wall <= 0:
        return 0.0

    tree = cKDTree(centers)
    # Use a generous radius for query_pairs to ensure we find all close pairs
    # The max possible radius is ~0.037, so 0.1 is safe for finding candidates
    pairs = tree.query_pairs(r=0.1)
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        # Vectorized distance calculation for pairs
        diffs = centers[list(pairs[:, 0])] - centers[list(pairs[:, 1])]
        dists = np.linalg.norm(diffs, axis=1)
        min_pair_dist = np.min(dists)

    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r

def lse_objective_and_grad(flat_centers, beta=50.0):
    """
    Objective: Minimize -soft_min(clearance)
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    We want to maximize min(clearance), so we minimize -min(clearance).
    
    clearance_i for pair (i,j) = dist(i,j)/2
    clearance_k for wall k = 0.5 - |c_k|
    
    We combine all these into one set of values V.
    We want to maximize min(V).
    Equivalent to minimizing -min(V).
    Approximated by -soft_min(V).
    
    Gradient of -soft_min(V) w.r.t centers.
    """
    centers = flat_centers.reshape(-1, 2)
    
    # 1. Compute all pair distances
    tree = cKDTree(centers)
    # We need a cutoff that captures all "relevant" pairs.
    # If current min clearance is r, pairs closer than 2*r + margin are relevant.
    # However, for the gradient to be correct, we technically need all pairs.
    # But for 200 circles, N^2 is 40k, which is manageable.
    # Let's use N^2 for accuracy in gradient, or KDTree with a large radius.
    # Given N=200, N^2/2 = 20k pairs. This is fast enough in numpy.
    
    # Compute pairwise distances efficiently
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dist = np.linalg.norm(diff, axis=-1)
    
    # To save memory/time, let's stick to KDTree but with a large radius
    # The max distance in the square is sqrt(2) ~ 1.41.
    # Pairs with dist > 2*r_max are irrelevant to the min.
    # r_max is roughly 0.04. So dist > 0.08 is irrelevant.
    # Let's use 0.1 as a safe upper bound for "relevant" pairs.
    # Wait, if the configuration is bad, r is small. If r is large, pairs are far.
    # Actually, if r is large, the min distance is large, so we only care about
    # the few pairs that are close.
    # If the configuration is bad (overlapping), r is small/negative.
    # The LSE method works best when we are near a valid packing.
    
    # Let's compute all pairs to be safe for the gradient, but filter by distance
    # to avoid numerical issues with exp(-beta * large_dist) -> 0.
    
    # Efficient pairwise distance matrix
    # Since N=200, we can compute the full matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Mask out self-distances
    np.fill_diagonal(dists, np.inf)
    
    # Get the values for the soft min
    # Values are dists/2 for pairs, and wall distances for walls.
    
    # Wall distances
    # d_x_left = 0.5 - x
    # d_x_right = x + 0.5
    # d_y_bottom = 0.5 - y
    # d_y_top = y + 0.5
    # min_wall_dist_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # Actually, the constraint is 0.5 - |x| >= r and 0.5 - |y| >= r.
    # So the clearance for wall x is 0.5 - |x|, and for wall y is 0.5 - |y|.
    
    wall_x = HALF_SIDE - np.abs(centers[:, 0])
    wall_y = HALF_SIDE - np.abs(centers[:, 1])
    
    # Combine all clearance values
    # Pair clearances: dists / 2 (upper triangle)
    iu = np.triu_indices(N_CIRCLES, k=1)
    pair_clearances = dists[iu] / 2.0
    
    # Wall clearances
    wall_clearances = np.concatenate([wall_x, wall_y])
    
    # All clearances
    all_clearances = np.concatenate([pair_clearances, wall_clearances])
    
    # Filter out very large clearances to avoid overflow/underflow in exp
    # If clearance > 1.0, exp(-50 * 1.0) is tiny.
    # We only care about the minimum, so large values don't contribute much.
    # But for gradient, we need to be careful.
    # Let's just compute the LSE on the whole vector.
    
    # To handle potential overflow in exp(-beta * v) if v is very negative (overlap),
    # we should shift. But typically we start with valid or near-valid configs.
    # If v is negative, exp(-beta * v) is huge.
    # Let's clamp v to a minimum of -0.1 to prevent explosion, or use a smaller beta initially.
    
    # Better approach: Standard LSE min
    # f = -1/beta * log( sum( exp(-beta * v_i) ) )
    # grad_f = -1/beta * ( sum( -beta * exp(-beta * v_i) * grad_v_i ) / sum( exp(-beta * v_i) ) )
    # grad_f = sum( exp(-beta * v_i) * grad_v_i ) / sum( exp(-beta * v_i) )
    
    # Let's compute weights w_i = exp(-beta * v_i) / sum(exp(-beta * v_j))
    # To avoid overflow, shift by min_v
    min_v = np.min(all_clearances)
    # If min_v is very negative, exp(-beta * (v_i - min_v)) is safe.
    # Let z_i = -beta * (v_i - min_v)
    # w_i = exp(z_i) / sum(exp(z))
    
    # However, if v_i is much larger than min_v, z_i is large negative -> 0.
    # This is good.
    
    # Gradient of v_i w.r.t centers:
    # For pair (i,j): v = dist/2. grad_dist = (c_i - c_j) / dist.
    # grad_v_i = 0.5 * (c_i - c_j) / dist
    # grad_v_j = 0.5 * (c_j - c_i) / dist
    # For wall x (left/right): v = 0.5 - |x|.
    # If x > 0: v = 0.5 - x, grad = -1
    # If x < 0: v = 0.5 + x, grad = 1
    # grad_v_x = -sign(x)
    # Similarly for y.
    
    # We need to accumulate gradients.
    grad_centers = np.zeros_like(centers)
    
    # Compute weights
    # z = -beta * (all_clearances - min_v)
    # If beta is large, z is very negative for non-min elements.
    # We can just compute exp(z) directly.
    z = -beta * (all_clearances - min_v)
    # Clip z to prevent underflow to 0 too early? No, underflow to 0 is fine.
    # But if z is positive (shouldn't happen since min_v is min), exp(z) > 1.
    # max(z) = 0.
    weights = np.exp(z)
    sum_weights = np.sum(weights)
    
    # If sum_weights is 0 (shouldn't happen), return 0
    if sum_weights == 0:
        return 0.0, np.zeros_like(flat_centers)
    
    weights /= sum_weights
    
    # Objective value
    obj = -min_v - (1.0 / beta) * np.log(sum_weights * np.exp(-beta * min_v)) 
    # Wait, the formula is:
    # soft_min = min_v - (1/beta) * log( sum(exp(-beta(v_i - min_v))) )
    # We want to minimize -soft_min.
    # So obj = -min_v + (1/beta) * log( sum(exp(z)) )
    # Since sum_weights was normalized, log(sum(exp(z))) = log(sum_weights_original)
    # Let's recompute sum_weights_original before normalization.
    # Actually, I normalized weights.
    # log_sum_exp = min_v * (-beta) + log(sum(exp(z))) ? No.
    # LSE(x) = max(x) + log(sum(exp(x - max(x))))
    # Here we have exp(-beta * v). Let u = -beta * v.
    # LSE(u) = max(u) + log(sum(exp(u - max(u))))
    # max(u) = -beta * min(v)
    # u - max(u) = -beta * v + beta * min(v) = -beta * (v - min(v)) = z
    # So log_sum_exp(-beta * v) = -beta * min_v + log(sum(exp(z)))
    # soft_min = -1/beta * log_sum_exp(-beta * v) = -1/beta * (-beta * min_v + log(sum(exp(z))))
    # soft_min = min_v - (1/beta) * log(sum(exp(z)))
    # We want to minimize -soft_min.
    obj = -min_v + (1.0 / beta) * np.log(np.sum(weights * sum_weights)) # sum(weights * sum_weights) = sum_weights_original
    # Wait, I normalized weights. So sum(weights) = 1.
    # sum_weights_original = sum_weights (before normalization)
    # I lost sum_weights_original. Let's keep it.
    
    # Let's restart the weight calculation to be safe.
    z = -beta * (all_clearances - min_v)
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z)
    if sum_exp_z == 0:
         return 0.0, np.zeros_like(flat_centers)
    
    obj = -min_v + (1.0 / beta) * np.log(sum_exp_z)
    
    # Weights w_i = exp(z_i) / sum_exp_z
    w_pairs = exp_z[:len(pair_clearances)] / sum_exp_z
    w_walls = exp_z[len(pair_clearances):] / sum_exp_z
    
    # Accumulate gradients for pairs
    # For each pair (i,j), contribution to grad_c_i is w_ij * 0.5 * (c_i - c_j) / dist_ij
    # Contribution to grad_c_j is w_ij * 0.5 * (c_j - c_i) / dist_ij
    
    # We have the full dists matrix.
    # We need to map pair_clearances back to indices.
    # iu[0], iu[1] are the indices.
    
    # grad_pair = w_pairs * 0.5 / dists[iu]
    # We need to add this to grad_centers[i] and subtract from grad_centers[j]
    
    # Vectorized accumulation is tricky. Let's loop or use bincount.
    # N=200, pairs ~20k. Loop is fine.
    
    # To speed up, only process pairs with significant weight?
    # If w_ij < 1e-6, skip.
    # This is an approximation but often good enough for optimization.
    # However, for correctness, let's do all.
    
    # Create a sparse-like accumulation
    # grad_centers[i] += sum_{j>i} w_ij * 0.5 * (c_i - c_j) / dist_ij
    # grad_centers[i] -= sum_{j<i} w_ji * 0.5 * (c_j - c_i) / dist_ji  (which is same term)
    
    # Let's just loop. 20k iterations in Python is ~10ms. Acceptable.
    
    for k in range(len(w_pairs)):
        if w_pairs[k] < 1e-7:
            continue
        i, j = iu[0][k], iu[1][k]
        dist = dists[i, j]
        if dist < 1e-9:
            continue
        factor = 0.5 * w_pairs[k] / dist
        diff_vec = centers[i] - centers[j]
        grad_centers[i] += factor * diff_vec
        grad_centers[j] -= factor * diff_vec
    
    # Accumulate gradients for walls
    # w_walls corresponds to [wall_x_0, ..., wall_x_N-1, wall_y_0, ..., wall_y_N-1]
    # For wall_x_i: v = 0.5 - |x_i|
    # grad_v = -sign(x_i)
    # Contribution to grad_centers[i, 0] += w_wall_x_i * (-sign(x_i))
    
    for k in range(N_CIRCLES):
        # Wall X
        w = w_walls[k]
        if w > 1e-7:
            x = centers[k, 0]
            if abs(x) > 1e-9:
                sign_x = 1.0 if x > 0 else -1.0
            else:
                sign_x = 0.0 # Subgradient, pick 0
            grad_centers[k, 0] += w * (-sign_x)
        
        # Wall Y
        w = w_walls[N_CIRCLES + k]
        if w > 1e-7:
            y = centers[k, 1]
            if abs(y) > 1e-9:
                sign_y = 1.0 if y > 0 else -1.0
            else:
                sign_y = 0.0
            grad_centers[k, 1] += w * (-sign_y)
    
    return obj, grad_centers.flatten()

def hexagonal_initialization(perturbation=0.0):
    """Create a hexagonal lattice initialization with optional perturbation."""
    best_points = None
    best_score = 0.0
    
    # Try a range of spacings
    for scale in [0.072, 0.074, 0.076, 0.078]:
        dx = scale
        dy = scale * np.sqrt(3) / 2
        
        x_vals = np.arange(-0.5, 0.5 + dx/2, dx)
        y_vals = np.arange(-0.5, 0.5 + dy/2, dy)
        
        grid_points = []
        for i, y in enumerate(y_vals):
            for x in x_vals:
                if i % 2 == 1:
                    x_offset = dx / 2
                else:
                    x_offset = 0
                px = x + x_offset
                if -0.5 <= px <= 0.5 and -0.5 <= y <= 0.5:
                    grid_points.append([px, y])
        
        if len(grid_points) < N_CIRCLES:
            continue
        
        grid_points = np.array(grid_points)
        
        # Add perturbation if requested
        if perturbation > 0:
            rng = np.random.RandomState(abs(hash(scale)) % (2**32))
            grid_points += rng.normal(0, perturbation, grid_points.shape)
            grid_points = np.clip(grid_points, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Sort by distance from center and take N closest
        dists = np.sum(grid_points ** 2, axis=1)
        idx = np.argsort(dists)[:N_CIRCLES]
        points = grid_points[idx]
        
        r = compute_radius(points)
        if r > best_score:
            best_score = r
            best_points = points.copy()
    
    if best_points is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N_CIRCLES)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N_CIRCLES]
    
    return best_points

def optimize_lse(centers, time_limit=10.0, beta_start=10.0, beta_end=100.0):
    """
    Optimize the configuration using L-BFGS-B on the LSE objective.
    """
    start_time = time.time()
    current = centers.copy()
    
    # Annieling beta? Or just use a fixed high beta?
    # High beta -> sharper min -> better approximation of true min.
    # Low beta -> smoother -> easier to optimize.
    # Let's try increasing beta over time.
    
    best_r = compute_radius(current)
    best_config = current.copy()
    
    # Simple strategy: Run L-BFGS-B with a few different betas
    betas = [10.0, 50.0, 200.0]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        # Define objective and gradient for this beta
        def obj_grad(x):
            return lse_objective_and_grad(x, beta=beta)
        
        try:
            res = minimize(
                obj_grad,
                current.flatten(),
                method='L-BFGS-B',
                jac=True, # Returns (f, g)
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        new_config = res.x.reshape(-1, 2)
        
        # Clip to square
        new_config = np.clip(new_config, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = compute_radius(new_config)
        
        if new_r > best_r:
            best_r = new_r
            best_config = new_config.copy()
            current = new_config.copy()
    
    return best_config, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_config = None
    best_r = 0.0
    
    # Multi-start strategy
    perturbations = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    
    for pert in perturbations:
        if time.time() > deadline - 15:
            break
        
        # Generate initial configuration
        config = hexagonal_initialization(perturbation=pert)
        
        # Optimize using LSE
        optimized, r = optimize_lse(config, time_limit=20.0)
        
        if r > best_r:
            best_r = r
            best_config = optimized
    
    # Final check
    if best_config is None or compute_radius(best_config) <= 0:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N_CIRCLES)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_config = grid[:N_CIRCLES]
    
    return best_config

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time() - t0}")
    print(f"Radius: {compute_radius(res)}")
