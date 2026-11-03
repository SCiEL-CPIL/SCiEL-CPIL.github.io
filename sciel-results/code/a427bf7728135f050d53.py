"""
Sphere packing optimization using Multi-Start L-BFGS-B on a Log-Sum-Exp surrogate.
Strategy:
1. Define a smooth surrogate objective that approximates the minimum radius (r).
   r = min( min_{i<j} d_ij / 2, min_i w_i )
   We maximize r, which is equivalent to minimizing -r.
   Since min is non-differentiable, we use the Log-Sum-Exp approximation:
   soft_min(x) = -1/beta * log( sum( exp(-beta * x_i) ) )
   This is differentiable everywhere and converges to the true min as beta -> infinity.
2. Use SciPy's L-BFGS-B to optimize this smooth objective from multiple diverse starting points.
3. The gradient allows the optimizer to resolve close pairs effectively by pushing all
   nearby points apart simultaneously based on their contribution to the soft-min.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
CUBE_HALF = 0.5
DEADLINE = 142.0  # Safety margin for 150s limit

def _soft_min_radius(pts, beta):
    """
    Compute the smooth surrogate for the minimum radius.
    Returns the estimated radius r and the gradient of -r (for minimization).
    """
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    
    # To be numerically stable with large beta, we can compute log-sum-exp carefully.
    # However, standard LSE: -1/beta * log( sum( exp(-beta * d) ) )
    # For large beta, exp(-beta * d) underflows for d > 0.
    # We can use the trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    
    # Pairwise terms
    # We exclude i=j by setting dist to inf (exp(-inf)=0)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # Wall terms
    # Distance to each of the 6 walls
    wall_dists = np.concatenate([
        CUBE_HALF - pts[:, 0],  # +x wall
        pts[:, 0] + CUBE_HALF,  # -x wall
        CUBE_HALF - pts[:, 1],  # +y wall
        pts[:, 1] + CUBE_HALF,  # -y wall
        CUBE_HALF - pts[:, 2],  # +z wall
        pts[:, 2] + CUBE_HALF   # -z wall
    ])
    
    # We want to minimize -soft_min_radius.
    # soft_min_radius = min( soft_min_pairs / 2, soft_min_walls )
    # It is better to combine all terms into a single LSE to get a single smooth value.
    # Let S be the set of all constraints: {d_ij/2 for i<j} U {w_i for i}.
    # We want r = min(S).
    # Surrogate R_beta = -1/beta * log( sum_{s in S} exp(-beta * s) )
    
    # Construct vector of all constraint values
    # Pairs: N*(N-1)/2 values
    # Walls: 3*N values (actually 6*N if we count each face separately, but the min over coordinates is sufficient for the 'inside cube' constraint if we use abs)
    # Using abs: w_i = min_k (0.5 - |x_ik|). 
    # If we use 6*N terms, the soft min will effectively pick the closest face.
    # Let's stick to 6*N terms for robustness.
    
    # Get upper triangle indices for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Wall distances
    # wall_vals = wall_dists.flatten() # Already flat
    
    # Combine
    all_vals = np.concatenate([pair_dists, wall_dists])
    
    # Calculate soft min
    # For numerical stability:
    min_val = np.min(all_vals)
    # If min_val is very large, exp(-beta * (val - min_val)) might still be problematic if beta is huge?
    # exp(-beta * val) = exp(-beta * min_val) * exp(-beta * (val - min_val))
    # log(sum(exp(-beta * val))) = -beta * min_val + log(sum(exp(-beta * (val - min_val))))
    # R_beta = min_val - 1/beta * log(sum(exp(-beta * (val - min_val))))
    
    # Shift values to be non-negative relative to min
    shifted = all_vals - min_val
    # exp(-beta * shifted) is in (0, 1]
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    # R_beta = min_val - (1/beta) * log(sum_exp)
    r_val = min_val - np.log(sum_exp) / beta
    
    # Gradient of -R_beta with respect to pts
    # d/dx (-R_beta) = d/dx ( (1/beta) * log(sum_exp) - min_val )
    # Note: min_val depends on x too! But in the LSE approximation, we treat the "soft min" as the function.
    # The gradient of the soft min function f(x) = -1/beta log(sum exp(-beta s_i(x))) is:
    # grad f = sum( p_i * grad s_i(x) ) where p_i = exp(-beta s_i) / sum(exp(-beta s_j))
    # So grad (-f) = - sum( p_i * grad s_i )
    
    # We need grad s_i for each term.
    # Instead of computing full gradient matrix (150x1375), we can accumulate.
    
    # p_i are the weights
    p = exp_terms / sum_exp
    
    # Gradient of pair distance d_ij / 2
    # s_pair = 0.5 * ||x_i - x_j||
    # grad_x_i s_pair = 0.5 * (x_i - x_j) / ||x_i - x_j||
    # grad_x_j s_pair = 0.5 * (x_j - x_i) / ||x_i - x_j||
    
    # Gradient of wall distance w_ik = 0.5 - |x_ik|
    # grad_x_ik w_ik = -sign(x_ik)
    
    grad = np.zeros_like(pts)
    
    # Pair contributions
    # p_pairs = p[:len(pair_dists)]
    p_pairs = p[:len(pair_dists)]
    
    # We have vectors i_idx, j_idx.
    # For each pair (i, j), weight w = p_pairs[k]
    # Contribution to grad[i]: w * 0.5 * (pts[i] - pts[j]) / dist_ij
    # Contribution to grad[j]: w * 0.5 * (pts[j] - pts[i]) / dist_ij
    
    # Vectorized accumulation using np.add.at or loops?
    # N=50, pairs=1225. A loop is fast enough in Python if simple, but let's try vectorized.
    
    # diff_vecs = pts[i_idx] - pts[j_idx] (shape 1225x3)
    # dists_vec = norm(diff_vecs, axis=1) (shape 1225)
    # unit_vecs = diff_vecs / dists_vec[:, None]
    # coeffs = 0.5 * p_pairs[:, None]
    
    diff_vecs = pts[i_idx] - pts[j_idx]
    # Avoid division by zero
    dists_vec = np.sqrt(np.sum(diff_vecs * diff_vecs, axis=1))
    dists_vec[dists_vec == 0] = 1e-10
    unit_vecs = diff_vecs / dists_vec[:, np.newaxis]
    
    # Weighted unit vectors
    weighted_units = unit_vecs * (0.5 * p_pairs[:, np.newaxis])
    
    # Sum contributions to i
    np.add.at(grad, i_idx, weighted_units)
    # Sum contributions to j (negative)
    np.add.at(grad, j_idx, -weighted_units)
    
    # Wall contributions
    p_walls = p[len(pair_dists):]
    # p_walls shape (3*N,) -> actually (6*N,)
    # Reshape to (N, 6)
    p_walls_reshaped = p_walls.reshape(N, 6)
    
    # The 6 terms for point i are:
    # 0: 0.5 - x_i  (grad -1)
    # 1: x_i + 0.5  (grad +1)
    # 2: 0.5 - y_i  (grad -1)
    # 3: y_i + 0.5  (grad +1)
    # 4: 0.5 - z_i  (grad -1)
    # 5: z_i + 0.5  (grad +1)
    
    # grad (-R) for walls:
    # dF/dx = - sum p_i grad s_i
    # For term 0 (0.5 - x): grad s = -1. Contribution to dF/dx = - p_0 * (-1) = p_0.
    # For term 1 (x + 0.5): grad s = +1. Contribution to dF/dx = - p_1 * (+1) = -p_1.
    # For term 2 (0.5 - y): grad s = -1. Contribution to dF/dy = p_2.
    # For term 3 (y + 0.5): grad s = +1. Contribution to dF/dy = -p_3.
    # For term 4 (0.5 - z): grad s = -1. Contribution to dF/dz = p_4.
    # For term 5 (z + 0.5): grad s = +1. Contribution to dF/dz = -p_5.
    
    # So:
    # grad_x += p_walls_reshaped[:, 0] - p_walls_reshaped[:, 1]
    # grad_y += p_walls_reshaped[:, 2] - p_walls_reshaped[:, 3]
    # grad_z += p_walls_reshaped[:, 4] - p_walls_reshaped[:, 5]
    
    grad[:, 0] += p_walls_reshaped[:, 0] - p_walls_reshaped[:, 1]
    grad[:, 1] += p_walls_reshaped[:, 2] - p_walls_reshaped[:, 3]
    grad[:, 2] += p_walls_reshaped[:, 4] - p_walls_reshaped[:, 5]
    
    # Now `grad` is dF/dx (gradient of the function we want to minimize).
    return -r_val, grad

def _objective_and_grad(pts_flat, beta):
    """Wrapper for L-BFGS-B."""
    pts = pts_flat.reshape(N, 3)
    val, grad = _soft_min_radius(pts, beta)
    return val, grad

def _score(pts):
    """
    Compute the true maximum common radius.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    min_wall = np.min(0.5 - np.abs(pts))
    return min(min_dist / 2.0, min_wall)

def _optimize_lbgfgs(start_pts, time_limit, seed=0):
    """
    Run L-BFGS-B optimization on the smooth surrogate.
    """
    start_time = time.time()
    x0 = start_pts.flatten()
    
    # Bounds for each coordinate
    bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
    
    # Adaptive beta strategy:
    # Start with moderate beta to find a good basin, increase beta to sharpen the minimum.
    betas = [20, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
    
    current_x = x0
    best_val = np.inf
    best_x = x0
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
            
        try:
            res = minimize(
                _objective_and_grad,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                jac=True, # We provide gradient
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                current_x = res.x
        except Exception:
            pass
            
    # Convert back to points
    best_pts = best_x.reshape(N, 3)
    return best_pts

def _generate_grid_start():
    """Generate a 50-point starting configuration from a jittered 4x4x4 grid."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.seed(42)
    grid += np.random.randn(*grid.shape) * 0.02
    grid = np.clip(grid, -0.49, 0.49)
    return grid[:N].copy()

def _generate_fcc_start():
    """Generate starting configuration from FCC lattice points."""
    points = []
    # FCC lattice vectors: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    # Unit cell size a=1.
    # We need to cover [-0.5, 0.5].
    # Generate points in a 3x3x3 supercell of unit cells?
    # Or just generate many points and filter.
    
    # Let's generate points on a grid of unit cells from -1 to 2 (size 4)
    # to ensure coverage.
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]])
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    
    points = np.array(points)
    
    # Add some jitter
    np.random.seed(123)
    points += np.random.randn(*points.shape) * 0.01
    points = np.clip(points, -0.49, 0.49)
    
    if len(points) > N:
        # Randomly select N points
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Pad with random points if needed (unlikely)
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
        
    return points[:N].copy()

def _generate_bcc_start():
    """Generate starting configuration from BCC lattice points."""
    points = []
    # BCC: (0,0,0) and (0.5, 0.5, 0.5)
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                for offset in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]])
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    
    points = np.array(points)
    
    # Add some jitter
    np.random.seed(456)
    points += np.random.randn(*points.shape) * 0.01
    points = np.clip(points, -0.49, 0.49)
    
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
        
    return points[:N].copy()

def _generate_random_start(seed):
    np.random.seed(seed)
    pts = np.random.uniform(-0.45, 0.45, size=(N, 3))
    return pts

def construct():
    start_time = time.time()
    
    # Generate diverse initial points
    starts = []
    starts.append(_generate_grid_start())
    starts.append(_generate_fcc_start())
    starts.append(_generate_bcc_start())
    
    # Add random starts
    for i in range(3):
        starts.append(_generate_random_start(100 + i))
        
    best_pts = None
    best_score = -np.inf
    
    # Score initial points to pick the best ones to start with
    scored_starts = [(s, _score(s)) for s in starts]
    scored_starts.sort(key=lambda x: x[1], reverse=True)
    
    # Allocate time budget
    # 142s total.
    # We have 6 starts. Let's run on the best 4.
    num_runs = 4
    time_per_run = (DEADLINE - (time.time() - start_time) - 5) / num_runs
    
    for i in range(num_runs):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        init_pts, init_score = scored_starts[i]
        
        # Run L-BFGS
        optimized_pts = _optimize_lbgfgs(init_pts, time_per_run, seed=i)
        
        # Evaluate true score
        current_score = _score(optimized_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = optimized_pts
            
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break
            
    # If we have time left, do a final polish on the best result with higher beta
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        remaining_time = DEADLINE - 15 - (time.time() - start_time)
        # Run a single high-beta L-BFGS on the best result found
        x0 = best_pts.flatten()
        bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
        
        # High beta for polishing
        betas = [500, 1000, 2000]
        current_x = x0
        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
            try:
                res = minimize(
                    _objective_and_grad,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.success or res.fun < _score(res.x.reshape(N,3)): # Approx check
                     current_x = res.x
            except:
                pass
        
        polished_pts = current_x.reshape(N, 3)
        polished_score = _score(polished_pts)
        
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score
            
    # Fallback
    if best_pts is None:
        best_pts = scored_starts[0][0]
        
    # Ensure validity
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    
    return best_pts
