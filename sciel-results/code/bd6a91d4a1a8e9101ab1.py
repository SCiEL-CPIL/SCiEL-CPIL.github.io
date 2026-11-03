"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _get_radius(centers):
    """Compute the radius score: min of half min-distance and min boundary distance."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: centers must be inside [-0.5, 0.5]^2
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _bottleneck_info(centers):
    """
    Identify the bottleneck: either a pair of circles or a circle-boundary contact.
    Returns:
        type: 'pair' or 'boundary'
        indices: tuple of indices for 'pair', index for 'boundary'
        value: current constraint value (distance/2 or boundary margin)
    """
    # Boundary
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary_idx = np.argmin(boundary_margin)
    min_boundary_val = boundary_margin[min_boundary_idx]
    
    # Pairwise
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_idx = np.unravel_index(np.argmin(dists), dists.shape)
    min_dist_val = dists[min_dist_idx[0], min_dist_idx[1]] / 2.0
    
    if min_dist_val < min_boundary_val:
        return 'pair', (min_dist_idx[0], min_dist_idx[1]), min_dist_val
    else:
        return 'boundary', min_boundary_idx, min_boundary_val

def _soft_min_energy(centers, k=20):
    """
    Compute a smooth approximation of the negative minimum radius.
    We want to maximize r, so we minimize -r.
    Using soft-min: r_soft = -1/k * log(sum(exp(-k*r_i)))
    where r_i are the constraint values (distances/2 and boundary margins).
    """
    # Boundary constraints
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Pairwise constraints
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constr = dists / 2.0
    
    # Combine all constraints
    # To avoid memory explosion with 200x200 matrix in exp, we can be careful.
    # 200x200 is 40000 elements. exp(-k*val) is fine.
    
    # We only care about the minimum. A common trick is to just use the actual min 
    # for the local search direction, but for a smooth objective we use soft-min.
    # However, computing the full 200x200 exp matrix every step in a local optimizer 
    # (which calls this many times) might be slow.
    # Let's optimize: we only need the gradient. 
    # But scipy minimize requires a function.
    
    # For speed, let's use a "hard" min with a small jitter for the optimizer?
    # No, the prompt suggests Nelder-Mead on a smooth objective.
    # Let's stick to soft-min but maybe reduce k or use a different formulation.
    
    # Alternative: Just minimize the sum of squared violations of a target radius?
    # That's constrained optimization.
    
    # Let's try the soft-min with a moderate k.
    # To save time, we can subsample? No, we need exactness.
    
    # Actually, for N=200, N^2=40000. 
    # exp(-20 * 0.03) ~ exp(-0.6) ~ 0.5.
    # This is computationally feasible in numpy.
    
    # Combine boundary and pair constraints into a single array?
    # No, they have different shapes.
    
    # Let's compute log-sum-exp for pairs and boundaries separately?
    # r_min = min(r_pairs, r_bounds)
    # -log(sum(exp(-k*r))) is not simply the sum of two log-sum-exps.
    # log(sum(exp(-k*r_pairs)) + sum(exp(-k*r_bounds)))
    
    # Let's just concatenate.
    # Pair constraints: 200x200 matrix. Flatten it.
    # Boundary constraints: 200 vector.
    
    # Memory: 40200 floats. Fine.
    
    pair_flat = pair_constr.flatten()
    # Filter out inf (diagonal)
    pair_flat = pair_flat[np.isfinite(pair_flat)]
    
    all_constr = np.concatenate([pair_flat, boundary_margin])
    
    # Soft min
    # To avoid overflow/underflow in exp, shift by min
    min_c = np.min(all_constr)
    shifted = all_constr - min_c
    # exp(-k * shifted)
    # If k is large, most terms are 0 except the min.
    # log(sum(exp(-k*shifted)))
    
    # We want to maximize min_c, so minimize -min_c.
    # Objective: -min_c + (1/k) * log(sum(exp(-k*(all_constr - min_c))))
    # As k -> inf, this approaches -min_c.
    
    # For numerical stability:
    # log_sum_exp(-k * shifted)
    # max_val = 0 (since shifted >= 0, -k*shifted <= 0)
    # sum_exp = sum(exp(-k * shifted))
    # result = min_c - (1/k) * log(sum_exp)
    
    sum_exp = np.sum(np.exp(-k * shifted))
    obj_val = min_c - (1.0/k) * np.log(sum_exp)
    
    # We want to maximize obj_val, so return -obj_val
    return -obj_val

def _generate_hex_initial(n, jitter=0.0):
    """Generate a hexagonal lattice packing with optional jitter."""
    # Estimate rows/cols for square aspect ratio
    # Hex lattice: dx = 2r, dy = r*sqrt(3)
    # Area = 1. N * pi * r^2 <= 1.
    # Let's just fit a grid.
    
    best_points = None
    best_score = 0.0
    
    # Try different orientations and densities
    for angle_deg in [0, 15, 30, 45]:
        angle = np.radians(angle_deg)
        
        # Find appropriate rows/cols
        # For a square, we want roughly equal number of rows and cols?
        # Hex lattice aspect ratio is 1/sqrt(3) for the unit cell, but the packing
        # in a square needs adjustment.
        
        # Let's try to fit N points.
        # Rows * Cols >= N
        # Width = Cols * dx
        # Height = Rows * dy + dx/2 (approx)
        # dx = 2r, dy = r*sqrt(3)
        # Width = Cols * 2r
        # Height = Rows * r*sqrt(3)
        # To fit in 1x1:
        # Cols * 2r <= 1 - margin
        # Rows * r*sqrt(3) <= 1 - margin
        
        # Let's iterate over possible row counts
        for rows in range(10, 20):
            cols = int(np.ceil(n / rows))
            if rows * cols < n:
                continue
            
            # Calculate max r for this grid
            # r <= 1 / (2 * cols)
            # r <= 1 / (rows * sqrt(3))
            r_est = min(1.0 / (2.0 * cols), 1.0 / (rows * np.sqrt(3)))
            
            # Generate points
            points = []
            dx = 2.0 * r_est
            dy = r_est * np.sqrt(3)
            
            for r in range(rows):
                y = -0.5 + (r + 0.5) * dy
                offset = 0 if r % 2 == 0 else dx / 2
                for c in range(cols):
                    x = -0.5 + (c + 0.5) * dx + offset
                    # Check bounds before adding, or add and filter later
                    # We will rotate later, so we need to be careful.
                    # Let's generate in a larger box and then center/rotate/clip?
                    # No, generate in [-0.5, 0.5] assuming no rotation first.
                    if abs(x) <= 0.5 and abs(y) <= 0.5:
                        points.append([x, y])
            
            if len(points) < n:
                continue
                
            points = np.array(points[:n])
            
            # Rotate
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotated = np.zeros_like(points)
            rotated[:, 0] = points[:, 0] * cos_a - points[:, 1] * sin_a
            rotated[:, 1] = points[:, 0] * sin_a + points[:, 1] * cos_a
            
            # Check if inside
            if np.all(np.abs(rotated) <= 0.5):
                # Add jitter if requested
                if jitter > 0:
                    rotated += np.random.normal(0, jitter, rotated.shape)
                    rotated = np.clip(rotated, -0.5, 0.5)
                
                s = _get_radius(rotated)
                if s > best_score:
                    best_score = s
                    best_points = rotated.copy()
                    
    if best_points is None:
        # Fallback random
        np.random.seed(0)
        best_points = np.random.uniform(-0.45, 0.45, (n, 2))
        
    return best_points

def _local_search(centers, max_iters=50, time_limit=None, start_time=None):
    """
    Perform bottleneck-elimination local search.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 10.0
        
    best_centers = centers.copy()
    best_score = _get_radius(best_centers)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        b_type, b_idx, b_val = _bottleneck_info(best_centers)
        
        # Define a local objective function that focuses on the bottleneck
        # We want to increase the minimum of the constraints.
        # We can use a "soft-min" of the constraints, but weighted towards the current bottleneck.
        # Or simpler: Just minimize the soft-min energy of the whole system.
        # The soft-min will naturally be dominated by the smallest constraint.
        
        # To speed up convergence, we can restrict the search to the points involved in the bottleneck?
        # No, moving other points might help.
        # But for speed, let's try optimizing the whole system with Nelder-Mead.
        
        # Nelder-Mead on 400 dimensions is VERY slow.
        # We need a better strategy.
        
        # Strategy: Move only the points involved in the bottleneck pair, or the boundary point.
        # This reduces dimensionality to 2 or 4.
        
        if b_type == 'pair':
            i, j = b_idx
            # Optimize positions of i and j
            def local_obj(x):
                c_new = best_centers.copy()
                c_new[i] = x[0:2]
                c_new[j] = x[2:4]
                # Clip to square
                c_new = np.clip(c_new, -0.5, 0.5)
                # Calculate radius
                # We only care about the min radius, but we must ensure no other pair becomes smaller?
                # Actually, if we only move i and j, other pairs are unchanged.
                # The new min radius is min(
                #   dist(i,j)/2,
                #   min_{k != i,j} dist(i,k)/2, dist(j,k)/2,
                #   min_boundary
                # )
                # We can compute this efficiently.
                
                # Distances from i to all others
                di = np.sqrt(np.sum((c_new[i] - c_new)**2, axis=1))
                di[i] = np.inf
                min_di = np.min(di)
                
                # Distances from j to all others
                dj = np.sqrt(np.sum((c_new[j] - c_new)**2, axis=1))
                dj[j] = np.inf
                min_dj = np.min(dj)
                
                # Dist between i and j
                dij = np.linalg.norm(c_new[i] - c_new[j])
                
                # Boundary
                bi = 0.5 - np.max(np.abs(c_new[i]))
                bj = 0.5 - np.max(np.abs(c_new[j]))
                
                # Other boundaries unchanged, but let's assume they are not the bottleneck anymore
                # To be safe, we can just compute the full score? 
                # Full score is O(N^2). 40000 ops. 
                # Nelder-Mead calls objective ~100-1000 times.
                # 1000 * 40000 = 40M ops. Should be fine in < 1s.
                
                # Let's just use the full score for correctness and simplicity.
                # But wait, if we move i and j, they might push against k, making dist(i,k) small.
                # The full score handles this.
                
                return -_get_radius(c_new)
            
            x0 = np.concatenate([best_centers[i], best_centers[j]])
            
            # Nelder-Mead options
            options = {
                'maxiter': 100,
                'xatol': 1e-8,
                'fatol': 1e-10
            }
            
            res = minimize(local_obj, x0, method='Nelder-Mead', options=options)
            
            if res.fun < -best_score:
                # Improved
                c_new = best_centers.copy()
                c_new[i] = res.x[0:2]
                c_new[j] = res.x[2:4]
                c_new = np.clip(c_new, -0.5, 0.5)
                
                new_score = _get_radius(c_new)
                if new_score > best_score:
                    best_centers = c_new
                    best_score = new_score
                else:
                    break # No improvement found
            else:
                break # No improvement found
                
        else: # boundary
            i = b_idx
            def local_obj(x):
                c_new = best_centers.copy()
                c_new[i] = x[0:2]
                c_new = np.clip(c_new, -0.5, 0.5)
                return -_get_radius(c_new)
            
            x0 = best_centers[i]
            options = {
                'maxiter': 100,
                'xatol': 1e-8,
                'fatol': 1e-10
            }
            res = minimize(local_obj, x0, method='Nelder-Mead', options=options)
            
            if res.fun < -best_score:
                c_new = best_centers.copy()
                c_new[i] = res.x[0:2]
                c_new = np.clip(c_new, -0.5, 0.5)
                
                new_score = _get_radius(c_new)
                if new_score > best_score:
                    best_centers = c_new
                    best_score = new_score
                else:
                    break
            else:
                break
                
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Initialize with best hexagonal lattice
    np.random.seed(42)
    best_centers = _generate_hex_initial(N, jitter=0.001)
    best_score = _get_radius(best_centers)
    
    # 2. Local search with bottleneck elimination
    # We will run local search multiple times or for a long time
    # Since Nelder-Mead is slow, we limit iterations.
    
    # To handle the "stuck in local minimum" issue, we can add small random perturbations
    # and restart local search if it stalls.
    
    stall_count = 0
    last_score = best_score
    
    while time.time() < deadline - 5:
        centers, score = _local_search(best_centers, max_iters=20, time_limit=10, start_time=start_time)
        
        if score > best_score + 1e-12:
            best_centers = centers
            best_score = score
            stall_count = 0
        else:
            stall_count += 1
            
        if stall_count > 5:
            # Perturb and restart
            # Add small noise to the worst constrained points or random points
            b_type, b_idx, b_val = _bottleneck_info(best_centers)
            if b_type == 'pair':
                i, j = b_idx
                best_centers[i] += np.random.normal(0, 0.005, 2)
                best_centers[j] += np.random.normal(0, 0.005, 2)
            else:
                i = b_idx
                best_centers[i] += np.random.normal(0, 0.005, 2)
            
            best_centers = np.clip(best_centers, -0.5, 0.5)
            best_score = _get_radius(best_centers)
            stall_count = 0
            
    return best_centers
# EVOLVE-BLOCK-END
