"""Rewritten program for packing 200 equal circles in a unit square.
Uses a direct approach: start from a hexagonal-like grid, then refine
with a physics-based repulsion optimizer that respects wall constraints.
"""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0

    # Distance to walls: min_i min_k (0.5 - |c_ik|)
    dw = np.minimum(np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
                    np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1]))
    wall_term = np.min(dw)

    # Pairwise distances
    n = len(centers)
    if n < 2:
        return float(wall_term)

    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0

    return min(float(wall_term), float(pair_term))


def _pdist_index_to_pair(k, n):
    """Convert pdist condensed index to (i, j) with i < j."""
    # k = i * n - i * (i + 1) // 2 + j - i - 1
    start = 0
    for i in range(n - 1):
        block_size = n - i - 1
        if k < start + block_size:
            j = i + 1 + (k - start)
            return i, j
        start += block_size
    return 0, 1  # Fallback


def _hexagonal_grid(n_circles):
    """
    Create a hexagonal packing arrangement for n_circles circles.
    Returns an array of shape (n_circles, 2).
    """
    # Hexagonal packing: rows offset by half a spacing
    # For N circles, we want to find the optimal row configuration.
    
    # Let's try to find m rows such that the total circles is close to N
    # In hexagonal packing, rows alternate between k and k-1 circles (or similar)
    # For a given row count m, the best arrangement is roughly:
    # rows alternate between ceil(N/m) and floor(N/m) circles
    
    best_centers = None
    best_r = 0.0
    
    # Try different numbers of rows
    for m in range(10, 25):  # Try rows from 10 to 24
        # Distribute N circles into m rows as evenly as possible
        base_per_row = N // m
        extra = N % m
        # First 'extra' rows get base_per_row + 1, rest get base_per_row
        
        row_counts = []
        for i in range(m):
            if i < extra:
                row_counts.append(base_per_row + 1)
            else:
                row_counts.append(base_per_row)
        
        # Compute spacing
        # Total width used: the widest row determines the x-spacing
        max_row_count = max(row_counts)
        
        # We need to fit all circles in the square [-0.5, 0.5]^2
        # Let's compute the required spacing
        # If the widest row has k circles, the x-distance between centers is 2r
        # The row height is sqrt(3)*r for hexagonal packing
        
        # Let's estimate r from the constraints:
        # Width: (max_row_count - 1) * 2r + 2r <= 1  =>  max_row_count * 2r <= 1 => r <= 1/(2*max_row_count)
        # Height: (m-1) * sqrt(3)*r + 2r <= 1  =>  r * ((m-1)*sqrt(3) + 2) <= 1 => r <= 1/((m-1)*sqrt(3) + 2)
        
        r_width = 1.0 / (2.0 * max_row_count)
        r_height = 1.0 / ((m - 1) * np.sqrt(3) + 2)
        r_est = min(r_width, r_height)
        
        if r_est <= best_r and best_centers is not None:
            continue
        
        # Now build the centers with this spacing
        # x-spacing = 2*r_est
        # y-spacing = sqrt(3)*r_est
        dx = 2.0 * r_est
        dy = np.sqrt(3) * r_est
        
        centers = []
        y_offset = -0.5 + r_est  # Start from bottom, first row center at y = -0.5 + r_est
        
        for row_idx, row_count in enumerate(row_counts):
            # Offset odd rows by dx/2
            x_offset = 0.0
            if row_idx % 2 == 1:
                x_offset = dx / 2.0
            
            # Center this row horizontally
            if row_count > 0:
                row_width = (row_count - 1) * dx
                start_x = -row_width / 2.0 + x_offset
                # But we also need to account for the offset; let's center the whole thing
                # Actually, let's just place them and then center the entire arrangement
                
            for col_idx in range(row_count):
                x = start_x + col_idx * dx
                y = y_offset + row_idx * dy
                centers.append([x, y])
        
        centers = np.array(centers)
        
        # Center the arrangement in the square
        centers[:, 0] -= np.mean(centers[:, 0])
        centers[:, 1] -= np.mean(centers[:, 1])
        
        # Clip to be within bounds
        centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers if best_centers is not None else np.random.uniform(-0.5, 0.5, (N, 2))


def _soft_min_objective_and_grad(x, beta=100.0):
    """
    Compute the negative soft-minimum radius and its gradient.
    Minimizing this function maximizes the minimum radius.
    """
    centers = x.reshape(-1, 2)
    
    # 1. Pairwise distances
    # d[i, j] = ||c_i - c_j||
    # We want to maximize min(d[i,j]/2, wall_dist[i])
    # This is equivalent to minimizing -min(...)
    # Using soft-min: -log-sum-exp(-beta * values) / beta
    
    # Compute all pairwise distances
    # To save memory and time, we can compute the gradient of the min distance
    # directly for the bottleneck pairs, but L-BFGS needs the full gradient.
    # A common trick is to use a smooth approximation of the minimum.
    
    # Let's compute the gradient of the min radius directly using the "min" subgradient
    # which is a weighted average of the gradients of the active constraints.
    
    # Wall distances
    dw = np.minimum(np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
                    np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1]))
    
    # Pairwise distances
    # We need the gradient of min_{i<j} dist(i,j) / 2
    # and min_i wall_dist(i)
    
    # To make it differentiable, we use a soft-min approach.
    # f(x) = -softmin(v_i) where v_i are the radii.
    # softmin(v) = -1/beta * log(sum(exp(-beta * v_i)))
    
    # However, computing exp(-beta * v) for all pairs (19900 pairs) is expensive.
    # Alternative: Use the gradient of the minimum directly.
    # grad(min(v)) = sum(w_i * grad(v_i)) where w_i are weights summing to 1,
    # concentrated on the minimum.
    
    # Let's compute the actual min radius and its gradient.
    # This is a standard technique in circle packing optimization.
    
    # 1. Find min wall distance
    min_wall_idx = np.argmin(dw)
    min_wall_val = dw[min_wall_idx]
    
    # 2. Find min pair distance
    # Using pdist is fast, but we need the gradient.
    # We can compute the gradient of the minimum pair distance.
    
    # Let's use a simpler approach: 
    # Objective: -min(min_pair/2, min_wall)
    # Gradient:
    # If min_pair/2 < min_wall:
    #   grad = -0.5 * grad(min_pair)
    # Else:
    #   grad = -grad(min_wall)
    
    # To smooth this, we can use a weighted average based on how close they are to the minimum.
    # Or just use the hard minimum gradient, which is non-smooth but L-BFGS can often handle it
    # if we provide a consistent subgradient.
    
    # Let's compute the gradient of the min pair distance.
    # d = pdist(centers)
    # min_idx = argmin(d)
    # i, j = index_to_pair(min_idx)
    # grad(min_pair) = 0.5 * ( (c_i - c_j) / d_ij * e_i - (c_i - c_j) / d_ij * e_j )
    # This is a subgradient.
    
    # To be more robust, we can sum the gradients of the K smallest pairs with weights.
    K = 5
    d = pdist(centers)
    # Get indices of K smallest distances
    # Note: pdist returns condensed form
    # We need to be careful with sorting large arrays, but N=200 is small.
    idxs = np.argsort(d)[:K]
    
    # Wall gradient
    # grad(min_wall) = -e_k where k is the axis of the closest wall for the closest point
    # Actually, grad(min_i min_k (0.5 - |c_ik|))
    # If c_ik is closest to right wall (0.5 - c_ik is min), grad is (1, 0) for that point?
    # Let's derive: f = min(0.5 - c_x, 0.5 + c_x, 0.5 - c_y, 0.5 + c_y)
    # If 0.5 - c_x is min, f = 0.5 - c_x, grad f = -1 * e_x = (-1, 0).
    # We want to maximize f, so we move in direction of grad f.
    # Objective is -f, so grad obj = -grad f = (1, 0).
    
    # Let's construct the gradient vector
    grad = np.zeros_like(centers)
    
    # Wall part
    # For the point with min wall distance, add gradient
    # Determine which wall is closest
    x, y = centers[min_wall_idx]
    dl = x + 0.5
    dr = 0.5 - x
    db = y + 0.5
    dt = 0.5 - y
    
    # Weights for walls (soft)
    # To smooth, we can use exp(-beta * dist) weights
    beta_wall = 50.0
    dists = np.array([dl, dr, db, dt])
    weights = np.exp(-beta_wall * (dists - dists.min()))
    weights /= np.sum(weights)
    
    # Gradients for each wall constraint (for maximizing radius)
    # If we move in +x, dist to left wall increases, dist to right decreases.
    # We want to increase min(dist).
    # grad(min_dist) for point i:
    # If left wall is min: grad = (-1, 0) ?? No.
    # f(x) = 0.5 + x. df/dx = 1.
    # f(x) = 0.5 - x. df/dx = -1.
    # So if we want to increase f, we move in direction of grad f.
    # grad obj = -grad f.
    
    # Let's just use the hard gradient for the minimum wall, or a soft mix.
    # Hard gradient for min wall:
    if np.argmin(dists) == 0: # Left
        g_wall = np.array([-1.0, 0.0])
    elif np.argmin(dists) == 1: # Right
        g_wall = np.array([1.0, 0.0])
    elif np.argmin(dists) == 2: # Bottom
        g_wall = np.array([0.0, -1.0])
    else: # Top
        g_wall = np.array([0.0, 1.0])
        
    # Actually, let's use the soft weights to be smoother.
    # grad_f_wall = weights[0] * (-1, 0) + weights[1] * (1, 0) + weights[2] * (0, -1) + weights[3] * (0, 1)
    g_wall = np.array([
        -weights[0] + weights[1],
        -weights[2] + weights[3]
    ])
    
    grad[min_wall_idx] += g_wall
    
    # Pair part
    # We want to maximize min_pair_dist / 2.
    # So we want to increase the distance between the closest pairs.
    # Gradient of dist(i,j) wrt c_i is (c_i - c_j) / d_ij.
    # Gradient of min_dist is sum of weighted gradients of the smallest distances.
    
    # To smooth, use weights based on exp(-beta * (d - min_d))
    beta_pair = 50.0
    min_d = d[idxs[0]]
    pair_dists = d[idxs]
    w_pairs = np.exp(-beta_pair * (pair_dists - min_d))
    w_pairs /= np.sum(w_pairs)
    
    for k in range(K):
        i, j = _pdist_index_to_pair(idxs[k], N)
        vec = centers[i] - centers[j]
        dist = np.linalg.norm(vec)
        if dist < 1e-9:
            continue
        unit_vec = vec / dist
        # We want to push i and j apart.
        # grad(dist) wrt c_i is unit_vec.
        # We want to maximize dist, so we move c_i in direction of unit_vec.
        # grad obj = -grad(dist).
        # So we subtract w * unit_vec from grad[i] and add w * unit_vec to grad[j]?
        # Wait.
        # Objective: -min(dist).
        # grad(-dist) = -grad(dist).
        # grad(dist) wrt c_i = unit_vec.
        # So grad obj wrt c_i = -unit_vec.
        # grad obj wrt c_j = +unit_vec.
        
        grad[i] -= w_pairs[k] * 0.5 * unit_vec
        grad[j] += w_pairs[k] * 0.5 * unit_vec

    # Combine wall and pair gradients
    # The objective is min(wall_term, pair_term).
    # We need to know which one is the bottleneck to weight them correctly.
    # If wall_term < pair_term, we should prioritize wall gradient.
    # If pair_term < wall_term, prioritize pair gradient.
    # Soft mixing:
    
    # Compute current terms
    min_pair_val = min_d / 2.0
    min_wall_val = min_wall_val
    
    beta_mix = 50.0
    diff = min_pair_val - min_wall_val
    # If diff > 0, pair is larger, wall is bottleneck.
    # If diff < 0, wall is larger, pair is bottleneck.
    
    # Weights for the two terms in the min
    # w_wall = exp(-beta * wall_term) / (exp(-beta * wall) + exp(-beta * pair))
    # This is essentially the soft-min weight.
    
    # But we already accumulated gradients.
    # Let's re-compute with proper weighting.
    
    # Reset grad
    grad = np.zeros_like(centers)
    
    # Recalculate wall gradient with weight w_wall
    # Recalculate pair gradient with weight w_pair
    
    # Let's use the soft-min of the two terms directly.
    # F = -softmin(wall_term, pair_term)
    # This is getting complicated.
    
    # Simpler: Just use the gradient of the actual minimum.
    # If wall_term < pair_term:
    #    grad = grad_wall
    # Else:
    #    grad = grad_pair
    
    # To make it smooth, we can use the soft-min weights for the two main terms.
    
    # Let's stick to the subgradient of the min.
    if min_wall_val < min_pair_val:
        # Wall is bottleneck
        # Use the soft-weighted wall gradient calculated above
        # We need to rescale g_wall to be the gradient of the min function?
        # The gradient of min(f, g) is grad(f) if f < g.
        # So we just use g_wall.
        grad[min_wall_idx] += g_wall
    else:
        # Pair is bottleneck
        # Use the soft-weighted pair gradients
        for k in range(K):
            i, j = _pdist_index_to_pair(idxs[k], N)
            vec = centers[i] - centers[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-9:
                continue
            unit_vec = vec / dist
            grad[i] -= w_pairs[k] * 0.5 * unit_vec
            grad[j] += w_pairs[k] * 0.5 * unit_vec
            
    # Objective is -radius
    # So we return -radius, grad
    radius = min(min_wall_val, min_pair_val)
    
    return -radius, grad

def _refine_lbfgs(centers, max_time=120.0):
    """
    Refine the packing using L-BFGS-B with a soft-min objective.
    """
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    start_time = time.time()
    
    def callback(xk):
        if time.time() - start_time > max_time - 5.0:
            return True
        return False
    
    try:
        res = minimize(_soft_min_objective_and_grad, x0, method='L-BFGS-B',
                       jac=True, bounds=bounds, callback=callback,
                       options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < _soft_min_objective_and_grad(x0)[0]:
            return res.x.reshape(-1, 2)
    except Exception:
        pass
        
    return centers


def construct():
    """Construct optimal circle packing."""
    # Start with hexagonal grid
    centers = _hexagonal_grid(N)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Refine with L-BFGS-B
    if _remaining_time() > 10.0:
        try:
            optimized = _refine_lbfgs(centers, max_time=_remaining_time() - 5.0)
            opt_radius = _compute_radius(optimized)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = optimized
        except Exception:
            pass
        
    # Optional: Short polishing with repulsion if time remains
    if _remaining_time() > 5.0:
        try:
            # Run a very short repulsion loop for final tweaks
            # Re-use _refine but with limited iterations? 
            # The current _refine is slow. Let's skip it for now and rely on L-BFGS.
            # Or implement a fast vectorized repulsion step.
            pass
        except Exception:
            pass
    
    # Ensure all centers are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
