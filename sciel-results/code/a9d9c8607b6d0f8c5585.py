"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
def _hex_init():
    """Initialize centers in a hexagonal lattice pattern."""
    # Hexagonal packing density is higher than square.
    # We want to fit N points.
    # Approximate radius for hex grid:
    # Area per point ~ 1/N.
    # Hex unit cell area = (sqrt(3)/2) * s^2 * 2 ? No.
    # Let's just generate a large hex grid and pick the best N.
    
    # Generate grid points
    # Spacing s.
    # Let's estimate s. Area = 1. N points.
    # s ~ sqrt(1/N / (sqrt(3)/2)) = sqrt(2/(N*sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate points
    points = []
    # Range of x and y
    # We need to cover [-0.5, 0.5]
    # Hex grid rows:
    # y = 0, s*sqrt(3)/2, s*sqrt(3), ...
    # x offsets alternate by s/2
    
    max_y = 0.5
    min_y = -0.5
    row_height = s * np.sqrt(3.0) / 2.0
    num_rows = int((max_y - min_y) / row_height) + 3
    # Center the grid roughly
    y_start = min_y - row_height
    
    for i in range(-num_rows, num_rows):
        y = y_start + i * row_height
        if y < min_y - s or y > max_y + s:
            continue
        x_offset = (i % 2) * (s / 2.0)
        x_start = min_y - s + x_offset # Just start early enough
        # Actually, let's just generate a dense grid and filter
        pass

    # Simpler: Generate a dense hex grid covering [-0.6, 0.6]
    pts = []
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    h = s * np.sqrt(3.0) / 2.0
    # Iterate over possible rows
    for i in range(-20, 20):
        y = i * h
        if y < -0.6 or y > 0.6:
            continue
        x_off = (i % 2) * (s / 2.0)
        for j in range(-20, 20):
            x = j * s + x_off
            if -0.6 <= x <= 0.6 and -0.6 <= y <= 0.6:
                pts.append((x, y))
    
    pts = np.array(pts)
    
    # We have more points than N. We need to select N points that maximize min distance.
    # Or just take the first N? Better to take the ones closest to center?
    # Or solve a small selection problem?
    # For initialization, taking a subset of the hex grid is fine.
    # Let's sort by distance from origin and take the N closest?
    # Or just take the first N? The grid is generated in a somewhat structured way.
    # Let's shuffle and take N? No, that's random.
    # Let's take the N points with the smallest index in the generated list?
    # The generation order is row by row.
    # If len(pts) > N, we can just take the first N.
    # But we might have cut off the corners.
    # Let's try to fit exactly N by adjusting s slightly?
    # For now, if len(pts) >= N, take the first N.
    # If len(pts) < N, we need to add more or use a different s.
    
    if len(pts) < N:
        # Fallback to square grid if hex generation failed
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # To ensure we get a good distribution, let's select N points from the generated set
    # that maximize the minimum distance. This is hard.
    # Simple heuristic: Take the N points closest to the center (0,0).
    # This ensures the center is filled.
    dists = np.sum(pts**2, axis=1)
    indices = np.argsort(dists)
    selected = pts[indices[:N]]
    return selected

def _objective_and_grad(x, k):
    """
    Compute the soft-min objective and its gradient.
    x: shape (2N,)
    k: sharpness parameter
    """
    c = x.reshape(N, 2)
    cx = c[:, 0]
    cy = c[:, 1]
    
    # 1. Pairwise distances
    # We need sum of exp(-k * d_ij) for i < j
    # And gradient contributions
    
    # Compute all pairwise squared distances
    # Diff matrix
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    d_sq = dx**2 + dy**2
    
    # Only upper triangle i < j
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Distances
    d = np.sqrt(d_sq)
    # Set diagonal to 0 (not used in sum due to mask)
    
    exp_terms = np.exp(-k * d)
    # Zero out non-pairs
    exp_terms = exp_terms * mask
    
    sum_pairs = np.sum(exp_terms)
    
    # Boundary terms
    # d_left = cx + 0.5
    # d_right = 0.5 - cx
    # d_bottom = cy + 0.5
    # d_top = 0.5 - cy
    
    d_left = cx + 0.5
    d_right = 0.5 - cx
    d_bottom = cy + 0.5
    d_top = 0.5 - cy
    
    exp_left = np.exp(-k * d_left)
    exp_right = np.exp(-k * d_right)
    exp_bottom = np.exp(-k * d_bottom)
    exp_top = np.exp(-k * d_top)
    
    sum_bounds = np.sum(exp_left) + np.sum(exp_right) + np.sum(exp_bottom) + np.sum(exp_top)
    
    total_sum = sum_pairs + sum_bounds
    
    if total_sum == 0:
        return 0.0, np.zeros_like(x)
        
    # Objective: (1/k) * ln(total_sum)
    obj = (1.0 / k) * np.log(total_sum)
    
    # Gradient
    # grad = -k * (sum_pairs_grad / total_sum + sum_bounds_grad / total_sum)
    # grad = -(k / total_sum) * (sum_pairs_grad + sum_bounds_grad)
    
    grad = np.zeros_like(x)
    
    # Pairwise Gradient
    # For each pair (i, j), term is exp(-k * d_ij)
    # d/dx_i exp(-k * d_ij) = exp(-k * d_ij) * (-k) * (x_i - x_j)/d_ij
    # Sum over j != i
    
    # We can compute this vectorized.
    # Let W_ij = exp_terms[i, j] / total_sum
    # grad_xi += -k * W_ij * (x_i - x_j) / d_ij
    # Note: d_ij = 0 if i=j, but mask is 0 there.
    
    # To avoid division by zero, replace 0 distances with 1 in denominator where mask is 0
    d_safe = np.where(d > 1e-8, d, 1.0)
    
    # Weight matrix
    W = exp_terms / total_sum
    
    # dx/dx_i = (x_i - x_j) / d_ij
    # We need sum over j of W_ij * (x_i - x_j) / d_ij
    # This is effectively a force sum.
    
    # grad_x contribution from pairs:
    # For x_i: sum_j W_ij * (x_i - x_j) / d_ij
    # For x_j: sum_i W_ij * (x_j - x_i) / d_ij = - sum_i W_ij * (x_i - x_j) / d_ij
    
    # Let's compute the vector F_i = sum_{j != i} W_ij * (x_i - x_j) / d_ij
    # We can do this by matrix multiplication?
    # F_i = sum_j W_ij * (x_i / d_ij - x_j / d_ij)
    # F_i = x_i * sum_j W_ij / d_ij - sum_j W_ij * x_j / d_ij
    
    # Let's compute 1/d_ij matrix
    inv_d = 1.0 / d_safe
    # Apply mask
    inv_d_masked = inv_d * mask
    
    # Sum_j W_ij / d_ij
    # W_ij = exp_terms[i, j]
    # Term_ij = exp_terms[i, j] * inv_d[i, j]
    term_mat = exp_terms * inv_d
    
    # Sum over j (axis 1)
    sum_term_x = np.sum(term_mat, axis=1) # Shape (N,)
    
    # Sum_j W_ij * x_j / d_ij
    # This is (term_mat @ cx)
    sum_term_xj = term_mat @ cx
    
    # F_x_i = cx * sum_term_x - sum_term_xj
    F_x = cx * sum_term_x - sum_term_xj
    
    # Similarly for y
    sum_term_y = np.sum(term_mat, axis=1)
    sum_term_yj = term_mat @ cy
    F_y = cy * sum_term_y - sum_term_yj
    
    # The gradient of the pair part for x_i is -k * F_x_i / total_sum ?
    # Wait.
    # grad_obj = (1/k) * (1/total_sum) * d(total_sum)/dx
    # d(total_sum)/dx_i = sum_j d(exp(-k d_ij))/dx_i
    # = sum_j exp(-k d_ij) * (-k) * (x_i - x_j)/d_ij
    # = -k * sum_j exp(-k d_ij) * (x_i - x_j)/d_ij
    # = -k * sum_j exp_terms[i, j] * (x_i - x_j)/d_ij
    # Note that W_ij = exp_terms[i, j] / total_sum
    # So sum_j exp_terms[i, j] * ... = total_sum * sum_j W_ij * ...
    # So d(total_sum)/dx_i = -k * total_sum * F_x_i
    # Then grad_obj_x_i = (1/k) * (1/total_sum) * (-k * total_sum * F_x_i) = -F_x_i
    
    # So the gradient of the pair part is simply -F_x and -F_y.
    grad_x = -F_x
    grad_y = -F_y
    
    # Boundary Gradient
    # d/dx_i exp(-k (x_i + 0.5)) = -k exp(-k (x_i + 0.5))
    # d/dx_i exp(-k (0.5 - x_i)) = k exp(-k (0.5 - x_i))
    # Sum these up.
    
    # grad_x_bound_i = (1/k) * (1/total_sum) * [ -k exp_left_i + k exp_right_i ]
    # = (1/total_sum) * [ -exp_left_i + exp_right_i ]
    
    grad_x += (-exp_left + exp_right) / total_sum
    grad_y += (-exp_bottom + exp_top) / total_sum
    
    # Combine
    grad[:N] = grad_x
    grad[N:] = grad_y
    
    return obj, grad

def construct():
    start_time = time.time()
    timeout = 145.0 # Leave 5s margin
    
    # Initial guess
    x0 = _hex_init()
    x = x0.flatten()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_x = x.copy()
    best_score = -np.inf
    
    def _exact_score(x_flat):
        c = x_flat.reshape(N, 2)
        # Pairwise min
        d_sq = np.sum((c[:, np.newaxis] - c[np.newaxis, :])**2, axis=2)
        np.fill_diagonal(d_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(d_sq))
        
        # Boundary min
        dists = np.minimum(np.minimum(c[:, 0] + 0.5, 0.5 - c[:, 0]),
                           np.minimum(c[:, 1] + 0.5, 0.5 - c[:, 1]))
        min_bound_dist = np.min(dists)
        
        return np.min([min_pair_dist, min_bound_dist]) / 2.0 # Radius is half distance?
        # Task: radius r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|) )
        # Wait, the formula in task description:
        # r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|) )
        # My _exact_score returns this value.
        
    # Update best score
    current_score = _exact_score(x)
    if current_score > best_score:
        best_score = current_score
        best_x = x.copy()
        
    # Annealing schedule for k
    # Start with small k, increase.
    k_values = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for k in k_values:
        if time.time() - start_time > timeout:
            break
            
        # Run L-BFGS-B
        # maxiter should be reasonable
        # We have 400 variables.
        res = minimize(
            _objective_and_grad,
            x,
            args=(k,),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
        )
        
        x = res.x
        
        # Update best score
        current_score = _exact_score(x)
        if current_score > best_score:
            best_score = current_score
            best_x = x.copy()
            
        # Check time
        if time.time() - start_time > timeout:
            break
            
    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
