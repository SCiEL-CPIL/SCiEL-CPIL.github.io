"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _get_hex_lattice():
    """Generate a hexagonal lattice of points inside the unit circle."""
    # Estimate spacing
    # Area of unit circle is pi. Area of hex cell is approx 2 * r^2 * sqrt(3)/2 = r^2 * sqrt(3).
    # N * r^2 * sqrt(3) <= pi  => r <= sqrt(pi / (N * sqrt(3)))
    # r is diameter/2? No, r is radius of circles. Spacing is 2r.
    # Let spacing = s. s^2 * sqrt(3)/2 * N <= pi.
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate grid
    k = int(1.0 / s) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - s / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        # Fallback to random if lattice fails (unlikely)
        return np.random.uniform(-1, 1, (N, 2))
    
    # Shuffle and take first N to ensure variety
    np.random.shuffle(points)
    return points[:N]

def _objective_and_grad(x, beta):
    """
    Objective: Soft-min of pairwise distances (divided by 2) and boundary distances.
    We minimize this value to maximize the minimum distance.
    f(x) = (1/beta) * ln( sum_{i<j} exp(-beta * d_ij / 2) + sum_i exp(-beta * (1 - ||x_i||)) )
    """
    p = x.reshape(N, 2)
    
    # Pairwise distances
    # d_ij = ||x_i - x_j||
    # Vectorized calculation
    # diff[i, j] = x_i - x_j
    # We only need i < j
    # Efficient way:
    # diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    # d = np.sqrt(np.sum(diff**2, axis=2))
    # This is O(N^2) memory and compute. N=300 -> 90k elements. Fine.
    
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    # Avoid division by zero in gradient
    d = np.sqrt(d_sq + 1e-12)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Terms for sum A
    # exp(-beta * d / 2)
    # We only sum over i < j
    d_pair = d[mask]
    term_pair = np.exp(-beta * d_pair / 2.0)
    sum_A = np.sum(term_pair)
    
    # Boundary terms
    r = np.sqrt(d_sq.diagonal() + 1e-12) # ||x_i||
    b = 1.0 - r
    term_bound = np.exp(-beta * b)
    sum_B = np.sum(term_bound)
    
    total_sum = sum_A + sum_B
    
    # Objective value
    val = (1.0 / beta) * np.log(total_sum)
    
    # Gradient
    # grad_f = (1 / (beta * total_sum)) * (grad_A + grad_B)
    
    # grad_A
    # d/dx_i sum_{i<j} exp(-beta d_ij / 2)
    # = sum_{j != i} exp(-beta d_ij / 2) * (-beta/2) * (x_i - x_j)/d_ij
    # Note: for i<j, term is associated with pair (i,j).
    # Let's compute forces.
    # F_i = sum_{j != i} (-beta/2) * exp(-beta d_ij / 2) * (x_i - x_j)/d_ij
    
    # We can compute this using the full matrix d and term_pair_expanded
    # term_pair_full[i,j] = exp(-beta d_ij / 2) if i!=j else 0
    # But we only calculated d_pair for i<j.
    # Let's reconstruct or compute directly.
    
    # Recalculate exp for all pairs for gradient (or reuse)
    # It's cheaper to compute forces directly from diff and d.
    
    # exp_term = np.exp(-beta * d / 2.0)
    # exp_term[np.diag_indices(N)] = 0.0
    
    # coeff = (-beta / 2.0) * exp_term
    # Force_i = sum_j coeff[i,j] * (x_i - x_j)/d_ij
    # Note: (x_i - x_j)/d_ij is unit vector from j to i.
    
    # Let's do it efficiently.
    # d is (N, N).
    # We need to avoid division by zero.
    # d_safe = np.where(d > 1e-8, d, 1.0)
    # unit_vec = diff / d_safe[:,:,np.newaxis]
    # unit_vec[np.diag_indices(N)] = 0
    
    # weights = np.exp(-beta * d / 2.0)
    # weights[np.diag_indices(N)] = 0
    
    # force = (-beta / 2.0) * np.sum(weights * unit_vec, axis=1)
    
    # grad_B
    # d/dx_i sum_k exp(-beta (1 - ||x_k||))
    # = sum_k exp(-beta (1 - ||x_k||)) * beta * x_k / ||x_k||
    # Only term i contributes to x_i.
    # grad_B_i = exp(-beta (1 - ||x_i||)) * beta * x_i / ||x_i||
    
    # r is ||x_i||
    # r_safe = np.where(r > 1e-8, r, 1.0)
    # unit_r = p / r_safe[:, np.newaxis]
    # unit_r[np.where(r < 1e-8)[0]] = 0 # Handle origin
    
    # grad_B = term_bound * beta * unit_r
    
    # Combine
    # grad = (1 / (beta * total_sum)) * (force + grad_B)
    
    # Let's implement the gradient calculation carefully.
    
    # 1. Pairwise forces
    # d is (N, N)
    d_safe = np.where(d > 1e-8, d, 1.0)
    unit_vec = diff / d_safe[:, :, np.newaxis]
    np.fill_diagonal(unit_vec, 0.0)
    
    weights = np.exp(-beta * d / 2.0)
    np.fill_diagonal(weights, 0.0)
    
    # force_i = sum_j (-beta/2) * weights[i,j] * unit_vec[i,j]
    force = (-beta / 2.0) * np.sum(weights * unit_vec, axis=1)
    
    # 2. Boundary forces
    r_safe = np.where(r > 1e-8, r, 1.0)
    unit_r = p / r_safe[:, np.newaxis]
    # If r is very small, gradient is undefined. Set to 0.
    small_r_mask = r < 1e-8
    unit_r[small_r_mask] = 0.0
    
    grad_B = term_bound * beta * unit_r
    
    # Total gradient
    grad = (1.0 / (beta * total_sum)) * (force + grad_B)
    
    return val, grad.flatten()

def construct():
    start_time = time.time()
    deadline = 135.0 # 15s margin from 150s limit
    
    # 1. Initialization
    points = _get_hex_lattice()
    
    # Add small noise to break symmetry
    noise = np.random.normal(0, 0.01, points.shape)
    points += noise
    
    # Ensure points are inside
    r = np.hypot(points[:, 0], points[:, 1])
    mask = r > 0.99
    points[mask] *= 0.99 / r[mask, np.newaxis]
    
    best_points = points.copy()
    best_score = -1.0
    
    # Calculate initial score
    def calc_score(p):
        d = np.sqrt(np.sum((p[:, np.newaxis, :] - p[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        r = np.hypot(p[:, 0], p[:, 1])
        min_bound = np.min(1.0 - r)
        return min(min_pair, min_bound)
    
    best_score = calc_score(points)
    
    # 2. Optimization
    betas = [20, 50, 100, 200]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        x0 = points.flatten()
        
        # Define objective function for minimize
        def obj(x):
            val, _ = _objective_and_grad(x, beta)
            return val
        
        def grad(x):
            _, g = _objective_and_grad(x, beta)
            return g
        
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res.success:
                new_points = res.x.reshape(N, 2)
                new_score = calc_score(new_points)
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    points = new_points # Continue from here
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
