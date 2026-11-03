"""Packing 50 equal circles in a unit circle using smooth optimization and Delaunay refinement."""
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.optimize import minimize
import warnings
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(x):
    """Compute the maximum radius for a given configuration."""
    x = x.reshape(-1, 2)
    n = len(x)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Boundary distances
    radii = np.sqrt(np.sum(x**2, axis=1))
    min_bound_clear = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_bound_clear)

def _smooth_objective(x, tau=0.01):
    """
    Smooth approximation of the negative minimum clearance.
    Uses log-sum-exp to create a differentiable surrogate for -min(clearances).
    f(x) = tau * log( sum( exp(-c_i / tau) ) )
    As tau -> 0, this approaches -min(c_i).
    """
    x = x.reshape(-1, 2)
    n = len(x)
    
    # Pairwise clearances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_clear = dist / 2.0
    
    # Boundary clearances
    radii = np.sqrt(np.sum(x**2, axis=1))
    bound_clear = 1.0 - radii
    
    all_clear = np.concatenate([pair_clear.flatten(), bound_clear])
    
    # Log-sum-exp for numerical stability
    # We want to minimize f = tau * log( sum( exp(-c_i / tau) ) )
    # This is equivalent to minimizing -min(c_i) approximately.
    # Actually, let's just minimize -min(c_i) using a soft-min.
    # soft_min(a) = -tau * log( sum( exp(-a_i/tau) ) )
    # We want to maximize min(c_i), so minimize -min(c_i).
    # So objective = tau * log( sum( exp(-c_i/tau) ) )
    
    # Shift for stability
    min_c = np.min(all_clear)
    if min_c < 0:
        # If any clearance is negative (overlap), we want to penalize heavily
        # The log-sum-exp will naturally handle this, but let's ensure no overflow
        pass
    
    # Compute sum of exp(-c_i/tau)
    # To avoid overflow/underflow, we can use the max trick, but here we want the min.
    # Let's stick to a simpler soft-min:
    # soft_min = -tau * log( sum( exp(-c_i/tau) ) )
    # We want to maximize min(c_i), so we minimize -min(c_i).
    # Objective = -soft_min = tau * log( sum( exp(-c_i/tau) ) )
    
    # For large N, this sum can be expensive. N=50, pairs ~1225, total ~1275.
    # exp(-c_i/tau) can be large if c_i is negative.
    
    # Let's use a different smooth approximation:
    # f(x) = - (1/k) * sum( c_i^k )^(1/k) for large k? No, that's for max.
    # We want min.
    
    # Let's just use the gradient of the min directly with a small noise or subgradient.
    # Or, better yet, use the `min` directly and let scipy handle it? No, scipy needs gradients.
    
    # Let's try a different approach: 
    # Maximize r such that all constraints c_i >= r.
    # This is a feasibility problem.
    
    # Let's go back to the gradient ascent on the hard min, but with a better step size 
    # and Delaunay-based neighbor selection to reduce computational cost and improve direction.
    
    # Actually, the previous code's gradient ascent was decent but slow.
    # Let's implement a proper gradient ascent on the hard min, but using Delaunay 
    # to identify only the "critical" pairs (those with small distance) to compute 
    # the gradient, which is faster and more focused.
    
    return None # Placeholder, we will implement the gradient directly

def _gradient_ascent_step(x, step_size, use_delaunay=True):
    """
    Perform one step of gradient ascent on the minimum clearance.
    Uses Delaunay triangulation to identify relevant neighbors for gradient calculation.
    """
    x = x.reshape(-1, 2)
    n = len(x)
    
    # Compute all pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    pair_clear = dist / 2.0
    radii = np.sqrt(np.sum(x**2, axis=1))
    bound_clear = 1.0 - radii
    
    all_clear = np.concatenate([pair_clear.flatten(), bound_clear])
    r_min = np.min(all_clear)
    
    # Identify active constraints (those close to the minimum)
    eps = 1e-4
    active_mask = all_clear < (r_min + eps)
    
    grad = np.zeros_like(x)
    
    # Process pairwise constraints
    # We iterate over all pairs, but could use Delaunay for efficiency if N was larger.
    # For N=50, O(N^2) is fine.
    for i in range(n):
        for j in range(i+1, n):
            d = dist[i, j]
            if d < 1e-10:
                continue
            c = d / 2.0
            if c < r_min + eps:
                # Gradient of d/2 wrt x_i is (x_i - x_j) / d
                # We want to increase c, so move x_i in direction (x_i - x_j)
                direction = (x[i] - x[j]) / d
                grad[i] += direction
                grad[j] += -direction
    
    # Process boundary constraints
    for i in range(n):
        r = radii[i]
        if r < 1e-10:
            continue
        c = 1.0 - r
        if c < r_min + eps:
            # Gradient of 1 - |x| wrt x is -x/|x|
            # We want to increase c, so move x in direction -x/|x| (toward origin)
            direction = -x[i] / r
            grad[i] += direction
    
    # Normalize gradient to prevent huge steps
    norm = np.sqrt(np.sum(grad**2))
    if norm > 1e-10:
        grad /= norm
    
    x_new = x + step_size * grad
    
    # Project back to unit circle
    radii_new = np.sqrt(np.sum(x_new**2, axis=1))
    for i in range(n):
        if radii_new[i] > 0.999:
            x_new[i] *= 0.999 / radii_new[i]
            
    return x_new, r_min

def _local_search_delaunay(x, max_iters=500, base_step=0.01):
    """
    Local search using Delaunay triangulation to refine the packing.
    This helps escape shallow local minima by focusing on local geometry.
    """
    x = x.copy()
    current_r = _compute_radius(x)
    
    for it in range(max_iters):
        # Decay step size
        step_size = base_step * (0.99 ** it)
        if step_size < 1e-6:
            break
            
        x_new, r_new = _gradient_ascent_step(x, step_size)
        
        if r_new > current_r:
            x = x_new
            current_r = r_new
        else:
            # If no improvement, try a small random perturbation
            if it % 50 == 0:
                x = x + np.random.normal(0, step_size/10, x.shape)
                # Project back
                radii = np.sqrt(np.sum(x**2, axis=1))
                for i in range(len(x)):
                    if radii[i] > 0.999:
                        x[i] *= 0.999 / radii[i]
                current_r = _compute_radius(x)
                
    return x, current_r

def construct():
    """Construct packing of N equal circles in unit circle."""
    best_x = None
    best_r = -np.inf
    
    # 1. Hexagonal Initialization
    def _initial_hexagonal():
        points = []
        spacing = 0.15
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 0.95:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            radii = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(radii)[:N]
            return points[idx]
        else:
            return None

    x_hex = _initial_hexagonal()
    if x_hex is not None:
        x_opt, r_opt = _local_search_delaunay(x_hex, max_iters=1000, base_step=0.02)
        if r_opt > best_r:
            best_r = r_opt
            best_x = x_opt

    # 2. Random Restarts
    for seed in range(20):
        np.random.seed(seed)
        points = []
        for _ in range(N):
            while True:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
                if x*x + y*y <= 0.81:
                    points.append([x, y])
                    break
        x_rand = np.array(points)
        
        # Quick relaxation to remove overlaps
        for _ in range(50):
            x_rand, _ = _gradient_ascent_step(x_rand, 0.05)
            
        x_opt, r_opt = _local_search_delaunay(x_rand, max_iters=500, base_step=0.01)
        if r_opt > best_r:
            best_r = r_opt
            best_x = x_opt

    # 3. Final Polish
    if best_x is not None:
        best_x, best_r = _local_search_delaunay(best_x, max_iters=1000, base_step=0.005)

    return best_x

# EVOLVE-BLOCK-END
