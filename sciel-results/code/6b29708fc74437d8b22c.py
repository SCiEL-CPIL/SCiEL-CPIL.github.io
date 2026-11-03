# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def _softmax(z):
    """Stable softmax."""
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def _bary_to_cart(w):
    """
    Convert barycentric weights (w0, w1, w2) to Cartesian.
    w: (N, 3) array where each row sums to 1.
    P = w0*V0 + w1*V1 + w2*V2
    """
    # w[:, 0] * V0 + w[:, 1] * V1 + w[:, 2] * V2
    x = w[:, 1] * 1.0 + w[:, 2] * 0.5
    y = w[:, 2] * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _min_triangle_area_softmin(points, k):
    """
    Calculate the soft-min of triangle areas.
    points: (N, 2) array
    Returns: -softmin_value (to be minimized)
    """
    n = len(points)
    # Generate all unique triplets i < j < k
    # For N=11, this is 165 triplets.
    # Using meshgrid is simple and fast enough for N=11.
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    vec1 = p_j - p_i
    vec2 = p_k - p_i
    cross = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft-min approximation: -1/k * ln(sum(exp(-k * areas)))
    # We want to maximize the minimum area, so we minimize -min_area.
    # Soft-min of areas A is approx min(A).
    # Objective to minimize: -SoftMin(A)
    
    # Numerical stability:
    # exp(-k * areas) can underflow if areas are large or k is large.
    # But areas are small (unit triangle).
    # Let's use the log-sum-exp trick properly.
    # S = -1/k * logsumexp(-k * areas)
    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    
    x = -k * areas
    x_max = np.max(x)
    if x_max == -np.inf:
        return 0.0
    log_sum_exp = x_max + np.log(np.sum(np.exp(x - x_max)))
    soft_min_val = -1.0 / k * log_sum_exp
    return -soft_min_val

def _objective_softmax(z_flat, k=30.0):
    """
    Objective function for unconstrained softmax parameters.
    z_flat: (3*N,) array
    """
    z = z_flat.reshape(N, 3)
    w = _softmax(z)
    points = _bary_to_cart(w)
    return _min_triangle_area_softmin(points, k)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = -float('inf')
    best_points = None
    
    # Initializations
    # We can use random z values.
    # z ~ N(0, 1) is a good start.
    # Or uniform in simplex mapped to log-space?
    # Random z is fine because softmax is surjective.
    
    rng = np.random.RandomState(42)
    
    n_restarts = 20 # Adjust based on time
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initialize z
        # Scale matters for initial step size.
        z_init = rng.randn(3 * N) * 1.0
        
        try:
            res = minimize(
                _objective_softmax,
                z_init,
                args=(30.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                z_opt = res.x.reshape(N, 3)
                w_opt = _softmax(z_opt)
                points_opt = _bary_to_cart(w_opt)
                
                # Evaluate true min area for scoring
                # Re-calculate min area with hard min for final score comparison
                n = len(points_opt)
                i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
                mask = (i_idx < j_idx) & (j_idx < k_idx)
                i_flat = i_idx[mask]
                j_flat = j_idx[mask]
                k_flat = k_idx[mask]
                p_i = points_opt[i_flat]
                p_j = points_opt[j_flat]
                p_k = points_opt[k_flat]
                vec1 = p_j - p_i
                vec2 = p_k - p_i
                cross = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]
                areas = 0.5 * np.abs(cross)
                true_min_area = np.min(areas)
                
                if true_min_area > best_score:
                    best_score = true_min_area
                    best_points = points_opt
                    
        except Exception:
            pass

    if best_points is None:
        # Fallback: Centroid + random
        # Create a valid configuration
        w = np.ones((N, 3)) / 3.0
        # Perturb slightly to break symmetry
        w += rng.randn(N, 3) * 0.01
        w = w / np.sum(w, axis=1, keepdims=True)
        best_points = _bary_to_cart(w)

    return best_points
# EVOLVE-BLOCK-END
