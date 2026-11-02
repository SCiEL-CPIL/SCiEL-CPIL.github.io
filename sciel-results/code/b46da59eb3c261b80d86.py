import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all unique triples (i, j, k) with i < j < k
# There are C(14, 3) = 364 triangles
_i, _j, _k = np.indices((N, N, N))
_mask = (_i < _j) & (_j < _k)
_IDX_I = _i[_mask]
_IDX_J = _j[_mask]
_IDX_K = _k[_mask]

def _score_from_points(points):
    """
    Compute the score: min triangle area / convex hull area.
    points: (N, 2) array.
    """
    # Calculate areas of all triangles
    p_i = points[_IDX_I]
    p_j = points[_IDX_J]
    p_k = points[_IDX_K]
    
    # 2 * Area = |cross(P_j - P_i, P_k - P_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    min_area = np.min(areas)
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # If hull fails, points are likely collinear or degenerate
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def _build_symmetric_points(params):
    """
    Construct 14 points from 14 parameters (7 radii, 7 angles) 
    assuming central symmetry.
    
    We define 7 vertices P_0 ... P_6 in the first half (including one on axis).
    The other 7 are -P_0 ... -P_6.
    To ensure convexity, we order them by angle.
    
    Representation:
    params[0:7] are angles theta_0 ... theta_6 in [0, pi)
    params[7:14] are radii r_0 ... r_6 > 0
    
    We sort the points by angle to maintain convex order.
    """
    angles = params[0:7]
    radii = params[7:14]
    
    # Generate first 7 points
    # Ensure angles are sorted and within [0, pi)
    # We use a soft constraint or just rely on the optimizer to find a valid region?
    # Better: enforce sorting in the construction? No, that breaks gradients.
    # Instead, we define the points in a fixed angular order.
    # Let's define the angles to be strictly increasing.
    # To handle the "strictly increasing" constraint, we can parameterize 
    # the differences between angles as positive values.
    
    # Let's try a different parameterization:
    # Let delta_angles = params[0:7] be positive.
    # theta_0 = 0
    # theta_i = theta_{i-1} + delta_i
    # Sum of deltas must be <= pi (for the half circle).
    # Actually, for central symmetry, the points cover the full circle.
    # P_k and P_{k+7} are opposite.
    
    # Let's stick to the simpler: 7 angles in [0, pi) and 7 radii.
    # We will sort the resulting 14 points by their polar angle around the origin.
    # This ensures the hull is the polygon formed by these points in angular order.
    # Convexity is guaranteed if the origin is inside the polygon and the points are in angular order.
    # For a centrally symmetric convex polygon, the origin is the center.
    
    # Construct points
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    half_points = np.column_stack([x, y])
    
    # Full set of points
    points = np.vstack([half_points, -half_points])
    
    # Sort by angle to ensure convex hull order
    # Angle calculation
    all_angles = np.arctan2(points[:, 1], points[:, 0])
    # Normalize to [0, 2*pi)
    all_angles[all_angles < 0] += 2 * np.pi
    
    sort_idx = np.argsort(all_angles)
    points_sorted = points[sort_idx]
    
    return points_sorted

def _neg_score_params(params):
    """
    Objective function for optimization.
    params: 14-dim vector.
    """
    # Enforce positivity of radii
    # We use exp(radii) to ensure they are positive
    # But we need to be careful with scaling.
    # Let's say params[7:14] are log_radii.
    angles = params[0:7]
    log_radii = params[7:14]
    
    # Angles should be in [0, pi). We can wrap them or constrain them.
    # Wrapping might cause discontinuities. 
    # Let's assume the optimizer stays in a good region.
    # To be safe, we can map angles to [0, pi) using modulo? 
    # No, modulo is non-differentiable at boundaries.
    # We'll just let the optimizer work with raw angles and hope it finds a solution.
    # However, if angles are not sorted, the sorting step in _build_symmetric_points 
    # creates non-differentiable points.
    
    # Alternative: Parameterize angles such that they are always sorted.
    # Let d_i = exp(params[0:7]) > 0.
    # theta_0 = 0
    # theta_i = sum(d_0...d_{i-1})
    # This ensures theta_i > theta_{i-1}.
    # We also need to ensure the total spread is appropriate.
    # Since we have central symmetry, the points P_i and P_{i+7} are opposite.
    # The angles of the first 7 points should span roughly [0, pi].
    # If the sum of deltas is too large, the points might wrap around?
    # Actually, if we just take the 7 points and their negatives, and sort by angle,
    # it works regardless of the initial angular spread, as long as they are distinct.
    
    d_angles = np.exp(params[0:7])
    radii = np.exp(params[7:14])
    
    # Cumulative sum to get sorted angles
    # theta_0 = 0
    # theta_1 = d_0
    # theta_2 = d_0 + d_1
    # ...
    theta = np.zeros(7)
    theta[1:] = np.cumsum(d_angles[:-1])
    
    # Now we have 7 points with angles theta and radii radii.
    # Construct them
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    half_points = np.column_stack([x, y])
    
    # Full set
    points = np.vstack([half_points, -half_points])
    
    # Sort by angle
    all_angles = np.arctan2(points[:, 1], points[:, 0])
    all_angles[all_angles < 0] += 2 * np.pi
    sort_idx = np.argsort(all_angles)
    points_sorted = points[sort_idx]
    
    score = _score_from_points(points_sorted)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    best_score = 0.0
    best_points = None
    
    # Initial guesses for params (14 dims)
    # params[0:7] are log(deltas) for angles
    # params[7:14] are log(radii)
    
    rng = np.random.default_rng(42)
    
    initial_guesses = []
    
    # 1. Regular 14-gon approximation
    # In the symmetric parameterization, a regular 14-gon has equal radii and equal angular steps.
    # The 7 points in the first half are spaced by pi/7.
    # d_angles should be roughly pi/7.
    # log(pi/7) ~ log(0.448) ~ -0.8
    d_init = np.full(7, np.log(np.pi / 7))
    r_init = np.full(7, 0.0) # log(1) = 0
    
    initial_guesses.append(np.concatenate([d_init, r_init]))
    
    # 2. Perturbed regular polygon
    d_pert = d_init + rng.normal(0, 0.1, 7)
    r_pert = r_init + rng.normal(0, 0.1, 7)
    initial_guesses.append(np.concatenate([d_pert, r_pert]))
    
    # 3. Random
    d_rand = rng.uniform(-1, 1, 7)
    r_rand = rng.uniform(-1, 1, 7)
    initial_guesses.append(np.concatenate([d_rand, r_rand]))
    
    # 4. Another random
    d_rand2 = rng.uniform(-1, 1, 7)
    r_rand2 = rng.uniform(-1, 1, 7)
    initial_guesses.append(np.concatenate([d_rand2, r_rand2]))
    
    # 5. More randoms
    for _ in range(5):
        d_r = rng.uniform(-1, 1, 7)
        r_r = rng.uniform(-1, 1, 7)
        initial_guesses.append(np.concatenate([d_r, r_r]))
        
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 5:
            break
            
        try:
            # Use Powell or Nelder-Mead
            # Powell is derivative-free and handles non-smoothness (from sorting) better sometimes?
            # Nelder-Mead is also robust.
            res = minimize(_neg_score_params, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-9})
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                # Reconstruct points
                best_points = _build_symmetric_points(res.x)
                
        except Exception as e:
            pass
            
    if best_points is None:
        # Fallback to regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
