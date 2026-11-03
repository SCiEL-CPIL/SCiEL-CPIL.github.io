import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _get_initial_structure():
    """
    Generate initial points based on the known optimal packing structure for N=50.
    Structure: 1 center, 6, 12, 18, 13 rings.
    """
    points = []
    
    # Center
    points.append([0.0, 0.0])
    
    # Ring 1: 6 circles
    r1 = 0.25
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for a in angles1:
        points.append([r1 * np.cos(a), r1 * np.sin(a)])
        
    # Ring 2: 12 circles
    r2 = 0.45
    angles2 = np.linspace(0, 2*np.pi, 12, endpoint=False) + np.pi/12 # Offset to nest
    for a in angles2:
        points.append([r2 * np.cos(a), r2 * np.sin(a)])
        
    # Ring 3: 18 circles
    r3 = 0.65
    angles3 = np.linspace(0, 2*np.pi, 18, endpoint=False)
    for a in angles3:
        points.append([r3 * np.cos(a), r3 * np.sin(a)])
        
    # Ring 4: 13 circles
    r4 = 0.85
    angles4 = np.linspace(0, 2*np.pi, 13, endpoint=False)
    for a in angles4:
        points.append([r4 * np.cos(a), r4 * np.sin(a)])
        
    return np.array(points)

def _compute_distances(points):
    """
    Compute all pairwise distances and distances to boundary.
    Returns:
    dists: array of shape (M,) containing all relevant distances.
    """
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    # Upper triangle indices to avoid duplicates and zeros
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[i_idx, j_idx])
    
    # Boundary distances
    radii = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - radii
    
    return np.concatenate([pair_dists, boundary_dists])

def _soft_min_value_and_grad(dists, beta):
    """
    Compute the soft min of dists and its gradient with respect to dists.
    soft_min = -1/beta * log(sum(exp(-beta * dists)))
    grad_dists = -1/beta * (sum(exp(-beta * dists) * (-beta) * 1) / sum(exp(-beta * dists)))
               = sum(exp(-beta * dists)) / sum(exp(-beta * dists))
    """
    y = -beta * dists
    # Shift for stability
    y_max = np.max(y)
    exp_y = np.exp(y - y_max)
    sum_exp = np.sum(exp_y)
    
    if sum_exp == 0:
        return -np.inf, np.zeros_like(dists)
        
    val = -(np.log(sum_exp) + y_max) / beta
    grad = exp_y / sum_exp
    
    return val, grad

def _objective_and_grad(x_flat, beta):
    """
    Objective function for minimization.
    We want to maximize the soft min of distances.
    So we minimize -soft_min.
    """
    points = x_flat.reshape(-1, 2)
    dists = _compute_distances(points)
    
    val, grad_dists = _soft_min_value_and_grad(dists, beta)
    
    # Objective is -val
    obj = -val
    
    # Gradient calculation
    # d(obj)/dx = - d(val)/dx = - sum_k (grad_dists[k] * d(dist_k)/dx)
    
    # 1. Boundary distances: d(1 - |c_i|)/dx = - c_i / |c_i|
    radii = np.linalg.norm(points, axis=1)
    # Avoid division by zero
    radii_safe = np.where(radii > 1e-10, radii, 1.0)
    grad_boundary = -points / radii_safe[:, np.newaxis]
    
    # 2. Pairwise distances: d(||c_i - c_j||)/dx_i = (c_i - c_j) / ||c_i - c_j||
    # We need to accumulate gradients for each point.
    grad_points = np.zeros_like(points)
    
    # Add boundary contributions
    # grad_dists contains pair_dists then boundary_dists
    n_pairs = len(dists) - N
    grad_pair = grad_dists[:n_pairs]
    grad_bound = grad_dists[n_pairs:]
    
    # Boundary gradient contribution
    grad_points += grad_bound[:, np.newaxis] * grad_boundary
    
    # Pairwise gradient contribution
    # We have indices i_idx, j_idx
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff = points[i_idx] - points[j_idx]
    dists_pair = np.linalg.norm(diff, axis=1)
    dists_pair_safe = np.where(dists_pair > 1e-10, dists_pair, 1.0)
    
    # Gradient of dist_ij w.r.t c_i is (c_i - c_j)/d_ij
    # Gradient of dist_ij w.r.t c_j is (c_j - c_i)/d_ij = - (c_i - c_j)/d_ij
    
    grad_diff = diff / dists_pair_safe[:, np.newaxis]
    
    # Accumulate
    # For each pair (i, j) with weight w:
    # grad[i] += w * grad_diff
    # grad[j] += w * (-grad_diff)
    
    np.add.at(grad_points, i_idx, grad_pair[:, np.newaxis] * grad_diff)
    np.add.at(grad_points, j_idx, -grad_pair[:, np.newaxis] * grad_diff)
    
    # Total gradient of objective w.r.t x_flat
    # obj = -val
    # grad_obj = - (grad_points)
    grad_flat = -grad_points.flatten()
    
    return obj, grad_flat

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    best_score = -np.inf
    best_points = None
    
    # Initial structure
    init_points = _get_initial_structure()
    
    # Perform optimization with increasing beta (annealing)
    betas = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    current_points = init_points.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        x0 = current_points.flatten()
        
        try:
            res = minimize(
                _objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, # We return (f, g)
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if res.success or res.status == 0:
                new_points = res.x.reshape(-1, 2)
                # Verify improvement
                dists = _compute_distances(new_points)
                current_min = np.min(dists)
                current_score = current_min / 2.0 # Radius is half min dist? No, radius is min(d_ij/2, 1-r_i).
                # Wait, the score is the radius r.
                # r = min( min_{i<j} d_ij/2, min_i (1 - |c_i|) )
                # My dists array contains d_ij and 1-|c_i|.
                # So min(dists) is the minimum of these values.
                # If min(dists) is a pairwise distance, r = min(dists)/2.
                # If min(dists) is a boundary distance, r = min(dists).
                # Actually, the objective function maximizes the "soft min" of these distances.
                # The true radius is determined by the tightest constraint.
                
                # Let's just calculate the true score for the new points
                true_score = _calculate_score(new_points)
                
                if true_score > best_score:
                    best_score = true_score
                    best_points = new_points
                    current_points = new_points
                    
        except Exception as e:
            # If optimization fails, keep current best
            pass
            
        # Check time
        if time.time() > deadline:
            break
            
    # If no optimization succeeded, use initial
    if best_points is None:
        best_points = init_points
        best_score = _calculate_score(best_points)
        
    return best_points

def _calculate_score(points):
    """Calculate the exact packing radius."""
    if len(points) != N:
        return -np.inf
    if not np.all(np.isfinite(points)):
        return -np.inf
        
    # Boundary
    radii = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - radii)
    
    # Pairwise
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

# Helper to ensure we don't crash if L-BFGS-B doesn't like the gradient format or similar
# The function _objective_and_grad returns (f, g). L-BFGS-B expects this if jac=True.

# One potential issue: L-BFGS-B might struggle if the initial guess is far from the local minimum 
# or if the "soft min" is too flat. The annealing scheme helps.
# Also, we should add a simple local search or Nelder-Mead fallback if L-BFGS-B stagnates.
# But for N=50, L-BFGS-B with a good start usually works well.

# Let's add a final polish step using Nelder-Mead on the best points found, 
# minimizing the negative of the true min distance (which is non-differentiable but NM handles it).
# This can squeeze out the last bit of precision.

def _polish(points, time_limit=10.0):
    start = time.time()
    def obj(x):
        p = x.reshape(-1, 2)
        return -_calculate_score(p)
        
    try:
        res = minimize(obj, points.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if time.time() - start < time_limit:
            return res.x.reshape(-1, 2)
    except:
        pass
    return points

# Integrate polish into construct
def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_score = -np.inf
    best_points = None
    
    init_points = _get_initial_structure()
    
    betas = [10.0, 20.0, 50.0, 100.0, 200.0]
    current_points = init_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 20.0: # Leave time for polish
            break
            
        x0 = current_points.flatten()
        
        try:
            res = minimize(
                _objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_points = res.x.reshape(-1, 2)
            true_score = _calculate_score(new_points)
            
            if true_score > best_score:
                best_score = true_score
                best_points = new_points
                current_points = new_points
        except:
            pass

    # Polish step
    if best_points is not None and time.time() < deadline - 5.0:
        polished = _polish(best_points, time_limit=10.0)
        polished_score = _calculate_score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    elif best_points is None:
        best_points = init_points
        best_score = _calculate_score(best_points)
        
    return best_points
