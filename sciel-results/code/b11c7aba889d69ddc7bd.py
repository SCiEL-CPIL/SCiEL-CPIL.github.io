import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 300

# EVOLVE-BLOCK-START
def _soft_min(values, tau):
    """
    Compute the soft minimum of a set of values.
    values: array of shape (M,)
    tau: temperature parameter
    """
    # Shift values to avoid underflow/overflow in exp
    # We are minimizing -soft_min, so we want soft_min to be large.
    # soft_min(x) approx min(x)
    # We use the formula: -tau * log(sum(exp(-x/tau)))
    # To prevent underflow, we can subtract the min, but since we are 
    # dealing with distances which are positive, and tau is small, 
    # exp(-x/tau) can be very small.
    # Actually, standard soft min for minimization of max is:
    # max(x) ~ tau log(sum(exp(x/tau)))
    # min(x) ~ -tau log(sum(exp(-x/tau)))
    
    # For numerical stability:
    # Let y = -x/tau. Then soft_min = -tau * log(sum(exp(y)))
    # log(sum(exp(y))) = max(y) + log(sum(exp(y - max(y))))
    # max(y) = -min(x)/tau
    # So soft_min = -tau * (-min(x)/tau + log(sum(exp(-x/tau + min(x)/tau))))
    # soft_min = min(x) - tau * log(sum(exp(-(x - min(x))/tau)))
    
    x = np.asarray(values)
    min_val = np.min(x)
    # If all values are equal, sum(exp(0)) = M, log(M) is fine.
    # If one value is much smaller, exp(-(large)/tau) is 0.
    # This formulation is stable.
    
    # However, for L-BFGS-B, we just need the gradient.
    # The analytic gradient of soft_min w.r.t x_i is:
    # d/dx_i (-tau log(sum exp(-x_j/tau))) = 
    # -tau * (1/sum) * sum_j d/dx_i exp(-x_j/tau)
    # = -tau * (1/sum) * (exp(-x_i/tau) * (-1/tau))
    # = exp(-x_i/tau) / sum(exp(-x_j/tau))
    
    # This is essentially the softmax of -x.
    
    # We will implement the objective and gradient manually for speed.
    pass

def _objective_and_grad(x, tau):
    """
    Objective: -soft_min(distances)
    Gradient: -d(soft_min)/dx
    """
    points = x.reshape(-1, 2)
    n = points.shape[0]
    
    # Calculate pairwise distances
    # diff[i, j] = points[i] - points[j]
    # dists[i, j] = ||diff[i, j]||
    
    # To save memory and time, we can compute the objective in a loop or use vectorization carefully.
    # N=300, so 300x300 matrix is 90k elements. This is small.
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Boundary distances
    norms = np.sqrt(np.sum(points**2, axis=1))
    boundary_dists = 2.0 * (1.0 - norms)
    
    # Combine all distances into one array
    # We need the soft min of all pairwise dists (i<j) and boundary dists.
    # Note: dists[i,i] is 0, we must exclude them.
    
    # Mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairwise_dists = dists[mask]
    
    all_dists = np.concatenate([pairwise_dists, boundary_dists])
    
    # Calculate soft min
    # S = -tau * log(sum(exp(-all_dists/tau)))
    # For stability:
    min_d = np.min(all_dists)
    # exp terms
    # y = -(all_dists - min_d) / tau
    # S = min_d - tau * log(sum(exp(y)))
    
    y = -(all_dists - min_d) / tau
    # Clip y to avoid exp underflow/overflow issues, though y <= 0 here.
    # exp(y) is between 0 and 1.
    sum_exp = np.sum(np.exp(y))
    
    if sum_exp == 0:
        # Fallback, shouldn't happen
        soft_min_val = min_d
        grad_all = np.zeros_like(all_dists)
    else:
        soft_min_val = min_d - tau * np.log(sum_exp)
        # Gradient of soft_min w.r.t all_dists
        # g_i = exp(y_i) / sum_exp
        grad_all = np.exp(y) / sum_exp
        
    # We want to MINIMIZE -soft_min
    # So objective = -soft_min_val
    # Gradient of objective = -grad_all
    
    obj = -soft_min_val
    grad_all_neg = -grad_all
    
    # Now map grad_all back to x (points)
    grad_x = np.zeros_like(x)
    
    # 1. Boundary distances contribution
    # boundary_dists[k] = 2 * (1 - ||p_k||)
    # d(boundary_dists[k])/dp_k = -2 * p_k / ||p_k||
    # Contribution to grad_x from boundary:
    # sum_k grad_all_neg[boundary_idx_k] * d(boundary_dists[k])/dp_k
    
    # The indices for boundary dists are from len(pairwise_dists) to end
    boundary_grads = grad_all_neg[len(pairwise_dists):]
    
    for k in range(n):
        norm_k = norms[k]
        if norm_k > 1e-8:
            # d(2(1-norm))/dp = -2 * p / norm
            grad_x[2*k:2*k+2] += boundary_grads[k] * (-2.0 * points[k] / norm_k)
        else:
            # If point is at origin, gradient of norm is undefined, but we can assume 0 or small random
            pass
            
    # 2. Pairwise distances contribution
    # pairwise_dists[idx] = ||p_i - p_j||
    # d(||p_i - p_j||)/dp_i = (p_i - p_j) / ||p_i - p_j||
    # d(||p_i - p_j||)/dp_j = (p_j - p_i) / ||p_i - p_j|| = -(p_i - p_j) / ||p_i - p_j||
    
    # We need to iterate over i < j
    # This loop might be slow in Python. Let's vectorize if possible or use a pre-computed structure.
    # For N=300, N^2/2 = 45k iterations. In Python, this is ~0.05s. Acceptable.
    
    # Get indices i, j
    i_idx, j_idx = np.where(mask)
    
    for count in range(len(i_idx)):
        i = i_idx[count]
        j = j_idx[count]
        g = grad_all_neg[count]
        
        d = points[i] - points[j]
        dist = dists[i, j]
        
        if dist > 1e-8:
            # grad w.r.t p_i
            grad_x[2*i:2*i+2] += g * (d / dist)
            # grad w.r.t p_j
            grad_x[2*j:2*j+2] += g * (-d / dist)
            
    return obj, grad_x.flatten()

def _generate_hex_lattice():
    """
    Generate a hexagonal lattice of points inside the unit circle.
    """
    # Estimate radius
    # Area of disk = pi. Area per circle ~ pi/300.
    # r ~ 0.05
    # Spacing s = 2*r ~ 0.1
    
    # We want to generate enough points to pick 300.
    # Let's generate a grid larger than the unit circle.
    
    # Hex lattice parameters
    # Let spacing be s.
    # We don't know s exactly, but we can start with a guess and refine.
    # Or just generate a dense lattice and pick.
    
    # Let's try to estimate the number of points in a hex lattice of radius 1.
    # Area of hex lattice point ~ s^2 * sqrt(3)/2
    # N ~ pi / (s^2 * sqrt(3)/2)
    # 300 ~ pi / (s^2 * 0.866)
    # s^2 ~ pi / (300 * 0.866) ~ 0.012
    # s ~ 0.11
    
    s = 0.11
    
    # Generate points
    points = []
    # Range of i, j such that point is within 1.0
    # Max coord ~ 1.0
    # i, j range
    max_k = int(1.0 / s) + 2
    
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = i * s + (j % 2) * s * 0.5
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    # We need 300 points.
    # Sort by distance to center, keep closest 300.
    dists = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(dists)
    
    if len(indices) < N:
        # If we didn't generate enough, increase density or return what we have?
        # Better to ensure we generate enough.
        # If s is too large, we have fewer points.
        # Let's just return the first N if available, or pad with random if not (bad).
        # For s=0.11, area ~ 3.14, point area ~ 0.01, so ~300 points.
        # It should be close.
        pass
        
    return points[indices[:N]]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization
    points_init = _generate_hex_lattice()
    x0 = points_init.flatten()
    
    # 2. Optimization
    # We use L-BFGS-B
    # We will start with a larger tau and decrease it?
    # Or just use a fixed tau that works well.
    # Let's try tau = 0.01 first.
    
    tau = 0.01
    
    best_x = x0
    best_score = -np.inf
    
    # We might run multiple optimizations with different taus or restarts if time permits.
    # But L-BFGS-B is fast.
    
    # Define objective wrapper
    def obj(x):
        val, grad = _objective_and_grad(x, tau)
        return val, grad
        
    # L-BFGS-B in scipy.optimize.minimize expects (x) -> f
    # and (x) -> g
    # Or we can pass jac=True and return tuple?
    # No, scipy.minimize(method='L-BFGS-B') takes fun(x) and jac(x) separately.
    
    def fun(x):
        val, _ = _objective_and_grad(x, tau)
        return val
        
    def jac(x):
        _, grad = _objective_and_grad(x, tau)
        return grad
        
    try:
        # maxiter should be enough to converge.
        # N=300, 600 vars.
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-10})
        
        if res.success and res.fun < best_score:
            best_x = res.x
            best_score = res.fun
            
    except Exception as e:
        print(f"Optimization error: {e}", file=sys.stderr)
        
    # If time remains, we can try a second pass with smaller tau to sharpen the minimum.
    if time.time() < deadline - 20.0:
        tau2 = 0.001
        x_start = best_x
        
        def fun2(x):
            val, _ = _objective_and_grad(x, tau2)
            return val
            
        def jac2(x):
            _, grad = _objective_and_grad(x, tau2)
            return grad
            
        try:
            res2 = minimize(fun2, x_start, jac=jac2, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res2.success and res2.fun < best_score:
                best_x = res2.x
                best_score = res2.fun
        except Exception as e:
            print(f"Second optimization error: {e}", file=sys.stderr)

    # Final check and return
    points = best_x.reshape(-1, 2)
    
    # Ensure finite
    if not np.all(np.isfinite(points)):
        # Fallback to initial
        points = points_init
        
    return points
# EVOLVE-BLOCK-END
