"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

def _soft_min_objective(flat_points, beta=100.0):
    """
    Objective function for minimization.
    We want to maximize r = min(min_pair_dist/2, min_boundary_dist).
    This is equivalent to minimizing -r.
    We use a log-sum-exp approximation for the minimum to make it differentiable.
    
    -log(sum(exp(-beta * x))) / beta approximates min(x).
    So we minimize: (1/beta) * log( sum(exp(-beta * (d_ij/2))) + sum(exp(-beta * (1 - |c_i|))) )
    """
    n = len(flat_points) // 2
    points = flat_points.reshape(n, 2)
    
    # Compute pairwise distances
    # For 300 points, pdist is efficient enough
    d = pdist(points)
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms
    
    # Combine all constraints into one vector
    # Pair constraints: d / 2
    # Boundary constraints: 1 - norm
    all_vals = np.concatenate([d / 2.0, boundary_dists])
    
    # Numerical stability: subtract max to avoid underflow/overflow in exp
    # We want min(all_vals).
    # log-sum-exp(-beta * vals) = -beta * min(vals) + log(sum(exp(-beta*(vals - min))))
    # Actually, standard soft min: -1/beta * log(sum(exp(-beta * vals)))
    
    # To be safe with large beta, we can just compute directly if values are reasonable.
    # Values are distances, typically O(0.01 - 1.0).
    # exp(-100 * 0.05) = exp(-5) ~ 0.006.
    # exp(-100 * 1.0) = exp(-100) ~ 3e-44.
    # Sum will be dominated by the minimums.
    
    # Use np.logaddexp.reduce for stability if needed, but simple sum is often fine for beta=100
    # However, to be strictly safe:
    max_val = np.max(all_vals)
    # Wait, we are minimizing -soft_min.
    # soft_min = -1/beta * log(sum(exp(-beta * vals)))
    # obj = -soft_min = 1/beta * log(sum(exp(-beta * vals)))
    
    # Let's compute the log-sum-exp part
    # To avoid underflow in exp(-beta * vals), we can factor out the minimum?
    # No, the minimum term is the largest in exp(-beta * vals) because it's the least negative?
    # If vals = [0.1, 0.2], exp(-10) vs exp(-20). 0.1 is min. exp(-10) is larger.
    # So the sum is dominated by the minimums.
    
    # Let's just use the direct formula, it's stable enough for beta=100 and vals in [0,1]
    try:
        log_sum = np.log(np.sum(np.exp(-beta * all_vals)))
        obj = log_sum / beta
    except:
        return 1e10
        
    # We want to minimize obj.
    # Note: obj is negative of the soft min?
    # soft_min = -1/beta * log_sum.
    # We want to MAXIMIZE soft_min.
    # So we MINIMIZE -soft_min = 1/beta * log_sum.
    # Yes, obj = log_sum / beta.
    
    return obj

def _soft_min_gradient(flat_points, beta=100.0):
    """
    Analytical gradient of the soft min objective.
    obj = 1/beta * log(sum(exp(-beta * vals)))
    d(obj)/d(point_k) = sum_j ( d(obj)/d(val_j) * d(val_j)/d(point_k) )
    
    Let S = sum(exp(-beta * vals))
    d(obj)/d(val_j) = 1/beta * (1/S) * (-beta * exp(-beta * val_j))
                    = - exp(-beta * val_j) / S
                    = - p_j
    where p_j is the softmax probability.
    
    For pair (i, j) with distance d_ij:
    val_j = d_ij / 2
    d(val_j)/d(c_i) = (c_i - c_j) / (2 * d_ij)
    d(val_j)/d(c_j) = (c_j - c_i) / (2 * d_ij) = - (c_i - c_j) / (2 * d_ij)
    
    For boundary i:
    val_i = 1 - |c_i|
    d(val_i)/d(c_i) = - c_i / |c_i|
    
    """
    n = len(flat_points) // 2
    points = flat_points.reshape(n, 2)
    
    # Compute pairwise distances
    d = pdist(points)
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms
    
    all_vals = np.concatenate([d / 2.0, boundary_dists])
    
    # Compute weights p_j = exp(-beta * val_j) / S
    # For stability
    max_val = np.max(all_vals)
    exp_vals = np.exp(-beta * (all_vals - max_val))
    S = np.sum(exp_vals)
    p = exp_vals / S
    
    grad = np.zeros_like(flat_points)
    points_grad = grad.reshape(n, 2)
    
    # Pair contributions
    # p[0 : n*(n-1)/2] correspond to pairs
    n_pairs = n * (n - 1) // 2
    pair_probs = p[:n_pairs]
    
    # We need to map pair indices to (i, j)
    # pdist order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
    # We can iterate, but for 300 points, 44850 pairs.
    # Vectorized approach is tricky with pdist output.
    # Let's use a loop for the gradient accumulation, it's O(n^2) which is fine for 300.
    
    # Precompute inverse distances for pairs
    # d is the array of distances.
    # We need to update points_grad[i] and points_grad[j]
    
    # To speed up, we can just do a simple loop. 45k iterations is fast in Python/NumPy if vectorized inner ops, but pure python loop is slow.
    # Let's try to vectorize the update if possible, or just accept the loop cost for 300 points.
    # 45k iterations * simple math is ~1ms? No, Python loop is ~1us/iter, so 45ms. Acceptable per gradient call?
    # L-BFGS-B calls gradient many times. 45ms * 100 calls = 4.5s. Acceptable.
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            prob = pair_probs[idx]
            dist = d[idx]
            if dist < 1e-10:
                idx += 1
                continue
            # d(val)/d(c_i) = (c_i - c_j) / (2 * dist)
            # d(obj)/d(c_i) = -prob * (c_i - c_j) / (2 * dist)
            vec = (points[i] - points[j]) / (2.0 * dist)
            points_grad[i] -= prob * vec
            points_grad[j] += prob * vec  # d(val)/d(c_j) is negative of above
            idx += 1
            
    # Boundary contributions
    # p[n_pairs : ] correspond to boundaries
    boundary_probs = p[n_pairs:]
    
    for i in range(n):
        prob = boundary_probs[i]
        norm = norms[i]
        if norm < 1e-10:
            continue
        # val = 1 - norm
        # d(val)/d(c_i) = - c_i / norm
        # d(obj)/d(c_i) = -prob * (-c_i / norm) = prob * c_i / norm
        points_grad[i] += prob * points[i] / norm
        
    return points_grad.flatten()

def _generate_hexagonal_lattice(n_points):
    """Generate a hexagonal lattice of points."""
    # Estimate spacing
    # Area of unit circle = pi. Area of n circles = n * pi * r^2.
    # r ~ 1/sqrt(n). Spacing ~ 2r.
    spacing = 2.0 / (np.sqrt(n_points) + 1.0)
    
    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) >= n_points:
        # Sort by distance to center to keep the "core" which is usually better packed
        idx = np.argsort(radius[mask])[:n_points]
        return points[idx]
    return points

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 140.0  # 10s margin before 150s limit
    
    # 1. Generate initial guess
    points = _generate_hexagonal_lattice(N)
    if points is None or len(points) != N:
        # Fallback
        points = np.random.randn(N, 2) * 0.5
        points /= np.linalg.norm(points, axis=1, keepdims=True) * 0.9
        
    best_points = points
    best_r = _compute_radius(points)
    
    # 2. Optimize using L-BFGS-B
    # We minimize the negative soft min
    # Note: L-BFGS-B expects 1D array
    
    flat_points = points.flatten()
    
    def objective(x):
        if time.time() > _deadline:
            return 1e10
        return _soft_min_objective(x, beta=100.0)
        
    def gradient(x):
        if time.time() > _deadline:
            return np.zeros_like(x)
        return _soft_min_gradient(x, beta=100.0)

    try:
        result = minimize(
            objective,
            flat_points,
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': 1000,
                'maxfun': 2000,
                'gtol': 1e-6,
                'ftol': 1e-8
            }
        )
        
        if result.success or result.fun < objective(flat_points):
            opt_points = result.x.reshape(N, 2)
            r_opt = _compute_radius(opt_points)
            if r_opt > best_r:
                best_r = r_opt
                best_points = opt_points
    except Exception as e:
        pass

    # 3. If time remains, do a second pass with higher beta for sharper minimum
    if time.time() < _deadline - 20:
        flat_points = best_points.flatten()
        
        def objective2(x):
            if time.time() > _deadline:
                return 1e10
            return _soft_min_objective(x, beta=200.0)
            
        def gradient2(x):
            if time.time() > _deadline:
                return np.zeros_like(x)
            return _soft_min_gradient(x, beta=200.0)

        try:
            result2 = minimize(
                objective2,
                flat_points,
                jac=gradient2,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'maxfun': 1000,
                    'gtol': 1e-7
                }
            )
            
            if result2.success or result2.fun < objective2(flat_points):
                opt_points2 = result2.x.reshape(N, 2)
                r_opt2 = _compute_radius(opt_points2)
                if r_opt2 > best_r:
                    best_r = r_opt2
                    best_points = opt_points2
        except Exception:
            pass

    # Final check
    if best_points is None:
        return points
    return best_points
# EVOLVE-BLOCK-END
