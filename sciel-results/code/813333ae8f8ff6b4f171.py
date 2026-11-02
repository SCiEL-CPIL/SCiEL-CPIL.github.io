import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

def _smooth_max(z, eps=1e-6):
    """Smooth approximation of max(0, z)"""
    return 0.5 * (z + np.sqrt(z**2 + eps**2))

def _objective(params, n, mu, eps=1e-6):
    """
    params: (n, 3) array of (x, y, r)
    Returns: -sum(r) + mu * (sum of squared smooth violations)
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # 1. Boundary violations
    # x - r >= 0 => violation if r - x > 0
    # 1 - x - r >= 0 => violation if x + r - 1 > 0
    # y - r >= 0
    # 1 - y - r >= 0
    
    v_x_min = np.maximum(0, r - x)
    v_x_max = np.maximum(0, x + r - 1)
    v_y_min = np.maximum(0, r - y)
    v_y_max = np.maximum(0, y + r - 1)
    
    # Use smooth max for differentiability if needed, but np.maximum is fine for penalty if we use subgradient or if L-BFGS-B handles it.
    # However, L-BFGS-B expects smooth functions. Let's use the smooth version.
    v_x_min_s = _smooth_max(r - x, eps)
    v_x_max_s = _smooth_max(x + r - 1, eps)
    v_y_min_s = _smooth_max(r - y, eps)
    v_y_max_s = _smooth_max(y + r - 1, eps)
    
    boundary_penalty = np.sum(v_x_min_s**2) + np.sum(v_x_max_s**2) + \
                       np.sum(v_y_min_s**2) + np.sum(v_y_max_s**2)
    
    # 2. Pairwise overlap violations
    # d_ij >= r_i + r_j
    # violation if r_i + r_j - d_ij > 0
    
    # Compute distances
    # cdist gives (n, n) matrix
    dist_mat = cdist(params[:, :2], params[:, :2])
    r_sum_mat = r[:, None] + r[None, :]
    
    # We only need upper triangle
    iu = np.triu_indices(n, k=1)
    d_ij = dist_mat[iu]
    r_ij = r_sum_mat[iu]
    
    v_overlap = _smooth_max(r_ij - d_ij, eps)
    overlap_penalty = np.sum(v_overlap**2)
    
    total_penalty = boundary_penalty + overlap_penalty
    
    # Objective: Maximize sum(r) => Minimize -sum(r)
    # Add penalty
    return -np.sum(r) + mu * total_penalty

def _gradient(params, n, mu, eps=1e-6):
    """
    Analytical gradient for better performance and stability.
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    grad = np.zeros_like(params)
    
    # Boundary gradients
    # f = -sum(r) + mu * ( ... )
    # d/dr: -1 + mu * (2*v_x_min * dv/dr + ...)
    # dv_x_min/dr = 1 if r>x else 0 (smoothed)
    
    # Smooth max derivative: d/dz [0.5(z + sqrt(z^2+eps^2))] = 0.5(1 + z/sqrt(z^2+eps^2))
    # Let h(z) = 0.5(1 + z/sqrt(z^2+eps^2))
    
    def _smooth_max_grad(z, eps):
        return 0.5 * (1 + z / np.sqrt(z**2 + eps**2))
    
    # x - r < 0 -> z = r - x
    z_x_min = r - x
    g_x_min = _smooth_max_grad(z_x_min, eps)
    # 1 - x - r < 0 -> z = x + r - 1
    z_x_max = x + r - 1
    g_x_max = _smooth_max_grad(z_x_max, eps)
    
    z_y_min = r - y
    g_y_min = _smooth_max_grad(z_y_min, eps)
    z_y_max = y + r - 1
    g_y_max = _smooth_max_grad(z_y_max, eps)
    
    # Penalty term P = sum(v^2)
    # dP/dr = 2 * v * dv/dr
    # dv_x_min/dr = g_x_min
    # dv_x_min/dx = -g_x_min
    
    # grad_r += -1 (from -sum r)
    grad[:, 2] -= 1.0
    
    # Boundary contributions to grad
    # d/dr: 2 * v * g
    grad[:, 2] += 2 * mu * (
        _smooth_max(z_x_min, eps) * g_x_min +
        _smooth_max(z_x_max, eps) * g_x_max +
        _smooth_max(z_y_min, eps) * g_y_min +
        _smooth_max(z_y_max, eps) * g_y_max
    )
    
    # d/dx: 2 * v * dv/dx
    # dv_x_min/dx = -g_x_min
    # dv_x_max/dx = g_x_max
    grad[:, 0] += 2 * mu * (
        -_smooth_max(z_x_min, eps) * g_x_min +
        _smooth_max(z_x_max, eps) * g_x_max
    )
    
    # d/dy:
    grad[:, 1] += 2 * mu * (
        -_smooth_max(z_y_min, eps) * g_y_min +
        _smooth_max(z_y_max, eps) * g_y_max
    )
    
    # Pairwise overlap
    iu = np.triu_indices(n, k=1)
    d_ij = np.sqrt((x[iu[0]] - x[iu[1]])**2 + (y[iu[0]] - y[iu[1]])**2)
    r_ij = r[iu[0]] + r[iu[1]]
    
    z_overlap = r_ij - d_ij
    g_overlap = _smooth_max_grad(z_overlap, eps)
    v_overlap = _smooth_max(z_overlap, eps)
    
    # Penalty term Q = sum(v_overlap^2)
    # dQ/dr_i = 2 * v * g * 1 (since d(r_i+r_j)/dr_i = 1)
    # dQ/dx_i = 2 * v * g * d(r_i+r_j - d_ij)/dx_i = 2 * v * g * (0 - d(-d_ij)/dx_i)
    # d(-d_ij)/dx_i = (x_i - x_j) / d_ij
    # So dQ/dx_i = 2 * v * g * (x_i - x_j) / d_ij
    
    # Accumulate gradients for each pair
    # We need to scatter add to grad
    for k in range(len(iu[0])):
        i = iu[0][k]
        j = iu[1][k]
        
        d_k = d_ij[k]
        if d_k < 1e-12:
            continue # avoid division by zero, though smooth max handles it mostly
            
        v_k = v_overlap[k]
        g_k = g_overlap[k]
        
        # Contribution to grad_r[i] and grad_r[j]
        # dQ/dr_i = 2 * mu * v_k * g_k
        grad[i, 2] += 2 * mu * v_k * g_k
        grad[j, 2] += 2 * mu * v_k * g_k
        
        # Contribution to grad_x[i] and grad_x[j]
        # dQ/dx_i = 2 * mu * v_k * g_k * (x_i - x_j) / d_k
        # dQ/dx_j = 2 * mu * v_k * g_k * (x_j - x_i) / d_k
        
        factor = 2 * mu * v_k * g_k / d_k
        grad[i, 0] += factor * (x[i] - x[j])
        grad[j, 0] += factor * (x[j] - x[i])
        
        # Contribution to grad_y[i] and grad_y[j]
        grad[i, 1] += factor * (y[i] - y[j])
        grad[j, 1] += factor * (y[j] - y[i])
        
    return grad

def _is_valid(params, tol=1e-6):
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    if np.any(r < -tol): return False
    if np.any(x < -tol) or np.any(x > 1 + tol): return False
    if np.any(y < -tol) or np.any(y > 1 + tol): return False
    
    # Boundary
    if np.any(x - r < -tol): return False
    if np.any(x + r > 1 + tol): return False
    if np.any(y - r < -tol): return False
    if np.any(y + r > 1 + tol): return False
    
    # Overlaps
    dist_mat = cdist(params[:, :2], params[:, :2])
    r_sum_mat = r[:, None] + r[None, :]
    # Check upper triangle
    iu = np.triu_indices(N, k=1)
    d = dist_mat[iu]
    r_sum = r_sum_mat[iu]
    if np.any(d < r_sum - tol):
        return False
        
    return True

def _clean_up(params):
    """
    Attempt to fix small violations by shrinking radii or moving centers.
    """
    params = params.copy()
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # Clamp to boundaries strictly
    r = np.maximum(r, 1e-8)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    params[:, 0] = x
    params[:, 1] = y
    params[:, 2] = r
    
    # Fix overlaps by shrinking radii
    # Simple iterative shrink
    for _ in range(50):
        valid = True
        dist_mat = cdist(params[:, :2], params[:, :2])
        iu = np.triu_indices(N, k=1)
        d = dist_mat[iu]
        r_sum = params[iu[0], 2] + params[iu[1], 2]
        
        violations = r_sum - d
        if np.all(violations <= 1e-9):
            break
        
        # Find max violation
        max_viol_idx = np.argmax(violations)
        i = iu[0][max_viol_idx]
        j = iu[1][max_viol_idx]
        
        # Shrink both radii by half the violation
        shrink = violations[max_viol_idx] / 2.0 + 1e-9
        params[i, 2] -= shrink
        params[j, 2] -= shrink
        
        # Ensure radii don't go negative
        params[i, 2] = max(params[i, 2], 1e-8)
        params[j, 2] = max(params[j, 2], 1e-8)
        
        # Re-clamp centers
        params[i, 0] = np.clip(params[i, 0], params[i, 2], 1 - params[i, 2])
        params[i, 1] = np.clip(params[i, 1], params[i, 2], 1 - params[i, 2])
        params[j, 0] = np.clip(params[j, 0], params[j, 2], 1 - params[j, 2])
        params[j, 1] = np.clip(params[j, 1], params[j, 2], 1 - params[j, 2])
        
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = -np.inf
    best_params = None
    
    # Initializers
    initializers = []
    
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    initializers.append(np.array(cells[:N]))
    
    # 2. Hexagonal
    # Approximate hex packing
    # Spacing s, radius r = s / (2 * sin(60)) = s / sqrt(3)
    # In unit square, fit as many as possible.
    # Let's just generate a hex grid and pick 26
    hex_points = []
    r_hex = 0.08 # Start with a guess
    s = r_hex * np.sqrt(3)
    y = r_hex
    row = 0
    while y < 1 - r_hex:
        x_offset = 0 if row % 2 == 0 else s / 2
        x = x_offset + r_hex
        while x < 1 - r_hex:
            hex_points.append((x, y, r_hex))
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
    
    if len(hex_points) >= N:
        # Shuffle and take N? Or take first N?
        # Taking first N might be corner-biased.
        # Let's center them a bit.
        # For simplicity, just take first N for now, or random sample
        idx = np.random.choice(len(hex_points), N, replace=False)
        initializers.append(np.array([hex_points[i] for i in idx]))
        
    # 3. Random Uniform
    for _ in range(5):
        r_rand = np.random.uniform(0.02, 0.1, N)
        x_rand = np.random.uniform(r_rand, 1 - r_rand)
        y_rand = np.random.uniform(r_rand, 1 - r_rand)
        initializers.append(np.column_stack((x_rand, y_rand, r_rand)))
        
    # 4. Random Clustered
    for _ in range(5):
        # Create a few cluster centers
        n_clusters = 4
        centers = np.random.uniform(0.2, 0.8, (n_clusters, 2))
        r_rand = np.random.uniform(0.03, 0.1, N)
        # Assign each circle to a cluster
        cluster_idx = np.random.randint(0, n_clusters, N)
        x_rand = centers[cluster_idx, 0] + np.random.normal(0, 0.1, N)
        y_rand = centers[cluster_idx, 1] + np.random.normal(0, 0.1, N)
        # Clamp
        x_rand = np.clip(x_rand, r_rand, 1 - r_rand)
        y_rand = np.clip(y_rand, r_rand, 1 - r_rand)
        initializers.append(np.column_stack((x_rand, y_rand, r_rand)))

    for init_params in initializers:
        if time.time() > deadline:
            break
            
        # Annealing loop
        current_params = init_params.copy()
        mu = 1.0
        
        # Optimize
        try:
            # We will run minimize a few times, increasing mu
            for step in range(10):
                if time.time() > deadline:
                    break
                    
                res = minimize(
                    _objective, 
                    current_params, 
                    args=(N, mu, 1e-6), 
                    method='L-BFGS-B', 
                    jac=_gradient, 
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                current_params = res.x
                
                # Check validity
                if _is_valid(current_params, tol=1e-5):
                    score = np.sum(current_params[:, 2])
                    if score > best_score:
                        best_score = score
                        best_params = current_params.copy()
                        
                # Increase penalty
                mu *= 2.0
                
                # If penalty is very high and we are still invalid, we might be stuck
                # But L-BFGS-B should handle it.
                
        except Exception as e:
            pass
            
    # Final cleanup and check
    if best_params is not None:
        best_params = _clean_up(best_params)
        if _is_valid(best_params, tol=1e-6):
            return best_params
        else:
            # If still invalid, return a trivial valid solution
            # Grid solution is always valid
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
