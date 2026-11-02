"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 145  # seconds, safety margin from 150s limit

# Vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def _bary_to_cartesian(bary_coords):
    """Convert barycentric coordinates (n, 3) to Cartesian (n, 2)."""
    # bary_coords: each row sums to 1, shape (n, 3)
    # Cartesian = bary[:, 0:1]*V0 + bary[:, 1:2]*V1 + bary[:, 2:3]*V2
    return bary_coords[:, 0:1] * V0 + bary_coords[:, 1:2] * V1 + bary_coords[:, 2:3] * V2

def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the unit equilateral triangle area."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Compute all pairwise differences
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    # Cross product for all triples (i, j, k)
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    # Extract i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    min_area = np.min(valid_areas)
    return min_area / TRI_AREA

def _is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    eps = 1e-9
    if y < -eps or y > S3H + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    if y > S3H * (1 - x) + eps:
        return False
    if y > S3H * x + eps:
        return False
    return True

def _compute_all_areas_cartesian(points):
    """Compute all 165 triangle areas for 11 points. Returns array of shape (165,)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    return valid_areas

def _soft_min_objective(bary_params, temperature=0.01):
    """
    Objective: minimize negative soft-min of triangle areas + boundary penalty.
    bary_params: shape (n, 3), barycentric coordinates (each row sums to 1).
    We optimize over 2*n free parameters (first 2 bary coords per point, 3rd = 1 - sum).
    """
    # bary_params is flattened: 2*N values (u_i, v_i for each point i)
    # Reconstruct barycentric: (u_i, v_i, 1-u_i-v_i)
    uv = bary_params.reshape(-1, 2)
    bary = np.zeros((N, 3))
    bary[:, 0] = uv[:, 0]
    bary[:, 1] = uv[:, 1]
    bary[:, 2] = 1.0 - uv[:, 0] - uv[:, 1]
    
    # Convert to Cartesian
    points = _bary_to_cartesian(bary)
    
    # Compute all triangle areas
    areas = _compute_all_areas_cartesian(points)
    
    if len(areas) == 0:
        return 0.0
    
    # Soft minimum using log-sum-exp trick: soft_min = -T * log(sum(exp(-a_i / T)))
    # This approximates min(a_i) as T -> 0
    min_area = -temperature * np.log(np.sum(np.exp(-areas / temperature)))
    
    # Boundary penalty: penalize if any barycentric coord is negative (outside triangle)
    penalty = 0.0
    for i in range(N):
        for j in range(3):
            if bary[i, j] < 0:
                penalty += bary[i, j] ** 2
    
    return -min_area / TRI_AREA + 100.0 * penalty

def _soft_min_gradient(bary_params, temperature=0.01):
    """Compute gradient of the soft-min objective."""
    uv = bary_params.reshape(-1, 2)
    bary = np.zeros((N, 3))
    bary[:, 0] = uv[:, 0]
    bary[:, 1] = uv[:, 1]
    bary[:, 2] = 1.0 - uv[:, 0] - uv[:, 1]
    
    points = _bary_to_cartesian(bary)
    areas = _compute_all_areas_cartesian(points)
    
    if len(areas) == 0:
        return np.zeros_like(bary_params)
    
    # Soft min: f = -T * log(sum(exp(-a_i/T)))
    # df/da_i = exp(-a_i/T) / sum(exp(-a_j/T))
    exp_neg = np.exp(-areas / temperature)
    weights = exp_neg / np.sum(exp_neg)
    
    # Now we need da_i / d(bary_params)
    # Each area a_{ijk} = 0.5 * |cross(p_i - p_j, p_i - p_k)|
    # This is complex to differentiate analytically. 
    # For now, use numerical gradient or a simplified approach.
    # Let's use a finite difference approach for the gradient.
    
    grad = np.zeros_like(bary_params)
    step = 1e-7
    for idx in range(len(bary_params)):
        old_val = bary_params[idx]
        bary_params[idx] += step
        f_plus = _soft_min_objective(bary_params, temperature)
        bary_params[idx] -= 2 * step
        f_minus = _soft_min_objective(bary_params, temperature)
        bary_params[idx] = old_val
        grad[idx] = (f_plus - f_minus) / (2 * step)
    
    return grad

def _generate_initial_bary(rng):
    """Generate initial barycentric coordinates using farthest point sampling."""
    # Start with center
    bary = np.zeros((N, 3))
    bary[0] = [1/3, 1/3, 1/3]
    points = _bary_to_cartesian(bary)
    
    for i in range(1, N):
        best_bary = None
        best_min_dist = -1
        for _ in range(300):
            u, v = rng.random(), rng.random()
            if u + v < 1:
                candidate_bary = np.array([u, v, 1-u-v])
                candidate_pt = _bary_to_cartesian(candidate_bary.reshape(1, 3))[0]
                min_dist = np.min(np.linalg.norm(points[:i] - candidate_pt, axis=1))
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_bary = candidate_bary
        
        if best_bary is not None:
            bary[i] = best_bary
            points = np.vstack([points, _bary_to_cartesian(best_bary.reshape(1, 3))])
    
    return bary

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.RandomState(42)
    
    # Use L-BFGS-B with soft-minimum objective in barycentric coordinates
    # Parameters: 2*N = 22 free parameters (u_i, v_i for each point)
    
    restarts = 0
    while time.time() - start_time < DEADLINE - 20:
        restarts += 1
        
        # Generate initial barycentric coordinates
        bary_init = _generate_initial_bary(rng)
        # Extract free parameters: u_i, v_i (first 2 columns)
        x0 = bary_init[:, :2].flatten()
        
        # Try different temperature values
        temperatures = [0.05, 0.02, 0.01, 0.005]
        
        for temp in temperatures:
            if time.time() - start_time > DEADLINE - 15:
                break
            
            try:
                res = minimize(
                    lambda x: _soft_min_objective(x, temperature=temp),
                    x0,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (2 * N),  # u, v in [0, 1], and u+v <= 1 handled by 3rd coord
                    options={'maxiter': 500, 'maxls': 50, 'ftol': 1e-12}
                )
            except Exception:
                continue
            
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Convert back to Cartesian
            uv = res.x.reshape(-1, 2)
            bary = np.zeros((N, 3))
            bary[:, 0] = uv[:, 0]
            bary[:, 1] = uv[:, 1]
            bary[:, 2] = 1.0 - uv[:, 0] - uv[:, 1]
            
            # Check if all barycentric coords are non-negative
            if np.all(bary >= -1e-9):
                points = _bary_to_cartesian(bary)
                score = _min_area_normalized(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
        
        if best_points is not None:
            # Use best found as new starting point for next restart
            x0 = _bary_from_cartesian(best_points)[:, :2].flatten()
    
    # Fallback
    if best_points is None:
        bary = _generate_initial_bary(rng)
        best_points = _bary_to_cartesian(bary)
        best_score = _min_area_normalized(best_points)
    
    # Final polish with Nelder-Mead in Cartesian coordinates
    if time.time() - start_time < DEADLINE - 2:
        try:
            def _penalized_min_area_cartesian(x):
                points = x.reshape(-1, 2)
                penalty = 0.0
                for i in range(N):
                    px, py = points[i]
                    d_bottom = -py
                    d_top = py - S3H
                    d_left = -px
                    d_right = px - 1
                    d_slant_right = py - S3H * (1 - px)
                    d_slant_left = py - S3H * px
                    dist = max(0.0, d_bottom, d_top, d_left, d_right, d_slant_right, d_slant_left)
                    penalty += dist ** 2
                min_a = _min_area_normalized(points)
                return -min_a + 10.0 * penalty
            
            res_nm = minimize(
                _penalized_min_area_cartesian,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            polished = res_nm.x.reshape(-1, 2)
            if all(_is_inside_triangle(p) for p in polished):
                score = _min_area_normalized(polished)
                if score > best_score:
                    best_points = polished
        except Exception:
            pass
            
    return best_points

def _bary_from_cartesian(points):
    """Convert Cartesian points to barycentric coordinates."""
    # For triangle with vertices V0=(0,0), V1=(1,0), V2=(0.5, S3H)
    # Barycentric: p = b0*V0 + b1*V1 + b2*V2, b0+b1+b2=1
    # x = b1 + 0.5*b2
    # y = S3H*b2
    # So: b2 = y/S3H, b1 = x - 0.5*b2, b0 = 1 - b1 - b2
    bary = np.zeros((len(points), 3))
    bary[:, 2] = points[:, 1] / S3H
    bary[:, 1] = points[:, 0] - 0.5 * bary[:, 2]
    bary[:, 0] = 1.0 - bary[:, 1] - bary[:, 2]
    return bary
# EVOLVE-BLOCK-END
