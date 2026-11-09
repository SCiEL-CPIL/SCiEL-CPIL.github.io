"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Safety margin for the 150s limit (120s per task contract, but evaluator kills at 150s)
# We aim to finish within 110s to be safe.
TIME_LIMIT = 110.0
_start_time = time.time()

def _time_left():
    return TIME_LIMIT - (time.time() - _start_time)

def _compute_radius(centers):
    """
    Compute the exact score: largest common radius for given centers.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    """
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-10):
        return 0.0
    
    # Pairwise distances
    dist_matrix = cdist(centers, centers)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _hex_lattice_positions(n_circles, spacing):
    """Generate hexagonal lattice positions centered in unit square."""
    positions = []
    # Generate hexagonal lattice
    row = 0
    y = 0.0
    while len(positions) < n_circles + 50:  # Generate extra to ensure we have enough
        offset = (row % 2) * spacing * 0.5
        x = 0.0
        while x < 1.5 and len(positions) < n_circles + 50:
            positions.append([x + offset, y])
            x += spacing
        y += spacing * np.sqrt(3.0) / 2.0
        row += 1
    
    positions = np.array(positions[:n_circles])
    
    # Center in unit square [-0.5, 0.5]
    center = np.mean(positions, axis=0)
    positions = positions - center
    positions = positions + 0.0  # Already centered at origin
    
    return positions

def _stage1_lattice_fit(centers, target_positions, max_iter=500):
    """
    Stage 1: Minimize sum of squared distances to relaxed hexagonal lattice.
    This establishes a globally coherent structure.
    """
    def objective(x):
        centers_2d = x.reshape(-1, 2)
        # Soft constraint: stay in square
        penalty = np.sum(np.maximum(0, np.abs(centers_2d) - 0.49)**2)
        
        # Distance to target lattice positions (allowing for some relaxation)
        # Use soft assignment: for each center, find nearest target
        dists = cdist(centers_2d, target_positions)
        # Soft minimum distance to any target
        min_dists = np.min(dists, axis=1)
        obj = np.sum(min_dists**2) + 10.0 * penalty
        return obj
    
    def gradient(x):
        centers_2d = x.reshape(-1, 2)
        # Compute gradient numerically would be expensive, use finite differences
        # For now, use L-BFGS-B with numerical gradient
        return None
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    x0 = centers.flatten()
    
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-5, 'maxls': 20}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return centers

def _softmin_penalty_and_grad(centers, beta=50.0, r_target=0.036):
    """
    Stage 2: Smooth penalty for close pairs using log-sum-exp.
    Penalize pairs that are closer than 2*r_target.
    """
    # Pairwise distances
    dist_matrix = cdist(centers, centers)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_matrix[iu]
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Penalty: for each pair, penalty = max(0, 2*r_target - d)^2 smoothed
    # Use soft version: exp(-beta * (d - 2*r_target)) for d < 2*r_target
    # Actually, let's use a smooth max: softmax(0, 2*r_target - d)
    
    # For pairs: violation_ij = max(0, 2*r_target - d_ij)
    # Smooth: (1/beta) * ln(1 + exp(beta * (2*r_target - d_ij)))
    
    # For walls: violation_i = max(0, r_target - wall_dist_i)
    # Smooth: (1/beta) * ln(1 + exp(beta * (r_target - wall_dist_i)))
    
    pair_violations = 2.0 * r_target - pair_dists
    wall_violations = r_target - wall_dists
    
    # Smooth max(0, v) = (1/beta) * ln(1 + exp(beta * v))
    def smooth_max_0(v, beta):
        # Numerically stable version
        max_v = np.max(v)
        if max_v > 50.0 / beta:
            return max_v + (1.0/beta) * np.log1p(np.exp(-beta * max_v))
        else:
            return (1.0/beta) * np.log1p(np.exp(beta * v))
    
    # Compute penalty
    pair_pen = smooth_max_0(pair_violations, beta)
    wall_pen = smooth_max_0(wall_violations, beta)
    
    total_penalty = np.sum(pair_pen) + np.sum(wall_pen)
    
    # Gradient
    # d/dv smooth_max(0, v) = sigmoid(beta * v)
    def sigmoid(x):
        # Numerically stable
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    pair_grad_v = sigmoid(beta * pair_violations)
    wall_grad_v = sigmoid(beta * wall_violations)
    
    # Gradient with respect to centers
    grad = np.zeros_like(centers)
    
    # Wall gradient
    abs_x = np.abs(centers[:, 0])
    abs_y = np.abs(centers[:, 1])
    
    for i in range(N):
        if abs_x[i] > abs_y[i]:
            grad[i, 0] += wall_grad_v[i] * (-np.sign(centers[i, 0]))
            grad[i, 1] += 0.0
        elif abs_y[i] > abs_x[i]:
            grad[i, 0] += 0.0
            grad[i, 1] += wall_grad_v[i] * (-np.sign(centers[i, 1]))
        else:
            grad[i, 0] += wall_grad_v[i] * (-0.5 * np.sign(centers[i, 0]))
            grad[i, 1] += wall_grad_v[i] * (-0.5 * np.sign(centers[i, 1]))
    
    # Pair gradient
    i_idx, j_idx = iu
    diffs = centers[i_idx] - centers[j_idx]
    safe_dists = np.where(pair_dists < 1e-8, 1e-8, pair_dists)
    unit_vecs = diffs / safe_dists[:, np.newaxis]
    
    # d(pair_violation)/d(c_i) = -unit_vec, d(pair_violation)/d(c_j) = unit_vec
    # d(penalty)/d(c_i) = pair_grad_v * (-unit_vec)
    # d(penalty)/d(c_j) = pair_grad_v * unit_vec
    
    weighted_vecs_i = pair_grad_v[:, np.newaxis] * (-unit_vecs)
    weighted_vecs_j = pair_grad_v[:, np.newaxis] * unit_vecs
    
    np.add.at(grad, i_idx, weighted_vecs_i)
    np.add.at(grad, j_idx, weighted_vecs_j)
    
    return total_penalty, grad

def _stage2_refine(centers, beta=50.0, r_target=0.036, max_iter=200):
    """
    Stage 2: Refine using smooth penalty for close pairs.
    """
    centers = centers.copy()
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        try:
            penalty, grad = _softmin_penalty_and_grad(centers, beta, r_target)
        except Exception:
            break
        
        if np.max(np.abs(grad)) < 1e-8:
            break
        
        # Line search
        step_size = 0.005
        improved = False
        for _ in range(15):
            new_centers = centers - step_size * grad
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            # Check if penalty decreased
            new_penalty, _ = _softmin_penalty_and_grad(new_centers, beta, r_target)
            
            if new_penalty < penalty - 1e-10:
                centers = new_centers
                penalty = new_penalty
                improved = True
                break
            else:
                step_size *= 0.5
                if step_size < 1e-7:
                    break
        
        if not improved:
            # Try with smaller beta
            beta *= 0.8
            if beta < 5.0:
                break
    
    return centers

def _local_optimize(centers, beta=10.0, max_iter=2000):
    """
    Local optimization using gradient descent on the smooth surrogate.
    """
    centers = centers.copy()
    current_radius = _compute_radius(centers)
    
    # Adaptive step size
    step_size = 0.01
    min_step = 1e-6
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
            
        # Compute gradient
        try:
            # Use the softmin penalty approach
            penalty, grad = _softmin_penalty_and_grad(centers, beta, r_target=current_radius)
        except Exception:
            break
            
        if np.max(np.abs(grad)) < 1e-8:
            break
            
        # Line search
        found_step = False
        for _ in range(20):
            new_centers = centers + step_size * grad
            # Clip to square
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centers)
            
            if new_radius > current_radius + 1e-9:
                centers = new_centers
                current_radius = new_radius
                found_step = True
                break
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break
        
        if not found_step:
            break
            
    return centers, current_radius

def _grid_initial(m_cols):
    """Generate a grid initialization."""
    m_rows = int(np.ceil(N / m_cols))
    xs = (np.arange(m_cols) + 0.5) / m_cols - 0.5
    ys = (np.arange(m_rows) + 0.5) / m_rows - 0.5
    
    centers = []
    for y in ys:
        for x in xs:
            centers.append([x, y])
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
            
    # Pad if necessary (shouldn't happen with ceil, but just in case)
    while len(centers) < N:
        centers.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
        
    return np.array(centers[:N])

def _hex_initial():
    """Generate a hexagonal initialization."""
    centers = []
    # Approximate spacing for N circles in unit square
    # Area per circle ~ 1/N. Hex area ~ sqrt(3)/2 * s^2.
    # s ~ sqrt(2/(sqrt(3)*N))
    s = np.sqrt(2.0 / (np.sqrt(3.0) * N))
    
    y = -0.5 + s * 0.5
    row = 0
    while y < 0.5 and len(centers) < N:
        offset = (row % 2) * s * 0.5
        x = -0.5 + s * 0.5 + offset
        while x < 0.5 and len(centers) < N:
            if abs(x) <= 0.5 and abs(y) <= 0.5:
                centers.append([x, y])
            x += s
        y += s * np.sqrt(3.0) / 2.0
        row += 1
        
    # If we have too many, trim. If too few, pad.
    if len(centers) > N:
        centers = centers[:N]
    else:
        while len(centers) < N:
            centers.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
            
    return np.array(centers)

def construct():
    """
    Construct the centers of 200 equal circles inside [-1/2, 1/2]^2
    to maximize the common radius.
    """
    global _start_time
    _start_time = time.time()
    
    best_centers = None
    best_r = -1.0
    
    # List of initializations to try
    initializations = []
    
    # 1. Standard Grid
    for m in [10, 11, 12, 14, 15]:
        initializations.append(_grid_initial(m))
        
    # 2. Hexagonal
    initializations.append(_hex_initial())
    
    # 3. Random (a few)
    for _ in range(3):
        initializations.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # Process each initialization with two-stage optimization
    for init_idx, init_centers in enumerate(initializations):
        if _time_left() < 15.0:
            break
            
        # Evaluate initial
        r = _compute_radius(init_centers)
        if r > best_r:
            best_r = r
            best_centers = init_centers.copy()
        
        # Stage 1: Lattice fit for coherent structure
        if _time_left() > 30.0:
            try:
                # Generate target lattice positions
                spacing = np.sqrt(2.0 / (np.sqrt(3.0) * N))
                target_positions = _hex_lattice_positions(N, spacing)
                
                # Fit to lattice
                stage1_centers = _stage1_lattice_fit(init_centers, target_positions, max_iter=100)
                
                # Evaluate stage 1 result
                r1 = _compute_radius(stage1_centers)
                if r1 > best_r:
                    best_r = r1
                    best_centers = stage1_centers.copy()
            except Exception:
                pass
        
        # Stage 2: Refine with smooth penalty
        if _time_left() > 20.0:
            try:
                # Use current best or stage 1 result
                start_centers = best_centers if best_centers is not None else init_centers
                r_target = max(best_r, _compute_radius(start_centers))
                
                stage2_centers = _stage2_refine(start_centers, beta=50.0, r_target=r_target, max_iter=100)
                
                # Evaluate stage 2 result
                r2 = _compute_radius(stage2_centers)
                if r2 > best_r:
                    best_r = r2
                    best_centers = stage2_centers.copy()
            except Exception:
                pass
        
        # Final local optimization
        if _time_left() > 10.0 and best_centers is not None:
            try:
                opt_centers, opt_r = _local_optimize(best_centers, beta=5.0, max_iter=500)
                if opt_r > best_r:
                    best_r = opt_r
                    best_centers = opt_centers.copy()
            except Exception:
                pass

    # Final Polish on the best result
    if best_centers is not None and _time_left() > 15.0:
        try:
            # Use a smaller beta for final precision
            final_centers, final_r = _local_optimize(best_centers, beta=2.0, max_iter=1000)
            if final_r > best_r:
                best_r = final_r
                best_centers = final_centers
        except Exception:
            pass
            
    # Ensure valid output
    if best_centers is None:
        return _grid_initial(10)
        
    # Final validation check
    if _compute_radius(best_centers) <= 0:
        return _grid_initial(10)
        
    return best_centers

# EVOLVE-BLOCK-END
