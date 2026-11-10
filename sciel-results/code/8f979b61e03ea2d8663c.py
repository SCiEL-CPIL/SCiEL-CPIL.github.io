"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the score: largest common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube [-0.5, 0.5]^3
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Wall distance
    wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dist)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_wall, min_pair / 2.0)


def _soft_min(values, temperature=0.01):
    """Soft minimum using log-sum-exp trick (negative log-sum-exp of negated values)."""
    m = np.min(values)
    return m - temperature * np.log(np.sum(np.exp(-(values - m) / temperature)))


def _objective_and_grad(centers, temp=0.02):
    """
    Differentiable surrogate objective: maximize soft-min of (pairwise_dist/2, wall_dist).
    Returns (value, gradient) where gradient points in direction of increase.
    """
    centers = np.asarray(centers, dtype=np.float64)
    
    # Wall distances: 0.5 - |c_ik| for each axis
    # wall_dist_i = min over k of (0.5 - |c_ik|)
    abs_c = np.abs(centers)  # (N, 3)
    wall_per_axis = 0.5 - abs_c  # (N, 3)
    # Soft min over axes for each sphere, then soft min over spheres
    # Actually we want soft min of all wall distances and all pairwise half-distances
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]  # (N, N, 3)
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)  # (N, N)
    np.fill_diagonal(dists, np.inf)
    half_dists = dists / 2.0  # (N, N)
    
    # We want to maximize r such that all constraints hold.
    # Surrogate: soft-min of {half_dists[i,j] for i<j} union {wall_dist[i] for i}
    
    # Flatten all constraints
    # Wall distances per sphere (use soft min over axes first)
    # For each sphere, wall_dist = min_k (0.5 - |c_ik|)
    # Soft min over axes:
    wall_soft_per_sphere = -temp * np.log(np.sum(np.exp(-wall_per_axis / temp), axis=1))  # (N,)
    
    # For pairwise, we have N*(N-1)/2 values. Let's use a subset or all.
    # Use all upper triangle
    iu = np.triu_indices(N, k=1)
    pd_vals = half_dists[iu]  # (N*(N-1)/2,)
    
    # Combine all constraint values into one array
    all_vals = np.concatenate([wall_soft_per_sphere, pd_vals])
    
    # Soft min of all
    m = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - m) / temp)
    sum_exp = np.sum(exp_vals)
    value = m - temp * np.log(sum_exp)
    
    # Gradient with respect to centers
    # d(value)/d(c) = -sum_i exp(-v_i/temp) / sum_exp * d(v_i)/d(c)
    # where v_i are the constraint values
    
    weights = exp_vals / sum_exp  # (num_constraints,)
    
    grad = np.zeros_like(centers)
    
    # Wall part: for sphere i, v_i = soft_min over axes of (0.5 - |c_ik|)
    # d(v_i)/d(c_ik) = -soft_max_weight_k * sign(c_ik)
    # soft_max_weight_k = exp(-wall_per_axis[i,k]/temp) / sum_k exp(-wall_per_axis[i,k]/temp)
    wall_exp = np.exp(-wall_per_axis / temp)  # (N, 3)
    wall_sum = np.sum(wall_exp, axis=1, keepdims=True)  # (N, 1)
    wall_weights = wall_exp / wall_sum  # (N, 3)
    signs = np.sign(centers)
    # d(wall_soft_per_sphere[i])/d(c_ik) = -wall_weights[i,k] * signs[i,k]
    # Contribution to grad from wall constraints:
    for i in range(N):
        w_i = weights[i]  # weight of wall constraint i in the overall soft min
        grad[i] += -w_i * wall_weights[i] * signs[i]
    
    # Pairwise part: for pair (i,j), v = dists[i,j]/2
    # d(v)/d(c_i) = (c_i - c_j) / dists[i,j]
    # d(v)/d(c_j) = (c_j - c_i) / dists[i,j]
    # We need to map from flat index to (i,j)
    # iu gives us indices
    i_idx = iu[0]
    j_idx = iu[1]
    # weights for pairwise part start at index N
    pd_weights = weights[N:]  # (N*(N-1)/2,)
    
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        d = dists[i, j]
        if d < 1e-10:
            continue
        w = pd_weights[k]
        direction = (centers[i] - centers[j]) / d
        grad[i] += -w * direction
        grad[j] += w * direction
    
    return value, grad


def _initial_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_centers = _initial_grid()
    best_score = _compute_radius(best_centers)
    
    # Gradient ascent with adaptive temperature and step size
    temp = 0.05
    step_size = 0.01
    n_iter = 0
    
    centers = best_centers.copy()
    
    # Schedule: gradually reduce temperature
    while time.time() < deadline - 5.0:
        # Compute objective and gradient
        val, grad = _objective_and_grad(centers, temp)
        
        # Check time
        if time.time() > deadline - 2.0:
            break
        
        # Take a step
        centers_new = centers + step_size * grad
        
        # Project back into cube with small margin
        margin = 1e-6
        centers_new = np.clip(centers_new, -0.5 + margin, 0.5 - margin)
        
        # Evaluate
        score_new = _compute_radius(centers_new)
        
        if score_new > best_score:
            best_score = score_new
            best_centers = centers_new.copy()
            # Slightly increase step size on success
            step_size = min(step_size * 1.05, 0.02)
            # Reduce temperature slowly
            temp = max(temp * 0.995, 0.005)
        else:
            # Reduce step size on failure
            step_size = max(step_size * 0.9, 0.001)
            # Keep centers as is (don't move)
            pass
        
        centers = centers_new
        n_iter += 1
        
        # Periodically try a small random perturbation to escape local minima
        if n_iter % 200 == 0 and time.time() < deadline - 10.0:
            perturbation = 0.005 * np.random.randn(N, 3)
            centers_pert = np.clip(centers + perturbation, -0.5 + margin, 0.5 - margin)
            score_pert = _compute_radius(centers_pert)
            if score_pert > best_score:
                best_score = score_pert
                best_centers = centers_pert.copy()
                centers = centers_pert.copy()
    
    # Final polish: try to improve with smaller steps
    if time.time() < deadline - 1.0:
        centers = best_centers.copy()
        temp = 0.005
        step_size = 0.001
        polish_deadline = min(time.time() + 3.0, deadline - 0.5)
        while time.time() < polish_deadline:
            val, grad = _objective_and_grad(centers, temp)
            centers_new = centers + step_size * grad
            centers_new = np.clip(centers_new, -0.5 + 1e-6, 0.5 - 1e-6)
            score_new = _compute_radius(centers_new)
            if score_new > best_score:
                best_score = score_new
                best_centers = centers_new.copy()
            else:
                step_size *= 0.8
            centers = centers_new
    
    return best_centers

# EVOLVE-BLOCK-END
