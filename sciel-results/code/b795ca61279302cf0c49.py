"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def _soft_max(vals, k=100.0):
    """Smooth approximation to max(vals) using log-sum-exp."""
    vals = np.asarray(vals, dtype=float)
    m = np.max(vals)
    if not np.isfinite(m):
        return m
    # soft max = m + log(sum(exp(vals - m))) / k  (scaled soft max)
    # Actually we want a smooth max that approximates max as k->inf
    # soft_max = (1/k) * log(sum(exp(k * vals)))
    # But for numerical stability:
    # soft_max = m + (1/k) * log(sum(exp(k*(vals - m))))
    shifted = k * (vals - m)
    # clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m + log_sum / k


def _covering_radius(x, sample_points):
    """Compute covering radius: max over sample points of min angular distance to x."""
    # x: (N, 3), sample_points: (M, 3)
    # Compute all pairwise dot products
    dots = np.einsum('ij,kj->ik', sample_points, x)  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    # Angular distance: arccos(dot)
    # We want max over i of min over j of arccos(dots[i,j])
    # = max over i of arccos(max over j of dots[i,j])  (since arccos is decreasing)
    max_dots = np.max(dots, axis=1)  # (M,)
    angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
    return np.max(angles)


def _soft_covering_radius(x, sample_points, k=100.0):
    """Smooth covering radius using soft-max over the angular distances."""
    dots = np.einsum('ij,kj->ik', sample_points, x)
    dots = np.clip(dots, -1.0, 1.0)
    # For each sample point, the nearest config point gives the largest dot
    # soft_min of angles = soft_max of dots (since angle = arccos(dot), decreasing)
    # soft_max of dots along axis=1
    max_dots = _soft_max_along_axis(dots, axis=1, k=k)
    angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
    # Now soft max over all angles
    return _soft_max(angles, k=k)


def _soft_max_along_axis(vals, axis, k=100.0):
    """Soft max along a given axis."""
    m = np.max(vals, axis=axis, keepdims=True)
    shifted = k * (vals - m)
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return (m + log_sum / k).squeeze(axis)


def _project_to_sphere(x):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def _gradient_soft_covering(x, sample_points, k=100.0):
    """
    Compute gradient of the soft covering radius w.r.t. x.
    x: (N, 3), sample_points: (M, 3)
    
    The soft covering radius is a smooth surrogate. We use automatic-diff-like 
    manual gradients.
    
    Actually, let's use a simpler approach: gradient descent with finite differences 
    or use the structure of the problem.
    
    Alternative: use the fact that the gradient of arccos(dot) w.r.t. x_j is 
    -dot / (sqrt(1-dot^2)) * y_i for the nearest point.
    
    For the soft version, we need weighted sums.
    """
    dots = np.einsum('ij,kj->ik', sample_points, x)  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Soft max weights for each sample point over config points
    m = np.max(dots, axis=1, keepdims=True)
    shifted = k * (dots - m)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    weights = exp_shifted / sum_exp  # (M, N) - soft assignment weights
    
    # soft max dot value
    max_dots_soft = np.sum(weights * dots, axis=1)  # (M,)
    
    # angles = arccos(max_dots_soft)
    # d(angle)/d(max_dots_soft) = -1/sqrt(1 - max_dots_soft^2)
    max_dots_clipped = np.clip(max_dots_soft, -0.999999, 0.999999)
    dangle_ddot = -1.0 / np.sqrt(1.0 - max_dots_clipped**2)  # (M,)
    
    # Soft max over angles: 
    # soft_max(angles) = m + (1/k) * log(sum(exp(k*(angles - m))))
    # We need gradient of this w.r.t. x
    # This is getting complex. Let's use a simpler gradient approach.
    
    # Simpler: gradient of sum of max angles (not soft), approximated by weighted sum
    # grad_x_j = sum_i weights[i,j] * dangle_ddot[i] * d(dot[i,j])/dx_j
    # d(dot[i,j])/dx_j = sample_points[i]  (since dot = y_i . x_j)
    # But we need to account for the soft max over angles too.
    
    # For simplicity, use the gradient of the max angle (hard) with soft assignment:
    # grad_x_j = sum_i weights[i,j] * dangle_ddot[i] * y_i
    
    # Reshape for einsum
    # weights: (M, N), dangle_ddot: (M,), sample_points: (M, 3)
    # grad[j] = sum_i weights[i,j] * dangle_ddot[i] * sample_points[i]
    grad = np.einsum('ij,i,ik->jk', weights, dangle_ddot, sample_points)
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
    grad_norm = np.maximum(grad_norm, 1e-12)
    grad = grad / grad_norm
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Generate a good set of sample points on the sphere
    # Use a Fibonacci spiral with more points for better coverage
    M = 2000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    sample_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial configuration: Fibonacci spiral
    x = np.zeros((N, 3))
    i_init = np.arange(N) + 0.5
    z_init = 1.0 - 2.0 * i_init / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i_init
    r_init = np.sqrt(1.0 - z_init * z_init)
    x[:, 0] = r_init * np.cos(phi_init)
    x[:, 1] = r_init * np.sin(phi_init)
    x[:, 2] = z_init
    
    best_x = x.copy()
    best_radius = _covering_radius(x, sample_points)
    
    # Gradient descent with backtracking line search
    k_soft = 50.0
    lr = 0.05
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Compute gradient
        grad = _gradient_soft_covering(x, sample_points, k=k_soft)
        
        # Backtracking line search
        step = lr
        for _ in range(30):
            x_trial = x - step * grad
            x_trial = _project_to_sphere(x_trial)
            
            # Check for coincident points
            min_dist = np.min(np.linalg.norm(x_trial[:, None, :] - x_trial[None, :, :], axis=2))
            if min_dist < 1e-6:
                step *= 0.5
                continue
            
            radius_trial = _covering_radius(x_trial, sample_points)
            if radius_trial < best_radius or radius_trial < _covering_radius(x, sample_points):
                x = x_trial
                best_radius = min(best_radius, radius_trial)
                if radius_trial < best_radius:
                    best_x = x.copy()
                break
            else:
                step *= 0.5
        
        # Occasionally restart with perturbation to escape local minima
        if it % 500 == 499 and time.time() < deadline - 10.0:
            # Try a few random perturbations
            for _ in range(3):
                if time.time() > deadline - 10.0:
                    break
                x_pert = x + 0.1 * np.random.randn(N, 3)
                x_pert = _project_to_sphere(x_pert)
                radius_pert = _covering_radius(x_pert, sample_points)
                if radius_pert < best_radius:
                    best_x = x_pert.copy()
                    best_radius = radius_pert
                    x = x_pert
    
    # Final polish: use the best configuration
    # Do a few more iterations with smaller step
    x = best_x.copy()
    lr = 0.01
    for _ in range(200):
        if time.time() > deadline - 2.0:
            break
        grad = _gradient_soft_covering(x, sample_points, k=100.0)
        step = lr
        for _ in range(20):
            x_trial = x - step * grad
            x_trial = _project_to_sphere(x_trial)
            min_dist = np.min(np.linalg.norm(x_trial[:, None, :] - x_trial[None, :, :], axis=2))
            if min_dist < 1e-6:
                step *= 0.5
                continue
            radius_trial = _covering_radius(x_trial, sample_points)
            if radius_trial < _covering_radius(x, sample_points):
                x = x_trial
                break
            else:
                step *= 0.5
        if _covering_radius(x, sample_points) < best_radius:
            best_x = x.copy()
            best_radius = _covering_radius(x, sample_points)
    
    return _project_to_sphere(best_x)
# EVOLVE-BLOCK-END
