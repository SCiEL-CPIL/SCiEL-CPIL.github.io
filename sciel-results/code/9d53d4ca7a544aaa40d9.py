"""Rewritten program: smooth potential-based covering radius minimization."""
import numpy as np
import time

N = 130


def _covering_radius_deg(pts):
    """Compute covering radius in degrees for points on unit sphere."""
    # Use the fact that covering radius = max over sphere of min angular distance
    # We can compute this by finding the largest empty cap.
    # Practical approach: sample the sphere densely and compute min distance to nearest point.
    # For better accuracy, use the convex hull approach or dense sampling.
    
    # Dense sampling approach: use a fine grid on the sphere
    # Number of samples should be sufficient to resolve the covering radius to ~0.01 degrees
    n_theta = 2000
    n_phi = 4000
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    # Reshape for memory efficiency - process in chunks
    max_ang = 0.0
    pts_flat = pts.reshape(-1, 3)
    
    # Process in chunks to manage memory
    chunk_size = 500
    for i in range(0, n_theta, chunk_size):
        i_end = min(i + chunk_size, n_theta)
        th = TH[i:i_end].ravel()
        ph = PH[i:i_end].ravel()
        # Convert to cartesian
        x = np.sin(th) * np.cos(ph)
        y = np.sin(th) * np.sin(ph)
        z = np.cos(th)
        # Compute dot products with all points
        # This is (n_samples * 3) x (3 * N) -> n_samples x N
        # To save memory, do in sub-chunks
        sub_chunk = 20000
        for j in range(0, len(x), sub_chunk):
            j_end = min(j + sub_chunk, len(x))
            xx = x[j:j_end].reshape(-1, 3)
            dots = xx @ pts_flat.T  # (n_sub, N)
            max_dots = np.max(dots, axis=1)
            # Angular distance = arccos(max_dot)
            # Clip for numerical safety
            max_dots = np.clip(max_dots, -1.0, 1.0)
            ang = np.arccos(max_dots)
            local_max = np.max(ang)
            if local_max > max_ang:
                max_ang = local_max
    
    return np.degrees(max_ang)


def _smooth_objective(pts, sigma=0.3):
    """
    Smooth approximation of covering radius using potential.
    V(y) = -sigma * log(sum_i exp(-||y - p_i||^2 / sigma^2))
    We want to minimize max_y V(y), which is equivalent to minimizing the covering radius.
    Actually, let's use a simpler approach: minimize the max of the distance to nearest point.
    
    Alternative: use the sum of exponentials as a proxy.
    """
    # For each point on a grid, compute the distance to nearest configuration point
    # Use a moderate grid
    n_theta = 400
    n_phi = 800
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Compute distances
    # (n_grid, 3) @ (3, N) -> (n_grid, N)
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    ang_dists = np.arccos(max_dots)
    
    # Use a smooth max approximation
    # log-sum-exp style
    temp = 0.05
    smooth_max = temp * np.log(np.sum(np.exp(ang_dists / temp)))
    return smooth_max


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Also try a few other initializations and pick the best
    initializations = [pts]
    
    # Random initialization
    rng = np.random.RandomState(42)
    for _ in range(3):
        random_pts = rng.randn(N, 3)
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        initializations.append(random_pts)
    
    # Pick best initial
    best_pts = None
    best_obj = np.inf
    for init in initializations:
        obj = _smooth_objective(init)
        if obj < best_obj:
            best_obj = obj
            best_pts = init.copy()
    
    pts = best_pts
    
    # Gradient descent on the smooth objective
    # We need the gradient of the smooth max with respect to each point p_k
    
    n_theta = 400
    n_phi = 800
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    gx = np.sin(TH) * np.cos(PH)
    gy = np.sin(TH) * np.sin(PH)
    gz = np.cos(TH)
    grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    
    temp = 0.05
    lr = 0.01
    n_iter = 5000
    
    for it in range(n_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Compute dots: (n_grid, N)
        dots = grid @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        ang_dists = np.arccos(max_dots)
        
        # Smooth max: f = temp * log(sum(exp(ang/temp)))
        # Gradient w.r.t. each p_k:
        # df/dp_k = temp * (1/sum_exp) * sum_j exp(ang_j/temp) * d(ang_j)/dp_k
        # where ang_j = arccos(max_i dot(y_j, p_i))
        # d(ang_j)/dp_k = -1/sqrt(1 - dot_jk^2) * y_j  (when p_k is the nearest)
        
        exp_vals = np.exp(ang_dists / temp)  # (n_grid,)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp  # (n_grid,)
        
        # For each grid point, find the nearest point index
        nearest_idx = np.argmax(dots, axis=1)  # (n_grid,)
        
        # Compute gradient for each point
        grad = np.zeros_like(pts)
        for k in range(N):
            mask = (nearest_idx == k)
            if not np.any(mask):
                continue
            y_j = grid[mask]  # (n_mask, 3)
            w_j = weights[mask]  # (n_mask,)
            dots_jk = dots[mask, k]  # (n_mask,)
            # d(ang)/dp_k = -y_j / sqrt(1 - dot^2)
            denom = np.sqrt(np.maximum(1.0 - dots_jk ** 2, 1e-12))
            d_ang_dp = -y_j / denom[:, np.newaxis]  # (n_mask, 3)
            # Weighted sum
            grad[k] = np.sum(w_j[:, np.newaxis] * d_ang_dp, axis=0)
        
        # Apply gradient
        pts -= lr * grad
        
        # Renormalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Adaptive learning rate
        if it % 100 == 0:
            obj = _smooth_objective(pts)
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
            # Reduce learning rate slowly
            lr *= 0.99
    
    # Final polishing with finer grid
    # Use the best points found
    pts = best_pts
    
    # Do a few more iterations with smaller learning rate and finer grid
    n_theta_fine = 600
    n_phi_fine = 1200
    theta_f = np.linspace(0, np.pi, n_theta_fine)
    phi_f = np.linspace(0, 2 * np.pi, n_phi_fine, endpoint=False)
    TH_f, PH_f = np.meshgrid(theta_f, phi_f, indexing='ij')
    gx_f = np.sin(TH_f) * np.cos(PH_f)
    gy_f = np.sin(TH_f) * np.sin(PH_f)
    gz_f = np.cos(TH_f)
    grid_f = np.column_stack([gx_f.ravel(), gy_f.ravel(), gz_f.ravel()])
    
    temp_f = 0.03
    lr_f = 0.005
    n_iter_f = 2000
    
    for it in range(n_iter_f):
        if time.time() > deadline - 5.0:
            break
        
        dots_f = grid_f @ pts.T
        dots_f = np.clip(dots_f, -1.0, 1.0)
        max_dots_f = np.max(dots_f, axis=1)
        ang_dists_f = np.arccos(max_dots_f)
        
        exp_vals_f = np.exp(ang_dists_f / temp_f)
        sum_exp_f = np.sum(exp_vals_f)
        weights_f = exp_vals_f / sum_exp_f
        
        nearest_idx_f = np.argmax(dots_f, axis=1)
        
        grad_f = np.zeros_like(pts)
        for k in range(N):
            mask_f = (nearest_idx_f == k)
            if not np.any(mask_f):
                continue
            y_jf = grid_f[mask_f]
            w_jf = weights_f[mask_f]
            dots_jkf = dots_f[mask_f, k]
            denom_f = np.sqrt(np.maximum(1.0 - dots_jkf ** 2, 1e-12))
            d_ang_dp_f = -y_jf / denom_f[:, np.newaxis]
            grad_f[k] = np.sum(w_jf[:, np.newaxis] * d_ang_dp_f, axis=0)
        
        pts -= lr_f * grad_f
        norms_f = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms_f
    
    # Ensure all points are on unit sphere
    norms_final = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms_final
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-6:
                # Perturb
                pts[j] += rng.randn(3) * 1e-4
                pts[j] /= np.linalg.norm(pts[j])
    
    # Final check: make sure it's not all in a hemisphere
    # Check by seeing if there exists a plane through origin with all points on one side
    # Simple check: the centroid should not be too far from origin for a well-distributed set
    centroid = np.mean(pts, axis=0)
    if np.linalg.norm(centroid) > 0.5:
        # Might be in a hemisphere, add some antipodal points
        for i in range(0, N, 2):
            if rng.rand() < 0.3:
                pts[i] = -pts[i]
                pts[i] /= np.linalg.norm(pts[i])
    
    return pts
