"""Improved packing via gradient-based optimization with soft-min surrogate."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_init(n):
    """Generate initial hexagonal lattice points within unit disk."""
    # Estimate spacing for n points in unit disk
    # Area per point ~ pi/n, hex packing density ~ 0.9069
    # spacing s: s^2 * sqrt(3)/2 * n ≈ pi => s ≈ sqrt(2*pi/(n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate hex lattice points
    points = []
    # Range to cover the disk
    max_r = 1.0
    n_range = int(np.ceil(max_r / s)) + 2
    
    for j in range(-n_range, n_range + 1):
        for i in range(-n_range, n_range + 1):
            x = i * s + (j % 2) * s * 0.5
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y))
    
    points = np.array(points)
    
    # If we have more than n, select the n closest to center (or random subset)
    if len(points) > n:
        # Sort by distance from center, take closest n
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists, kind='stable')
        points = points[idx[:n]]
    elif len(points) < n:
        # Add random points to fill
        while len(points) < n:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 1.0)
            points = np.vstack([points, [[r * np.cos(angle), r * np.sin(angle)]]])
    
    return points[:n]


def _compute_radius(pts):
    """Compute the packing radius for a set of points."""
    n = len(pts)
    # Minimum distance between any pair
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                min_dist = d
    
    # Minimum distance from any point to boundary
    min_boundary = np.inf
    for i in range(n):
        r = np.linalg.norm(pts[i])
        if 1.0 - r < min_boundary:
            min_boundary = 1.0 - r
    
    return min(min_dist / 2.0, min_boundary)


def _soft_min_logsumexp(values, beta):
    """Soft minimum using log-sum-exp trick."""
    # For soft-min, we use -log-sum-exp(-beta * values) / beta
    neg_beta_vals = -beta * values
    max_val = np.max(neg_beta_vals)
    if max_val > 700:
        max_val = 700
    elif max_val < -700:
        max_val = -700
    exp_vals = np.exp(neg_beta_vals - max_val)
    log_sum = np.log(np.sum(exp_vals)) + max_val
    return -log_sum / beta


def _residuals(pts, beta):
    """Compute soft-min residuals for all pairs and boundary distances."""
    n = len(pts)
    residuals = []
    
    # Pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            residuals.append(d)
    
    # Boundary distances
    for i in range(n):
        r = np.linalg.norm(pts[i])
        residuals.append(1.0 - r)
    
    residuals = np.array(residuals)
    return _soft_min_logsumexp(residuals, beta)


def _compute_gradients(pts, beta):
    """Compute gradients of the soft-min objective with respect to point positions."""
    n = len(pts)
    grads = np.zeros_like(pts)
    
    # We need to compute the gradient of the soft-min
    # soft_min = -1/beta * log(sum(exp(-beta * r_k)))
    # d(soft_min)/d(r_k) = exp(-beta * r_k) / sum(exp(-beta * r_k))
    
    # First compute all residuals
    residuals = []
    pair_indices = []  # (i, j) for pairs
    boundary_indices = []  # i for boundary
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            residuals.append(d)
            pair_indices.append((i, j))
    
    for i in range(n):
        r = np.linalg.norm(pts[i])
        residuals.append(1.0 - r)
        boundary_indices.append(i)
    
    residuals = np.array(residuals)
    
    # Compute weights
    neg_beta_vals = -beta * residuals
    max_val = np.max(neg_beta_vals)
    exp_vals = np.exp(neg_beta_vals - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Now compute gradients
    # For pair (i,j): r = |p_i - p_j|
    # d(r)/d(p_i) = (p_i - p_j) / r
    # d(r)/d(p_j) = (p_j - p_i) / r
    
    for k, ((i, j)) in enumerate(pair_indices):
        w = weights[k]
        diff = pts[i] - pts[j]
        d = np.linalg.norm(diff)
        if d > 1e-12:
            grad_vec = diff / d
            grads[i] += w * grad_vec
            grads[j] -= w * grad_vec
    
    # For boundary: r = 1 - |p_i|
    # d(r)/d(p_i) = -p_i / |p_i|
    for k, i in enumerate(boundary_indices):
        w = weights[k]
        r = np.linalg.norm(pts[i])
        if r > 1e-12:
            grads[i] -= w * pts[i] / r
    
    return grads


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Initialize with hexagonal lattice
    np.random.seed(42)
    pts = _hex_init(N)
    
    # Phase 1: Coarse optimization with moderate beta
    beta_start = 50.0
    beta_end = 500.0
    
    # Adaptive gradient descent
    lr = 0.01
    n_iter = 0
    
    # Phase 1: Coarse search
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 20.0:
            break
        
        for it in range(2000):
            if time.time() > deadline - 10.0:
                break
            
            grads = _compute_gradients(pts, beta)
            
            # Normalize gradients
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-12:
                pts += lr * grads / grad_norm
            
            # Project back to unit disk
            for i in range(N):
                r = np.linalg.norm(pts[i])
                if r > 1.0:
                    pts[i] = pts[i] / r
            
            n_iter += 1
    
    # Phase 2: Fine optimization with high beta
    if time.time() < deadline - 15.0:
        for beta in [500.0, 1000.0, 2000.0]:
            if time.time() > deadline - 10.0:
                break
            
            lr_fine = 0.005
            for it in range(3000):
                if time.time() > deadline - 5.0:
                    break
                
                grads = _compute_gradients(pts, beta)
                
                grad_norm = np.linalg.norm(grads)
                if grad_norm > 1e-12:
                    pts += lr_fine * grads / grad_norm
                
                # Project back to unit disk
                for i in range(N):
                    r = np.linalg.norm(pts[i])
                    if r > 1.0:
                        pts[i] = pts[i] / r
            
            n_iter += 1
    
    # Final polish: very high beta
    if time.time() < deadline - 5.0:
        beta_polish = 5000.0
        lr_polish = 0.001
        for it in range(2000):
            if time.time() > deadline - 2.0:
                break
            
            grads = _compute_gradients(pts, beta_polish)
            
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-12:
                pts += lr_polish * grads / grad_norm
            
            # Project back to unit disk
            for i in range(N):
                r = np.linalg.norm(pts[i])
                if r > 1.0:
                    pts[i] = pts[i] / r
        
        n_iter += 1
    
    # Ensure all points are strictly inside the unit disk
    for i in range(N):
        r = np.linalg.norm(pts[i])
        if r >= 1.0:
            pts[i] = pts[i] * 0.9999 / r if r > 0 else np.array([0.0, 0.0])
    
    return pts
# EVOLVE-BLOCK-END
