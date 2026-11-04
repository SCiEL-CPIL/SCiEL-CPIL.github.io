"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100
DEADLINE = 120.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START


def _fcc_initial(n):
    """Generate an FCC-like lattice initialization for n spheres in [-0.5, 0.5]^3."""
    # FCC lattice vectors
    # Conventional cubic cell of FCC has 4 atoms at:
    # (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # We'll generate points in a larger region and select the best n.
    
    # Generate FCC points in a range larger than the cube
    # The cube side is 1, centered at origin: [-0.5, 0.5]^3
    # We need enough points to pick n=100 with good spacing.
    
    # FCC nearest neighbor distance in conventional cell of side a is a/sqrt(2)
    # For 100 spheres in unit cube, volume per sphere = 1/100
    # If packed densely, typical spacing ~ (1/100)^(1/3) ≈ 0.215
    # FCC has 4 atoms per cell of side a, so density = 4/a^3
    # We want spacing ~0.215, so a/sqrt(2) ≈ 0.215 => a ≈ 0.304
    
    # Let's generate FCC points in a region slightly larger than the cube
    # and then keep those inside [-0.5, 0.5]^3, or adjust.
    
    # Better approach: generate a grid of FCC unit cells covering [-0.6, 0.6]^3
    # and collect all FCC points, then select 100 closest to center or best spaced.
    
    a = 0.35  # cubic cell side, gives nn distance a/sqrt(2) ≈ 0.247
    half_range = 0.6
    
    # Number of cells in each direction
    n_cells = int(np.ceil(2 * half_range / a))
    
    # Generate all FCC points
    points = []
    offsets = np.array([[0, 0, 0],
                        [0.5, 0.5, 0],
                        [0.5, 0, 0.5],
                        [0, 0.5, 0.5]])
    
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                base = np.array([ix * a - half_range,
                                 iy * a - half_range,
                                 iz * a - half_range])
                for off in offsets:
                    pt = base + off * a
                    points.append(pt)
    
    points = np.array(points)
    
    # Keep only points inside [-0.5, 0.5]^3 (with small margin)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points_inside = points[mask]
    
    if len(points_inside) >= n:
        # Select n points: use a greedy farthest-point sampling from center
        # Start from origin, repeatedly pick the point farthest from all selected
        selected = [np.zeros(3)]
        remaining = points_inside.copy()
        
        for _ in range(n - 1):
            if len(remaining) == 0:
                break
            # Compute min distance from each remaining point to selected set
            min_dists = np.array([
                np.min(np.sum((remaining - s) ** 2, axis=1))
                for s in selected
            ])
            min_dists = np.minimum.reduce(min_dists)
            best_idx = np.argmax(min_dists)
            selected.append(remaining[best_idx])
            remaining = np.delete(remaining, best_idx, axis=0)
        
        return np.array(selected[:n])
    else:
        # Fallback: use all inside points plus fill with random
        result = points_inside.copy()
        if len(result) < n:
            rng = np.random.default_rng(42)
            extra = rng.uniform(-0.45, 0.45, size=(n - len(result), 3))
            result = np.vstack([result, extra])
        return result[:n]


def _compute_radius(centres):
    """Compute the maximum common radius for the given centres."""
    # Min distance between pairs
    n = len(centres)
    min_pair_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centres[i] - centres[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    # Min distance to boundary
    min_bound = float('inf')
    for i in range(n):
        for k in range(3):
            d = 0.5 - abs(centres[i][k])
            if d < min_bound:
                min_bound = d
    
    return min(min_pair_dist, min_bound) / 2.0


def _soft_min_radius(centres, temp=0.01):
    """Smooth surrogate for the min radius using log-sum-exp."""
    n = len(centres)
    
    # Pair distances
    pair_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_dists.append(np.linalg.norm(centres[i] - centres[j]))
    pair_dists = np.array(pair_dists)
    
    # Boundary distances
    bound_dists = []
    for i in range(n):
        for k in range(3):
            bound_dists.append(0.5 - abs(centres[i][k]))
    bound_dists = np.array(bound_dists)
    
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min via log-sum-exp: soft_min(x) = -log(sum(exp(-x/temp)))/log(sum(exp(-x/temp)))
    # More precisely: soft_min ≈ -temp * log(sum(exp(-x/temp))) / (n_terms)
    # But we want a smooth approximation of min.
    # Use: soft_min = -temp * log( (1/m) * sum(exp(-x_i / temp)) )
    m = len(all_dists)
    scaled = -all_dists / temp
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)) / m)
    soft_min_val = -temp * log_sum
    
    return soft_min_val / 2.0


def _grad_soft_min_radius(centres, temp=0.01):
    """Gradient of soft min radius w.r.t. centres."""
    n = len(centres)
    grad = np.zeros_like(centres)
    
    # Build list of all distances and their gradients
    dists = []
    grads = []  # each is (n, 3)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = centres[i] - centres[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                dist = 1e-12
            grad_ij = d / dist
            g = np.zeros_like(centres)
            g[i] = grad_ij
            g[j] = -grad_ij
            dists.append(dist)
            grads.append(g)
    
    for i in range(n):
        for k in range(3):
            # distance to boundary: 0.5 - |c_ik|
            val = 0.5 - abs(centres[i][k])
            if val < 1e-12:
                val = 1e-12
            # gradient of (0.5 - |c_ik|) w.r.t. c_ik is -sign(c_ik)
            g = np.zeros_like(centres)
            g[i, k] = -np.sign(centres[i, k]) if centres[i, k] != 0 else -1.0
            dists.append(val)
            grads.append(g)
    
    dists = np.array(dists)
    grads = np.array(grads)  # shape (m, n, 3)
    
    m = len(dists)
    scaled = -dists / temp
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / (m * np.sum(exp_scaled))
    
    # Gradient of soft_min = -temp * log(mean(exp(-d/temp)))
    # d/dd_j [ -temp * log(mean(exp(-d_i/temp))) ]
    # = -temp * (1/mean) * sum_i (1/m) * exp(-d_i/temp) * (-1/temp) * d(d_j)/d(centre)
    # = sum_i weights_i * d(d_j)/d(centre)
    # Wait, let me re-derive:
    # f = -temp * log( (1/m) sum exp(-d_i/temp) )
    # df/dx = -temp * (1 / ((1/m) sum exp(-d_i/temp))) * sum_i (1/m) exp(-d_i/temp) * (-1/temp) * dd_i/dx
    # = -temp * (m / sum exp(-d_i/temp)) * (1/m) * sum_i exp(-d_i/temp) * (-1/temp) * dd_i/dx
    # = (1 / sum exp(-d_i/temp)) * sum_i exp(-d_i/temp) * dd_i/dx
    # = sum_i (exp(-d_i/temp) / sum exp(-d_i/temp)) * dd_i/dx
    
    # So the weights are softmax(-d/temp), and grad = sum_i w_i * grad(d_i)
    w = exp_scaled / np.sum(exp_scaled)
    
    for idx in range(m):
        grad += w[idx] * grads[idx]
    
    return grad / 2.0  # because radius = min_dist / 2


def construct():
    start_time = time.time()
    
    # Initialize with FCC-like configuration
    centres = _fcc_initial(N)
    
    # Compute initial radius
    current_radius = _compute_radius(centres)
    best_centres = centres.copy()
    best_radius = current_radius
    
    # Optimization loop with soft-min objective
    # Use gradient ascent on soft-min radius
    
    temp = 0.02  # temperature for soft-min
    lr = 0.005   # learning rate
    n_iter = 5000
    
    for it in range(n_iter):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:  # safety margin
            break
        
        # Compute gradient
        grad = _grad_soft_min_radius(centres, temp)
        
        # Gradient ascent
        centres_new = centres + lr * grad
        
        # Project back into cube
        centres_new = np.clip(centres_new, -0.4999, 0.4999)
        
        # Compute actual radius
        new_radius = _compute_radius(centres_new)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centres = centres_new.copy()
        
        # Adaptive temperature: decrease over time
        temp = 0.02 * (0.999 ** it)
        
        # Adaptive learning rate
        lr = 0.005 * (0.9995 ** it)
        
        centres = centres_new
    
    # Final polish: try to improve with smaller steps
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 15:
        # Do a fine-tuning pass
        lr_fine = 0.001
        temp_fine = 0.005
        for it in range(2000):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 5:
                break
            
            grad = _grad_soft_min_radius(best_centres, temp_fine)
            centres_new = best_centres + lr_fine * grad
            centres_new = np.clip(centres_new, -0.4999, 0.4999)
            
            new_radius = _compute_radius(centres_new)
            if new_radius > best_radius:
                best_radius = new_radius
                best_centres = centres_new.copy()
            
            lr_fine *= 0.999
            temp_fine *= 0.999
    
    # Ensure validity: all centres strictly inside cube
    best_centres = np.clip(best_centres, -0.49999, 0.49999)
    
    return best_centres


# EVOLVE-BLOCK-END
