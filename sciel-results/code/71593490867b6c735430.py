"""Rewritten packing program using vectorized force-directed optimization with soft constraints."""
import numpy as np
import time

N = 300


def _compute_score(points):
    """Compute the packing radius score for a set of points."""
    # Distance to boundary
    norms = np.sqrt(np.sum(points ** 2, axis=1))
    boundary = 1.0 - norms
    min_boundary = np.min(boundary)
    
    # Pairwise distances
    # Use a chunked approach to avoid memory issues
    min_pair = np.inf
    n = len(points)
    chunk = 200
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        # points[i:end] vs points[i+1:]
        if end <= i + 1:
            continue
        # Compute distances from points[i:end] to all points after them
        for j in range(i, end):
            if j + 1 >= n:
                break
            diffs = points[j + 1:] - points[j]
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            local_min = np.min(dists)
            if local_min < min_pair:
                min_pair = local_min
    
    return min(min_boundary, min_pair / 2.0)


def _init_hexagonal(n):
    """Generate an initial hexagonal lattice configuration clipped to unit disk."""
    # Estimate spacing for n circles in unit disk
    # Area of unit disk = pi, area per circle ~ pi/n, so radius ~ 1/sqrt(n)
    # Hexagonal packing: each circle has area ~ (sqrt(3)/2) * (2r)^2 = 2*sqrt(3)*r^2
    # So r ~ sqrt(pi / (n * 2 * sqrt(3)))
    r_est = np.sqrt(np.pi / (n * 2 * np.sqrt(3)))
    spacing = 2 * r_est
    
    # Generate hexagonal lattice points
    points = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3) / 2
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            if x * x + y * y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        # Fallback: random points in disk
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    else:
        # Select n points with largest distance from center (to fill the disk)
        dists = np.sqrt(np.sum(points ** 2, axis=1))
        idx = np.argsort(dists, kind='stable')[:n]
        points = points[idx]
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with hexagonal lattice
    points = _init_hexagonal(N)
    current_score = _compute_score(points)
    
    # Force-directed optimization with soft constraints
    # Objective: maximize min(pairwise_dist/2, 1 - norm)
    # We'll use a gradient-based approach with soft min
    
    # Parameters
    lr = 0.001
    decay = 0.999
    soft_beta = 100.0  # Temperature for soft min
    
    # Precompute for efficiency
    n = N
    
    def _soft_min(values, beta):
        """Compute soft minimum using log-sum-exp trick."""
        min_val = np.min(values)
        # For numerical stability
        exp_vals = np.exp(-beta * (values - min_val))
        log_sum = np.log(np.sum(exp_vals))
        return min_val + np.log(exp_vals.shape[0]) / beta - log_sum / beta
    
    def _compute_gradient(points, beta):
        """Compute gradient of the soft-min objective."""
        n = len(points)
        grad = np.zeros_like(points)
        
        # Boundary term: maximize (1 - ||p_i||)
        norms = np.sqrt(np.sum(points ** 2, axis=1))
        boundary_vals = 1.0 - norms
        # Gradient of boundary term
        for i in range(n):
            if norms[i] > 1e-10:
                grad[i] += points[i] / norms[i]  # Gradient of (1 - ||p_i||) is -p_i/||p_i||, so for max we want +p_i/||p_i||
        
        # Pairwise term: maximize min_{i<j} ||p_i - p_j||/2
        # Use soft min over all pairs
        pair_vals = []
        pair_indices = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[j] - points[i]
                dist = np.sqrt(np.sum(diff ** 2))
                pair_vals.append(dist / 2.0)
                pair_indices.append((i, j))
        
        pair_vals = np.array(pair_vals)
        min_pair = np.min(pair_vals)
        
        # Soft min weights
        exp_vals = np.exp(-beta * (pair_vals - min_pair))
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient contribution from pairwise term
        for idx, (i, j) in enumerate(pair_indices):
            if weights[idx] < 1e-10:
                continue
            diff = points[j] - points[i]
            dist = np.sqrt(np.sum(diff ** 2))
            if dist < 1e-10:
                continue
            # Gradient of ||p_i - p_j||/2 w.r.t. p_i is -diff/(2*dist), w.r.t. p_j is diff/(2*dist)
            # For max, we want to increase the minimum, so gradient direction
            w = weights[idx]
            grad[i] += w * (-diff / dist)  # -diff/(2*dist) * 2 (since we have /2 in objective)
            grad[j] += w * (diff / dist)
        
        return grad
    
    # Optimize
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Compute gradient
        grad = _compute_gradient(points, soft_beta)
        
        # Update
        points += lr * grad
        
        # Project back inside unit disk
        norms = np.sqrt(np.sum(points ** 2, axis=1))
        mask = norms > 1.0
        if np.any(mask):
            points[mask] /= norms[mask, np.newaxis]
        
        # Decay learning rate
        lr *= decay
        
        # Compute score periodically
        if iteration % 100 == 0:
            score = _compute_score(points)
            if score > current_score:
                current_score = score
                best_points = points.copy()
    
    # Final score
    final_score = _compute_score(points)
    if final_score > current_score:
        return points
    else:
        return best_points if 'best_points' in dir() else points
