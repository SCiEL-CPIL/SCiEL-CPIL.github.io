import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # 1. Initial Configuration: Hexagonal Lattice
    # Approximate optimal spacing for N=300 in unit circle
    # Area = pi. N circles area = N * pi * r^2.
    # Packing density ~ 0.9.
    # N * pi * r^2 / 0.9 <= pi  => r <= sqrt(0.9/N) ~ 0.0547
    # Spacing d = 2r ~ 0.109
    spacing = 0.105 
    points = []
    # Generate lattice points
    # Hex lattice:
    # x = i * spacing + (j % 2) * spacing / 2
    # y = j * spacing * sqrt(3) / 2
    max_coord = 1.0
    k = int(max_coord / spacing) + 2
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = i * spacing + (j % 2) * spacing / 2
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N, keep the N closest to the origin
    if len(points) > N:
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii)[:N]
        points = points[idx]
    elif len(points) < N:
        # If not enough, add random points near the boundary? 
        # Or just resize? The seed logic ensures enough. 
        # For safety, add random points if needed.
        while len(points) < N:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 1.0 - spacing / 2.0)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
    
    # Add small noise to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    
    # Ensure initial validity (clip to circle)
    norms = np.linalg.norm(points, axis=1)
    points[norms > 1.0] *= (1.0 / norms[norms > 1.0])[:, np.newaxis]

    best_points = points.copy()
    best_score = -np.inf
    
    # Calculate initial score
    def calc_score(pts):
        # Min pairwise distance / 2
        if len(pts) < 2:
            return np.min(1.0 - np.linalg.norm(pts, axis=1))
        
        # Vectorized pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        
        # Min boundary distance
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        min_bound_dist = np.min(boundary_dists)
        
        return min(min_pair_dist, min_bound_dist)
    
    current_score = calc_score(points)
    if current_score > best_score:
        best_score = current_score
        best_points = points.copy()
    
    # Gradient Ascent Loop
    # Soft min parameter
    eps = 0.01 
    step_size = 0.005
    anneal_rate = 0.999
    
    # We will run as many iterations as possible before deadline
    it = 0
    while time.time() < deadline:
        # Check time more frequently in later stages
        if it % 50 == 0 and time.time() > deadline - 5.0:
            break
            
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        norms = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - norms
        
        # Combine into single array for soft min
        # Pairs: N*(N-1)/2. Bounds: N.
        # To avoid huge memory for N=300 (45k pairs), we can compute gradients directly.
        
        # Gradients for pairs
        # grad_dist[i,j] = (points[i] - points[j]) / dists[i,j]
        # Only for dists > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vecs = diff / dists[..., np.newaxis]
        
        # Weights for pairs
        # w_ij = exp(-d_ij / eps) / Z
        # Z = sum exp(-d_ij / eps) + sum exp(-b_i / eps)
        
        pair_weights_raw = np.exp(-dists / eps)
        np.fill_diagonal(pair_weights_raw, 0.0)
        
        bound_weights_raw = np.exp(-bound_dists / eps)
        
        Z = np.sum(pair_weights_raw) + np.sum(bound_weights_raw)
        
        if Z == 0:
            break
            
        pair_weights = pair_weights_raw / Z
        bound_weights = bound_weights_raw / Z
        
        # Gradient of J = -eps * log(Z) w.r.t points
        # dJ/dp_i = sum_j pair_weights[i,j] * (p_i - p_j)/d_ij * (-1) ? 
        # Let's re-derive.
        # J = -eps * log( sum exp(-d/eps) )
        # dJ/dp = -eps * (1/Z) * sum (exp(-d/eps) * (-1/eps) * dd/dp)
        # dJ/dp = (1/Z) * sum (exp(-d/eps) * dd/dp)
        # dd_ij/dp_i = (p_i - p_j)/d_ij
        # So grad_pair_i = sum_j w_ij * (p_i - p_j)/d_ij
        # 
        # For boundary:
        # dJ/db_i term: w_i * db_i/dp_i
        # b_i = 1 - |p_i|
        # db_i/dp_i = -p_i/|p_i|
        # So grad_bound_i = w_i * (-p_i/|p_i|)
        
        # Compute pair gradients
        # unit_vecs shape (N, N, 2)
        # pair_weights shape (N, N)
        # grad_i = sum_j pair_weights[i,j] * unit_vecs[i,j]
        grad_pairs = np.sum(pair_weights[..., np.newaxis] * unit_vecs, axis=1)
        
        # Compute boundary gradients
        # unit_norms = points / norms
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_norms = points / norms[:, np.newaxis]
            unit_norms[norms == 0] = 0.0
            
        grad_bounds = -bound_weights[:, np.newaxis] * unit_norms
        
        total_grad = grad_pairs + grad_bounds
        
        # Update
        points = points + step_size * total_grad
        
        # Anneal
        eps *= anneal_rate
        # Adjust step size slightly?
        # step_size *= 0.999
        
        # Check validity and score
        # To save time, we can check score every 10 iterations
        if it % 10 == 0:
            current_score = calc_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        
        it += 1
        
    # Final polish: check best_points validity
    # Ensure no points are outside
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms > 1.0):
         best_points[norms > 1.0] *= (1.0 / norms[norms > 1.0])[:, np.newaxis]
         
    return best_points
