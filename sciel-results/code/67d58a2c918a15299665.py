import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import time

N = 300

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # 1. Initialization using Halton sequence
    def halton(index, base):
        f = 1
        r = 0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def halton_sequence(count, dims):
        points = []
        for i in range(1, count + 1):
            points.append([halton(i, base) for base in range(2, dims + 2)])
        return np.array(points)

    # Generate points in [0,1]^2 and map to unit disk
    # Using polar coordinates for better distribution
    u = halton_sequence(N, 1)[:, 0]
    v = halton_sequence(N, 1)[:, 0] # Use different base or just reuse with offset? 
    # Better: Use Halton for radius and angle
    # r = sqrt(u) * R, theta = 2 * pi * v
    r = np.sqrt(u) * 0.95  # Keep slightly inside
    theta = 2 * np.pi * v
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    centers = np.column_stack((x, y))
    
    # 2. Force-Directed Relaxation with K-Nearest Neighbors
    K = 10
    max_iter = 5000
    alpha = 0.01  # Initial step size
    decay = 0.999
    
    for i in range(max_iter):
        if time.time() - start_time > timeout - 30: # Leave 30s for polish
            break
        
        tree = cKDTree(centers)
        # Find K nearest neighbors for all points
        # query_ball_tree or query
        # query returns (distances, indices)
        distances, indices = tree.query(centers, k=K + 1)
        
        forces = np.zeros_like(centers)
        
        # Vectorized force calculation is tricky with variable neighbors per point if using full arrays,
        # but K is small. We can loop over points. N=300 is small enough.
        for j in range(N):
            # Exclude self (index 0 in the k=K+1 result)
            nn_indices = indices[j, 1:]
            nn_dists = distances[j, 1:]
            
            # Avoid division by zero
            valid = nn_dists > 1e-10
            if not np.any(valid):
                continue
            
            nn_indices = nn_indices[valid]
            nn_dists = nn_dists[valid]
            
            # Repulsive force: sum of (x_j - x_i) / dist^2
            # Vector from i to neighbor
            diff = centers[nn_indices] - centers[j]
            # Invert for repulsion (push away from neighbor)
            # Force on i is proportional to (x_i - x_neighbor) / dist^2
            # Let's use F = (x_i - x_j) / dist^2
            force_vecs = -diff / (nn_dists[:, np.newaxis] ** 2)
            total_repulsive = np.sum(force_vecs, axis=0)
            
            # Centering force: pull towards origin
            centering_force = -centers[j] * 0.1
            
            forces[j] = total_repulsive + centering_force
        
        # Update positions
        centers += alpha * forces
        
        # Project back to unit disk if necessary (optional, centering force usually keeps them in)
        # But to be safe for the next step's distance calc, let's clip radius to 0.99
        radii = np.linalg.norm(centers, axis=1)
        mask = radii > 0.99
        if np.any(mask):
            centers[mask] = centers[mask] / radii[mask, np.newaxis] * 0.99
            
        alpha *= decay
        
    # 3. Local Optimization (Polish)
    # Objective: Maximize min( min_pair_dist/2, min_boundary_dist )
    # Minimize negative of that.
    
    def objective(centers_flat):
        c = centers_flat.reshape(-1, 2)
        # Calculate pairwise distances
        # This is O(N^2), N=300 -> 90k ops, fast enough for a few hundred evals
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distance
        radii = np.linalg.norm(c, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        
        # The score is min(min_pair_dist / 2, min_boundary_dist)
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        return -current_score

    # Use Powell method, it's derivative-free and good for this
    # Limit function evaluations to avoid timeout
    max_evals = 1000
    
    def time_check():
        return time.time() - start_time > timeout - 5

    try:
        res = minimize(objective, centers.flatten(), method='Powell', 
                       options={'maxiter': 100, 'maxfev': max_evals, 'disp': False})
        
        if res.success or res.fun < objective(centers.flatten()):
            centers = res.x.reshape(-1, 2)
            
    except Exception:
        pass
        
    # Final validation and cleanup
    # Ensure no points are outside
    radii = np.linalg.norm(centers, axis=1)
    outside = radii > 1.0
    if np.any(outside):
        centers[outside] = centers[outside] / radii[outside, np.newaxis] * 0.9999
        
    return centers
