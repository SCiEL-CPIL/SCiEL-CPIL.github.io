"""Improved packing algorithm for 50 equal spheres in a unit cube."""
import numpy as np
import time

N = 50

def _compute_radius(centers):
    """Compute the radius for a given set of centers."""
    # Boundary constraint: distance from each center to the nearest face
    # Cube is [-0.5, 0.5]^3
    boundary = 0.5 - np.abs(centers)
    min_boundary = boundary.min()
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(centers)
    min_dist = dists.min()
    
    return min(min_dist / 2.0, min_boundary)

def _energy(centers):
    """Energy to minimize: negative of the minimum distance (with boundary penalty)."""
    # Boundary constraint
    boundary = 0.5 - np.abs(centers)
    min_boundary = boundary.min()
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(centers)
    min_dist = dists.min()
    
    # We want to maximize min(min_dist/2, min_boundary)
    # Equivalent to minimizing -min(min_dist/2, min_boundary)
    # But for smoothness, use a soft version
    r = min(min_dist / 2.0, min_boundary)
    return -r

def _repulsion_energy(centers, k=10.0):
    """Soft repulsion energy that approximates maximizing minimum distance."""
    from scipy.spatial.distance import pdist
    dists = pdist(centers)
    # Large penalty for close pairs, small for far pairs
    # Use 1/d^k to create strong repulsion
    # Clip to avoid division by zero
    dists = np.maximum(dists, 1e-10)
    repulsion = np.sum(1.0 / (dists ** k))
    
    # Boundary penalty
    boundary = 0.5 - np.abs(centers)
    min_boundary = boundary.min()
    boundary_penalty = -min_boundary * 100.0
    
    return repulsion + boundary_penalty

def _gradient_step(centers, lr, k=10.0):
    """Compute gradient of repulsion energy and take a step."""
    from scipy.spatial.distance import cdist
    
    # Compute pairwise distances
    D = cdist(centers, centers)
    np.fill_diagonal(D, 1.0)  # Avoid self-interaction
    
    # Gradient of sum of 1/|ri-rj|^k with respect to ri
    # d/dr_i sum_j 1/|ri-rj|^k = -k * sum_j (ri-rj) / |ri-rj|^(k+2)
    # Note: this is for the repulsion term only
    
    grad = np.zeros_like(centers)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            grad[i] += -k * diff / (d ** (k + 2))
    
    # Boundary gradient: push centers away from walls
    # For each coordinate, if center is near a wall, push it inward
    boundary_grad = np.zeros_like(centers)
    for i in range(N):
        for d in range(3):
            c = centers[i, d]
            if c > 0.45:  # Near +0.5 wall
                boundary_grad[i, d] += -100.0
            elif c < -0.45:  # Near -0.5 wall
                boundary_grad[i, d] += 100.0
    
    return centers + lr * (grad + boundary_grad)

def construct():
    """Construct the optimal packing of 50 spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple initializations and optimization strategies
    np.random.seed(42)
    
    # Strategy 1: Jittered grid initialization
    for init_type in ['grid', 'random', 'face_centered']:
        if time.time() > deadline:
            break
        
        if init_type == 'grid':
            # Create a 4x4x4 grid (64 points), take first 50
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            centers = grid[:N].copy()
            # Add small random jitter
            centers += np.random.randn(N, 3) * 0.01
            
        elif init_type == 'random':
            centers = np.random.uniform(-0.45, 0.45, (N, 3))
            
        elif init_type == 'face_centered':
            # Start from a face-centered cubic-like arrangement
            # Use a 3x3x3 grid with some points shifted
            m = 3
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            centers = grid[:N].copy()
            # Shift alternate points to create FCC-like structure
            for i in range(N):
                if (i % 2 == 0):
                    centers[i] += np.array([0.1, 0.1, 0.1])
                else:
                    centers[i] += np.array([-0.1, -0.1, -0.1])
                # Keep within bounds
                centers[i] = np.clip(centers[i], -0.48, 0.48)
        
        # Optimize using gradient descent with adaptive learning rate
        lr = 0.001
        k = 8.0
        no_improve_count = 0
        best_local_radius = 0.0
        best_local_centers = centers.copy()
        
        for step in range(5000):
            if time.time() > deadline:
                break
            
            # Compute current radius
            current_radius = _compute_radius(centers)
            
            if current_radius > best_local_radius:
                best_local_radius = current_radius
                best_local_centers = centers.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Take gradient step
            centers = _gradient_step(centers, lr, k)
            
            # Keep centers within bounds
            centers = np.clip(centers, -0.49, 0.49)
            
            # Adaptive learning rate
            if no_improve_count > 100:
                lr *= 0.5
                if lr < 1e-6:
                    break
        
        # Update global best
        final_radius = _compute_radius(best_local_centers)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = best_local_centers.copy()
    
    # If we didn't find anything better than a basic grid, use that
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
