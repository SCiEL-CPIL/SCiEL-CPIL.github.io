import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the centers of 50 equal circles inside a unit circle.
    Uses a physics-based relaxation method (Jiggling/MD) to maximize the minimum distance.
    """
    np.random.seed(42)  # For reproducibility, though random init is usually fine
    
    # 1. Initialization: Start with a random distribution inside the circle
    # Using a spiral or random points. Random points with a few iterations of relaxation 
    # usually converge well. To be safe, we can start with a slightly dense grid 
    # or just random points. Let's use random points in the disk.
    
    # Generate random points in the unit disk
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    centers = np.column_stack((x, y))
    
    # 2. Simulation Parameters
    # We want to maximize r = min(min_pair_dist/2, min_dist_to_boundary)
    # This is equivalent to maximizing the minimum distance between any two centers (d_min)
    # and ensuring centers are within (1 - r).
    # Actually, the score is r. 
    # Constraint: |c_i - c_j| >= 2r and |c_i| <= 1 - r.
    # We can simulate this by treating the circles as repelling particles 
    # with a target separation.
    
    # A more direct approach: Gradient ascent on the minimum distance.
    # Let d_min(t) be the minimum distance between any pair of circles at time t.
    # Let b_min(t) be the minimum distance from any circle center to the boundary (1 - |c_i|).
    # We want to maximize min(d_min/2, b_min).
    
    # We will use an iterative "jiggling" algorithm:
    # 1. Calculate all pairwise distances and boundary distances.
    # 2. Find the current bottleneck distance (the smallest constraint violation or the target).
    # 3. Apply forces to increase this bottleneck.
    
    # Better: Use a simple optimization loop.
    # Objective: Maximize f(x) = min( min_{i<j} ||x_i - x_j||, min_i (1 - ||x_i||) * 2 )
    # Note: The radius r is half the min distance between centers, and also 1 - max ||x_i||.
    # So we want to maximize min( min_{i<j} ||x_i - x_j|| / 2, 1 - max_i ||x_i|| ).
    
    # Let's define a function to calculate the current "score" (radius)
    def calculate_radius(centers):
        # Pairwise distances
        # For N=50, O(N^2) is very fast (2500 ops)
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distance
        norms = np.linalg.norm(centers, axis=1)
        min_boundary_dist = 1.0 - np.max(norms)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    # Optimization Loop
    # We will use a stochastic gradient descent / jiggling approach.
    # Since the function is non-smooth (min), we can use subgradients or just 
    # move points in the direction that increases the minimum distance.
    
    # A robust method for circle packing:
    # 1. Start with a guess for radius r.
    # 2. Try to fit circles of radius r.
    # 3. If they overlap, push them apart.
    # 4. If they are all inside, we can try to increase r.
    
    # Let's implement a simple "push apart" solver.
    # We iterate:
    #   - Calculate current min distance d_min and min boundary dist b_min.
    #   - Current r = min(d_min/2, b_min).
    #   - Identify the "tight" constraints (pairs with dist ~ 2r, or circles with norm ~ 1-r).
    #   - Apply small perturbations to these tight circles to increase r.
    
    # Simpler and often effective: 
    # Use scipy.optimize.minimize to minimize the negative of the radius?
    # No, min is non-differentiable.
    
    # Let's use the "Jiggling" method described in Graham et al.
    # 1. Initialize positions.
    # 2. For many steps:
    #    a. Find the minimum distance between any pair (d_min) and min distance to boundary (b_min).
    #    b. Let r = min(d_min/2, b_min).
    #    c. Identify all pairs with distance < 2*r + epsilon and circles with 1-norm < r + epsilon.
    #    d. Apply repulsive forces to these specific circles.
    
    # Implementation details:
    # To be efficient and robust, we can use a fixed number of iterations with decreasing step size.
    
    centers = centers.copy()
    
    # Pre-calculate indices for pairs to speed up loops? N=50 is small, vectorization is better.
    
    # We will run a simulation for a fixed number of steps.
    # Step size needs to be tuned.
    
    # Let's try a simpler heuristic first: 
    # Random restarts + local optimization.
    
    best_centers = None
    best_radius = 0.0
    
    num_restarts = 20
    steps_per_restart = 1000
    
    for restart in range(num_restarts):
        # Initialize
        if restart == 0:
            # Keep the initial random points
            c = centers.copy()
        else:
            # New random initialization
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 1, N))
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            c = np.column_stack((x, y))
        
        # Relaxation loop
        # We use a "soft" repulsion potential to avoid singularities.
        # Potential U = sum_{i<j} (1/d_{ij})^k + sum_i (1/(1-|c_i|))^k ?
        # No, we want to maximize the MIN distance.
        # Maximizing the min distance is equivalent to minimizing the sum of "violations" if we fix a target radius.
        
        # Alternative: Gradient ascent on the minimum distance.
        # Subgradient of min(x, y) is 1 if x<y, 0 if x>y.
        # So we only move the circles involved in the minimum distance pair.
        
        for step in range(steps_per_restart):
            # Calculate distances
            dists = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            
            norms = np.linalg.norm(c, axis=1)
            min_boundary_dist = 1.0 - np.max(norms)
            
            current_r = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if current_r > best_radius:
                best_radius = current_r
                best_centers = c.copy()
                
            # Determine which constraints are active (limiting the radius)
            # We want to increase current_r.
            # If min_pair_dist/2 < min_boundary_dist, we need to increase pair distances.
            # Else, we need to move circles closer to center.
            
            # Learning rate decay
            lr = 0.05 * (0.99 ** (step / 100.0))
            
            # Gradient calculation
            grad = np.zeros_like(c)
            
            # Case 1: Pair distance is the bottleneck
            if min_pair_dist / 2.0 < min_boundary_dist:
                # Find all pairs with distance close to min_pair_dist
                # To be stable, we consider pairs within a small epsilon of the min
                epsilon = 0.01
                mask = dists < (min_pair_dist + epsilon)
                np.fill_diagonal(mask, False)
                
                # For each active pair, push them apart
                # Vectorized approach:
                # We need to sum up the gradients.
                # grad_i += (c_i - c_j) / dist_ij
                # grad_j += (c_j - c_i) / dist_ij
                
                # Iterate over active pairs (N is small, loop is fine)
                for i in range(N):
                    for j in range(i + 1, N):
                        if mask[i, j]:
                            d = c[i] - c[j]
                            dist = np.linalg.norm(d)
                            if dist > 1e-10:
                                # Unit vector
                                u = d / dist
                                # Repulsion force
                                grad[i] += u
                                grad[j] -= u
                                
            # Case 2: Boundary distance is the bottleneck
            else:
                # Find circles closest to boundary
                epsilon = 0.01
                # Dist to boundary is 1 - norm. Min dist means max norm.
                max_norm = np.max(norms)
                mask = norms > (max_norm - epsilon)
                
                for i in range(N):
                    if mask[i]:
                        # Move towards center
                        # Gradient of (1 - ||c_i||) wrt c_i is -c_i / ||c_i||
                        # We want to INCREASE 1 - ||c_i||, so we move in direction -c_i
                        norm_i = norms[i]
                        if norm_i > 1e-10:
                            grad[i] -= c[i] / norm_i
            
            # Apply gradient
            c += lr * grad
            
            # Ensure circles stay inside the container (hard constraint)
            # If a circle goes outside, pull it back.
            norms = np.linalg.norm(c, axis=1)
            outside = norms > 1.0
            if np.any(outside):
                # Project back to boundary
                c[outside] = c[outside] / norms[outside, np.newaxis] * 0.999

    return best_centers

# EVOLVE-BLOCK-END
