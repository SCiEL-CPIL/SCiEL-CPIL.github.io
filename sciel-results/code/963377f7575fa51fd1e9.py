"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def _compute_radius(points):
    """
    Computes the maximum radius for a set of points.
    r = min(min_{i<j} dist(i,j)/2, min_i (1 - |p_i|))
    """
    if len(points) == 0:
        return 0.0
    
    # Distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_boundary)
    
    # Pairwise distances
    # For N=300, 300x300 matrix is 90k elements, very fast
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Initial configuration: Random points inside unit circle
    # Using polar coordinates for uniform distribution
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Track best
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    # Annealing parameters
    # We want to maximize r. 
    # Let's simulate particles repelling each other and being pushed outwards?
    # No, standard packing: particles repel each other, and are constrained to stay inside.
    # If we just repel them, they will move to the boundary and spread out.
    # The "pressure" of the repulsion determines the effective radius.
    # Actually, a simpler approach: 
    # Start with a small radius r0.
    # If circles overlap, push them apart.
    # If circles are inside, allow them to expand?
    
    # Alternative: Simulate "hard spheres" with a slight repulsive potential.
    # Or, use the "move to reduce overlap" heuristic.
    
    # Let's use a gradient-free local search combined with random restarts or perturbations.
    # But N=300 is large for pure local search to escape local minima.
    
    # Strategy: 
    # 1. Start with hexagonal lattice (good initial guess).
    # 2. Apply iterative relaxation: 
    #    - Calculate current r.
    #    - If overlapping (pairwise dist < 2r_current_estimate), push apart.
    #    - If too close to boundary, push in.
    #    - This is essentially solving for a packing where all circles are tangent.
    
    # Let's restart with the hexagonal lattice from the seed, as it's a good start.
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
        
    best_points = points.copy()
    best_r = _compute_radius(points)
    
    # Now optimize
    # We will use a simple force-directed method.
    # Force on i: 
    #   Repulsion from j if dist < target_dist. 
    #   What is target_dist? 
    #   We don't know the optimal r yet. 
    #   We can treat r as a variable. 
    #   Let's try to maximize r.
    #   
    #   Approach: 
    #   1. Estimate current r.
    #   2. Add a small delta_r.
    #   3. Try to adjust positions to fit 2*(r+delta_r).
    #   4. If successful (all constraints satisfied), keep new r.
    #   5. If not, revert and try smaller delta_r or different direction.
    
    # This is a "bump and check" method.
    
    current_points = best_points.copy()
    current_r = best_r
    
    # Pre-calculate indices for efficiency? No, N=300 is small enough for O(N^2) per step.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Try to increase radius
        # The potential gain is small. 
        # Let's define a step size for r.
        # Start with a reasonable step.
        dr = 1e-5
        
        # Proposed new radius
        new_r = current_r + dr
        
        # Target distance between centers
        target_d = 2 * new_r
        
        # Apply forces to satisfy constraints:
        # 1. Pairwise distance >= target_d
        # 2. Distance from origin <= 1 - new_r
        
        # We will do a few iterations of relaxation for this new_r
        # If we can satisfy constraints, we accept new_r.
        
        # Relaxation loop for fixed new_r
        # Use simple gradient descent on the "violation" function
        # V = sum(max(0, target_d - dist_ij)^2 + sum(max(0, (1-new_r) - dist_i)^2)
        
        # Instead of full gradient, use pairwise correction.
        # For each pair (i,j) with dist < target_d:
        #    push i and j apart along vector (p_i - p_j) by (target_d - dist)/2
        # For each i with |p_i| > 1 - new_r:
        #    push i towards origin by (|p_i| - (1-new_r))
        
        # This is the "Position Based Dynamics" (PBD) or simple constraint satisfaction.
        
        # We need to be careful not to oscillate.
        # Let's do 10 relaxation steps per dr increase.
        
        success = False
        for _ in range(10):
            # Check if current config satisfies new_r
            # If yes, success
            # But we need to adjust positions first.
            
            # Apply pairwise repulsions
            # Vectorized version for speed
            diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            
            # Mask for pairs that are too close
            # dists < target_d
            # We want to move them apart.
            # Displacement for i due to j: (diff_ij / dists_ij) * (target_d - dists_ij) / 2
            # Note: diff_ij = p_i - p_j. So we add to p_i in direction p_i-p_j.
            
            # Avoid division by zero
            mask = (dists < target_d) & (dists > 1e-8)
            # Also exclude diagonal
            np.fill_diagonal(mask, False)
            
            # Compute forces
            # Force matrix F[i,j] is the vector to add to p_i due to constraint with j.
            # F[i,j] = (p_i - p_j) / dist * (target_d - dist) / 2
            
            # Vectorized calculation
            # inv_dists = np.zeros_like(dists)
            # np.divide(1.0, dists, out=inv_dists, where=dists>1e-8)
            
            # This might be memory heavy if not careful, but 300x300 is fine.
            
            # Let's just loop for clarity and stability, N=300 is small.
            # 300*300/2 = 45k pairs. 10 iterations * 45k = 450k operations. Very fast.
            
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = current_points[i] - current_points[j]
                    d = np.linalg.norm(d_vec)
                    if d < target_d and d > 1e-8:
                        # Push apart
                        correction = (target_d - d) / d * 0.5
                        current_points[i] += d_vec * correction
                        current_points[j] -= d_vec * correction
            
            # Boundary constraint
            for i in range(N):
                r_i = np.linalg.norm(current_points[i])
                max_r = 1.0 - new_r
                if r_i > max_r and r_i > 1e-8:
                    # Push in
                    correction = (r_i - max_r) / r_i
                    current_points[i] -= current_points[i] * correction
            
            # Check if valid after relaxation
            # We can do a quick check every few iterations or at the end
            
        # After relaxation, check if it's valid
        # If valid, accept
        # If not, revert and try smaller dr or random perturbation
        
        # Calculate r for current_points
        # Note: current_points might have changed significantly.
        # The "valid" r is the min of min_pair/2 and min_boundary.
        # We want to know if we achieved new_r.
        
        # Let's compute the actual r of the current configuration
        # If actual_r >= new_r, then we succeeded in increasing r.
        
        # Efficient check:
        # We can just compute the min distances again.
        
        # But wait, if we relax to satisfy target_d, the actual min distance might be slightly larger.
        # And min boundary might be larger.
        
        # Let's compute the score.
        new_r_actual = _compute_radius(current_points)
        
        if new_r_actual > current_r:
            # Improvement!
            current_r = new_r_actual
            best_points = current_points.copy()
            best_r = current_r
            # Reset dr to a slightly larger value to continue increasing?
            # Or keep same dr.
            # Let's keep dr constant for now.
        else:
            # No improvement, revert to best_points
            current_points = best_points.copy()
            # Try a random perturbation to escape local minimum?
            # Or reduce dr?
            # Let's add a small random noise to current_points to escape
            # and try again with a slightly smaller dr?
            # Actually, if we can't increase r, we might be stuck.
            # Let's try to randomize the local search.
            
            # Apply a small random displacement
            # This helps escape local minima where the "push apart" gets stuck in a cycle.
            noise = np.random.normal(0, 1e-4, size=(N, 2))
            current_points += noise
            
            # Re-check
            new_r_actual = _compute_radius(current_points)
            if new_r_actual > current_r:
                current_r = new_r_actual
                best_points = current_points.copy()
                best_r = current_r
            else:
                # Still no improvement, revert
                current_points = best_points.copy()
                
                # If we are stuck for a while, maybe reduce dr?
                # Or increase randomization?
                # For now, just continue. The random noise will eventually help.
                
                # To avoid infinite loop on stuck state, we can try a larger random jump occasionally
                if steps % 100 == 0:
                    noise = np.random.normal(0, 1e-2, size=(N, 2))
                    current_points += noise
                    # Project back inside unit circle roughly
                    norms = np.linalg.norm(current_points, axis=1)
                    for i in range(N):
                        if norms[i] > 0.9:
                            current_points[i] *= 0.9 / norms[i]
                    
                    new_r_actual = _compute_radius(current_points)
                    if new_r_actual > current_r:
                        current_r = new_r_actual
                        best_points = current_points.copy()
                        best_r = current_r
                    else:
                        current_points = best_points.copy()

    return best_points
# EVOLVE-BLOCK-END
