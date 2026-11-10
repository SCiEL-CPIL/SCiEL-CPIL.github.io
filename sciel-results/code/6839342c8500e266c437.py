import numpy as np
import time
import math

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe within 120s

    # 1. Initialization: Random points in the unit disk
    # We use rejection sampling or polar coordinates for uniform distribution in disk
    # To ensure we start reasonably, we can also try a few different seeds or initializations
    # but random is a good baseline for MD.
    
    def init_random():
        # Generate N points uniformly in unit disk
        # r = sqrt(random()), theta = random() * 2*pi
        r = np.sqrt(np.random.uniform(0, 1, N))
        theta = np.random.uniform(0, 2 * np.pi, N)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))

    # Try a few initial configurations to pick the best one quickly
    # Or just start with one and let MD do the work. 
    # Given the time budget, we can afford a few restarts if we are fast.
    # But MD itself is the main workhorse.
    
    best_positions = None
    best_score = -1.0

    # We will run MD for a fixed amount of time or steps, 
    # possibly with multiple restarts if time permits.
    # However, a single long run with cooling is often robust.
    
    # Let's define the physics parameters
    # Repulsion: 1/r^2 or similar. We want to maximize min distance.
    # This is equivalent to minimizing a potential energy where close pairs have high energy.
    # A common choice is E = sum_{i<j} 1/d_ij^2 + sum_i |c_i|^2 (confinement)
    # Actually, we want to maximize the minimum distance. 
    # A soft repulsive potential like 1/d^k works well.
    # We also need a confinement force to keep them inside.
    
    # Let's use a simple velocity Verlet or Euler integration with damping.
    # Forces:
    # 1. Pairwise repulsion: F_ij = -grad(1/d^2) = 2 * (c_i - c_j) / d^4 (for i!=j)
    #    Wait, grad(1/d^2) wrt c_i is -2*(c_i-c_j)/d^4. So force is +2*(c_i-c_j)/d^4.
    #    This is a strong repulsion.
    # 2. Confinement: We want to keep |c_i| < 1. 
    #    If we just use repulsion, they will fly apart. 
    #    We need an attractive force to the center, or a hard wall.
    #    Hard wall is hard to differentiate. 
    #    Let's use a potential U_conf = |c_i|^4 or |c_i|^2.
    #    If we use U = sum 1/d_ij^2 + lambda * sum |c_i|^2, 
    #    the equilibrium will be a balance between spreading out and staying in.
    #    We want them to fill the disk. 
    #    A better confinement is a "soft wall" that pushes back strongly if r > R_target.
    #    But we don't know R_target (the optimal radius).
    #    However, the problem is to pack them in unit circle.
    #    So the constraint is |c_i| <= 1.
    #    We can use a penalty potential: U_wall = max(0, |c_i| - 0.99)^2 * large_constant.
    #    This keeps them well inside.
    
    # Let's refine the energy function:
    # E = sum_{i<j} 1/d_ij^2 + K * sum_i max(0, |c_i| - 0.95)^2
    # The second term penalizes going outside 0.95. This ensures they stay inside.
    # The first term pushes them apart.
    # The system will settle into a configuration that maximizes the minimum distance
    # subject to the boundary constraint.
    
    # Parameters
    dt = 0.001
    damping = 0.99
    K_wall = 100.0
    wall_radius = 0.95
    
    # Optimization loop
    # We will run for a certain number of steps or until time runs out.
    # We can also implement a simple "cooling" by reducing dt or adding noise.
    
    # To improve quality, we can run multiple restarts with different random seeds.
    # Let's structure it as: 
    # 1. Run MD for T1 seconds.
    # 2. Evaluate score.
    # 3. If time remains, restart with new random init and run for T2 seconds.
    # 4. Keep best.
    
    # Actually, a single long run with occasional "kicks" (adding random velocity) 
    # can help escape local minima.
    
    positions = init_random()
    velocities = np.zeros((N, 2))
    
    # Pre-compute indices for efficiency? 
    # N=50 is small, so O(N^2) per step is fine.
    # 50*50 = 2500 ops. 100k steps = 250M ops. Python might be slow.
    # Let's use vectorized NumPy operations.
    
    # Vectorized force calculation
    def calc_forces(pos):
        # pos shape (N, 2)
        # Compute pairwise distances
        # diff[i, j] = pos[i] - pos[j]
        # This is N x N x 2
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # dist^2
        dist2 = np.sum(diff**2, axis=2)
        # Avoid division by zero
        dist2[dist2 < 1e-10] = 1e-10
        
        # Force from repulsion: 2 * diff / dist2^2
        # F_ij = 2 * (c_i - c_j) / d^4
        # We need to sum over j for each i.
        # force[i] = sum_j F_ij
        # F_ij vector is 2 * diff[i,j] / dist2[i,j]^2
        # Note: F_ij = -F_ji, so we can just compute full matrix and sum.
        
        # dist2 is N x N
        # inv_dist4 = 1 / dist2^2
        inv_dist4 = 1.0 / (dist2**2)
        
        # Zero out diagonal (self-interaction)
        np.fill_diagonal(inv_dist4, 0.0)
        
        # Force contribution from repulsion
        # F_rep[i] = sum_j 2 * diff[i,j] * inv_dist4[i,j]
        # diff is N x N x 2
        # inv_dist4 is N x N
        # We need to multiply diff[i,j, :] by inv_dist4[i,j]
        # F_rep = 2.0 * np.sum(diff * inv_dist4[:, :, np.newaxis], axis=1)
        
        F_rep = 2.0 * np.sum(diff * inv_dist4[:, :, np.newaxis], axis=1)
        
        # Wall force
        # U_wall = K * max(0, r - R)^2
        # dU/dr = 2 * K * (r - R) if r > R else 0
        # Force = -grad(U) = -2 * K * (r - R) * (c/r) if r > R else 0
        r = np.linalg.norm(pos, axis=1)
        # mask for outside wall
        mask = r > wall_radius
        # Only compute for those outside
        # F_wall[i] = -2 * K * (r_i - R) * (pos_i / r_i)
        F_wall = np.zeros_like(pos)
        if np.any(mask):
            r_masked = r[mask]
            pos_masked = pos[mask]
            coeff = -2.0 * K_wall * (r_masked - wall_radius) / r_masked
            F_wall[mask] = pos_masked * coeff[:, np.newaxis]
            
        return F_rep + F_wall

    # Main MD Loop
    # We will run until deadline - 5s
    steps = 0
    # We can adaptively adjust dt or just run fixed steps.
    # Let's run in chunks to check time.
    
    while time.time() < deadline - 5.0:
        # Calculate forces
        forces = calc_forces(positions)
        
        # Update velocities (with damping)
        velocities = velocities * damping + forces * dt
        # Update positions
        positions = positions + velocities * dt
        
        # Optional: Add small random noise to escape local minima
        # if steps % 1000 == 0:
        #     velocities += np.random.normal(0, 0.01, positions.shape)
        
        steps += 1
        
        # Periodically check score to keep track of best
        # But checking score is O(N^2) too. 
        # We can do it every 1000 steps.
        if steps % 1000 == 0:
            # Calculate current min distance
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist2 = np.sum(diff**2, axis=2)
            np.fill_diagonal(dist2, np.inf)
            min_dist = np.sqrt(np.min(dist2))
            min_dist = min_dist / 2.0  # radius
            
            # Check boundary
            radii = np.linalg.norm(positions, axis=1)
            max_r = np.max(radii)
            if max_r <= 1.0:
                current_score = min(min_dist, 1.0 - max_r)
            else:
                current_score = -1.0  # Invalid
            
            if current_score > best_score:
                best_score = current_score
                best_positions = positions.copy()
                
    # Final evaluation of the current state
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    min_dist = np.sqrt(np.min(dist2)) / 2.0
    radii = np.linalg.norm(positions, axis=1)
    max_r = np.max(radii)
    
    if max_r <= 1.0:
        final_score = min(min_dist, 1.0 - max_r)
    else:
        final_score = -1.0
        
    if final_score > best_score:
        best_positions = positions
        best_score = final_score

    # If best_positions is still None (e.g. timeout before first check), use current
    if best_positions is None:
        best_positions = positions

    return best_positions
# EVOLVE-BLOCK-END
