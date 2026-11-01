"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Use a Halton sequence for initial low-discrepancy distribution
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    # Generate N points using Halton sequence in 3D
    # We start index from 1 to avoid 0
    indices = np.arange(1, N + 1)
    points = np.zeros((N, 3))
    for d in range(3):
        # Bases 2, 3, 5 are coprime
        base = [2, 3, 5][d]
        for i in range(N):
            points[i, d] = halton(indices[i], base)
    
    # Map to cube [-0.5, 0.5]
    # Halton returns [0, 1), so we scale and shift
    # To avoid points exactly on the boundary initially, we can add a small margin or just let optimization handle it
    points = points * 0.9 - 0.45  # Keep slightly inside to avoid initial singularities in force calc
    
    # Optimization using Repulsion Model
    # Potential Energy E = sum(1/d_ij^2) + sum(1/d_wall^2)
    # We want to minimize E, which maximizes the minimum distances.
    
    # Hyperparameters
    T = 0.05  # Initial temperature (step size)
    cool_rate = 0.999
    iterations = 3000
    min_T = 1e-5
    
    current_T = T
    
    for it in range(iterations):
        if current_T < min_T:
            break
            
        forces = np.zeros_like(points)
        
        # 1. Point-Point Repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    # Random nudge if coincident
                    diff = np.random.normal(0, 1e-6, 3)
                    dist_sq = np.dot(diff, diff)
                
                # Force magnitude ~ 1/dist^2, direction along diff
                # Gradient of 1/d^2 is -2/d^3 * diff
                # We want to push apart, so force on i is in direction of diff (i-j)
                # Let's use a softer potential to avoid huge spikes: 1/d^4 or similar? 
                # Standard Coulomb 1/d is stable. 1/d^2 is stronger.
                # Let's use force = diff / dist_sq^1.5 (which is grad of 1/dist)
                # Actually, minimizing sum(1/d) is a good proxy for maximizing min d.
                # Force on i from j: k * (pi - pj) / ||pi - pj||^3
                dist = np.sqrt(dist_sq)
                force_mag = 1.0 / (dist_sq * dist)
                force_vec = diff * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # 2. Wall Repulsion
        # Distance to walls
        # x in [-0.5, 0.5]
        # d_left = x + 0.5, d_right = 0.5 - x
        # We want to push away from walls.
        # Force is strong when close to wall.
        
        for i in range(N):
            p = points[i]
            # Check each dimension
            for d in range(3):
                # Left wall at -0.5
                d_left = p[d] + 0.5
                if d_left < 0.1: # Only apply force if close
                    # Force pushes in +x direction
                    forces[i, d] += 1.0 / (d_left * d_left)
                
                # Right wall at 0.5
                d_right = 0.5 - p[d]
                if d_right < 0.1:
                    # Force pushes in -x direction
                    forces[i, d] -= 1.0 / (d_right * d_right)
        
        # Update positions
        # Limit step size to avoid instability
        max_step = 0.05
        steps = forces * current_T
        # Normalize steps to max length
        step_lengths = np.linalg.norm(steps, axis=1)
        mask = step_lengths > max_step
        if np.any(mask):
            steps[mask] = steps[mask] / step_lengths[mask] * max_step
            
        points += steps
        
        # Ensure points stay within cube (hard constraint)
        points = np.clip(points, -0.5, 0.5)
        
        # Cool down
        current_T *= cool_rate

    return points
# EVOLVE-BLOCK-END
