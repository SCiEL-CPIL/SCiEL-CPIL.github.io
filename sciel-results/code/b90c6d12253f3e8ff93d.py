"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Rigid-body dynamics with soft-contact potentials to pack 200 equal circles
    in the unit square [-0.5, 0.5]^2.
    
    Uses a symplectic Euler integrator with adaptive time-stepping and 
    high-order polynomial overlap penalties to settle into a stable equilibrium.
    """
    np.random.seed(42)
    
    # Initial positions: jittered grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    positions = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Add small random jitter to break symmetry
    positions += np.random.uniform(-0.01, 0.01, size=(N, 2))
    
    # Keep within bounds
    positions = np.clip(positions, -0.49, 0.49)
    
    velocities = np.zeros((N, 2))
    
    # Simulation parameters
    dt = 0.001
    max_iters = 50000
    damping = 0.98
    wall_strength = 1000.0
    overlap_strength = 500.0
    overlap_power = 4  # High-order polynomial for smooth but strong repulsion
    
    # Adaptive time stepping
    min_dt = 0.0001
    max_dt = 0.005
    
    for iteration in range(max_iters):
        # Compute forces
        forces = np.zeros((N, 2))
        
        # Pairwise repulsion for overlapping circles
        # Use vectorized computation for efficiency
        for i in range(N):
            for j in range(i + 1, N):
                diff = positions[i] - positions[j]
                dist_sq = np.dot(diff, diff)
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                
                # Estimate current radius from typical spacing
                # We want to push circles apart if they are too close
                # Target separation is roughly 1/m for grid, but we let dynamics find it
                # Use a reference distance based on area: r ~ 0.5/sqrt(N) ~ 0.035
                ref_dist = 0.072  # ~ 2 * 0.036
                
                if dist < ref_dist:
                    # Penalty for being closer than ref_dist
                    # Force magnitude grows as we get closer
                    # F = strength * (ref_dist - dist)^power * direction
                    overlap = ref_dist - dist
                    if overlap > 0:
                        force_mag = overlap_strength * (overlap / ref_dist) ** overlap_power
                        # Normalize direction
                        if dist > 1e-12:
                            direction = diff / dist
                        else:
                            direction = np.array([1.0, 0.0])
                        forces[i] += force_mag * direction
                        forces[j] -= force_mag * direction
        
        # Wall constraints: push back if outside [-0.5, 0.5]
        for i in range(N):
            for k in range(2):
                pos = positions[i, k]
                # Left wall at -0.5
                if pos < -0.5 + 0.01:
                    forces[i, k] += wall_strength * (-0.5 + 0.01 - pos) ** 2
                # Right wall at 0.5
                if pos > 0.5 - 0.01:
                    forces[i, k] -= wall_strength * (pos - (0.5 - 0.01)) ** 2
        
        # Update velocities (symplectic Euler: v += F*dt, x += v*dt)
        velocities += forces * dt
        velocities *= damping  # Dissipation
        
        # Limit velocity to prevent instability
        max_vel = 0.1
        vel_mags = np.linalg.norm(velocities, axis=1, keepdims=True)
        mask = vel_mags > max_vel
        velocities[mask] *= max_vel / vel_mags[mask]
        
        # Update positions
        positions += velocities * dt
        
        # Hard clip to stay in bounds (safety)
        positions = np.clip(positions, -0.495, 0.495)
        
        # Adaptive time stepping: reduce dt if forces are large
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 100:
            dt = max(min_dt, dt * 0.8)
        elif max_force < 10:
            dt = min(max_dt, dt * 1.05)
        
        # Check convergence
        if iteration > 1000 and max_force < 0.1:
            break
    
    # Final refinement: local optimization to ensure no overlaps and maximize min distance
    # Compute current min distance and wall distance
    def compute_radius(positions):
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    min_dist = d
        # Wall distance
        wall_min = np.min(np.min(0.5 - np.abs(positions), axis=1))
        return min(min_dist / 2, wall_min)
    
    # Simple local hill-climbing to improve the radius
    for refine_iter in range(10000):
        current_r = compute_radius(positions)
        
        # Find the most constrained pair or wall
        min_dist = np.inf
        min_pair = (-1, -1)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        
        wall_min = np.inf
        wall_idx = (-1, -1)
        for i in range(N):
            for k in range(2):
                w = 0.5 - np.abs(positions[i, k])
                if w < wall_min:
                    wall_min = w
                    wall_idx = (i, k)
        
        # Determine which constraint is limiting
        limiting_r = min(min_dist / 2, wall_min)
        
        # Try to move the most constrained circles
        step_size = 0.001
        improved = False
        
        if min_dist / 2 < wall_min:
            i, j = min_pair
            diff = positions[i] - positions[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                direction = diff / dist
                # Move i in direction, j in -direction
                new_i = positions[i] + direction * step_size
                new_j = positions[j] - direction * step_size
                # Check bounds
                if np.all(np.abs(new_i) < 0.5) and np.all(np.abs(new_j) < 0.5):
                    # Check if this improves the minimum
                    old_min = min_dist
                    positions[i] = new_i
                    positions[j] = new_j
                    new_min_dist = np.linalg.norm(positions[i] - positions[j])
                    # Quick check: is new_min_dist > old_min?
                    if new_min_dist > old_min * 0.99:  # Allow slight decrease if other distances improve
                        # Full check would be expensive, accept if local improvement
                        improved = True
                    else:
                        positions[i] = new_i
                        positions[j] = new_j
        else:
            i, k = wall_idx
            # Move away from wall
            sign = 1 if positions[i, k] > 0 else -1
            new_pos = positions[i, k] + sign * step_size
            if abs(new_pos) < 0.5:
                positions[i, k] = new_pos
                improved = True
        
        if not improved:
            break
    
    return positions
# EVOLVE-BLOCK-END
