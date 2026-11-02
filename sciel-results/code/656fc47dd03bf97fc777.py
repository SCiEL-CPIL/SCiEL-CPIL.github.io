"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 26

# EVOLVE-BLOCK-START
def construct():
    """
    Physics-based circle packing using force-directed layout.
    Circles start small and grow, with forces resolving overlaps and boundary violations.
    """
    # Simulation Parameters
    num_steps = 10000
    dt = 0.01
    growth_rate = 0.0002
    damping = 0.85
    initial_radius = 0.005
    force_constant = 5.0
    boundary_force_constant = 5.0
    
    best_score = -np.inf
    best_result = None
    
    # Multiple restarts to escape local minima
    for restart in range(5):
        np.random.seed(restart * 98765)
        
        # Initialize
        centers = np.random.uniform(0.1, 0.9, size=(N, 2))
        radii = np.full(N, initial_radius)
        velocities = np.zeros((N, 2))
        
        for step in range(num_steps):
            # 1. Grow radii
            radii += growth_rate
            
            # 2. Calculate forces
            forces = np.zeros((N, 2))
            
            # Pairwise repulsion (O(N^2) is fine for N=26)
            for i in range(N):
                for j in range(i + 1, N):
                    delta = centers[i] - centers[j]
                    dist_sq = np.dot(delta, delta)
                    min_dist = radii[i] + radii[j]
                    
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                        if dist < min_dist:
                            overlap = min_dist - dist
                            # Force is proportional to overlap
                            force_mag = force_constant * overlap
                            # Direction is from j to i (repulsion)
                            if dist > 1e-9:
                                force_vec = (delta / dist) * force_mag
                                forces[i] += force_vec
                                forces[j] -= force_vec
                            else:
                                # If coincident, random push
                                rand_vec = np.random.normal(0, 1, size=2)
                                rand_vec /= np.linalg.norm(rand_vec)
                                forces[i] += rand_vec * force_mag
                                forces[j] -= rand_vec * force_mag

            # Boundary repulsion
            for i in range(N):
                r = radii[i]
                x, y = centers[i]
                
                # Left wall
                if x < r:
                    forces[i][0] += boundary_force_constant * (r - x)
                # Right wall
                if 1.0 - x < r:
                    forces[i][0] -= boundary_force_constant * (r - (1.0 - x))
                # Bottom wall
                if y < r:
                    forces[i][1] += boundary_force_constant * (r - y)
                # Top wall
                if 1.0 - y < r:
                    forces[i][1] -= boundary_force_constant * (r - (1.0 - y))
                
            # 3. Update velocities and positions
            velocities = damping * velocities + forces * dt
            centers += velocities * dt
            
            # 4. Hard clip to square to prevent numerical divergence
            # Note: This is a safety measure. The forces should keep them inside.
            # Clipping centers to [0,1] doesn't guarantee radius validity, but prevents escape.
            centers = np.clip(centers, 0.0, 1.0)
            
            # 5. Check validity and update best
            # We need a strictly valid configuration to report.
            is_valid = True
            current_sum = 0.0
            
            # Check boundaries
            for i in range(N):
                r = radii[i]
                x, y = centers[i]
                if x < r - 1e-9 or 1.0 - x < r - 1e-9 or y < r - 1e-9 or 1.0 - y < r - 1e-9:
                    is_valid = False
                    break
                current_sum += r
            
            # Check overlaps
            if is_valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        if dist < radii[i] + radii[j] - 1e-9:
                            is_valid = False
                            break
                    if not is_valid:
                        break
            
            if is_valid:
                if current_sum > best_score:
                    best_score = current_sum
                    best_result = np.column_stack((centers, radii)).copy()
                    
    # Fallback
    if best_result is None:
        centers = np.random.uniform(0.1, 0.9, size=(N, 2))
        radii = np.full(N, 0.001)
        best_result = np.column_stack((centers, radii))

    return best_result
# EVOLVE-BLOCK-END
