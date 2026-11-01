# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack N equal circles inside the unit circle using physics-based relaxation.
    
    Uses a symplectic integrator to simulate particles with:
    - Repulsive forces between overlapping circles (spring-like)
    - Boundary forces to keep circles inside the unit circle
    - Iterative radius increase with relaxation at each step
    """
    N = 50
    dt = 0.001
    damping = 0.999  # slight damping for stability
    
    def compute_forces(positions, r):
        """Compute forces on all particles given positions and target radius r."""
        n = len(positions)
        forces = np.zeros_like(positions)
        
        # Pairwise repulsive forces (spring-like)
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * r and dist > 1e-10:
                    # Spring force: proportional to overlap
                    overlap = 2 * r - dist
                    force_magnitude = 1000.0 * overlap  # spring constant
                    force_direction = diff / dist
                    forces[i] += force_magnitude * force_direction
                    forces[j] -= force_magnitude * force_direction
                elif dist <= 1e-10:
                    # Handle coincident points: push apart randomly
                    angle = np.random.uniform(0, 2 * np.pi)
                    push = np.array([np.cos(angle), np.sin(angle)]) * 1e-3
                    forces[i] += push
                    forces[j] -= push
        
        # Boundary forces: keep circles inside unit circle
        for i in range(n):
            radius = np.linalg.norm(positions[i])
            max_allowed = 1.0 - r
            if radius > max_allowed and radius > 1e-10:
                # Radial force pushing inward
                overlap = radius - max_allowed
                force_magnitude = 1000.0 * overlap
                force_direction = -positions[i] / radius
                forces[i] += force_magnitude * force_direction
        
        return forces
    
    def relax(positions, r, num_steps=5000):
        """
        Relax the configuration using symplectic integration.
        Returns the final positions.
        """
        pos = positions.copy()
        vel = np.zeros_like(pos)
        
        for step in range(num_steps):
            # Compute forces
            forces = compute_forces(pos, r)
            
            # Velocity Verlet integration
            # Update velocity half step
            vel_half = vel + 0.5 * dt * forces
            # Update position
            pos_new = pos + dt * vel_half
            # Compute new forces
            forces_new = compute_forces(pos_new, r)
            # Update velocity full step
            vel_new = vel_half + 0.5 * dt * forces_new
            # Apply damping
            vel_new *= damping
            
            pos = pos_new
            vel = vel_new
            
            # Check for convergence (optional, can be expensive)
            if step % 1000 == 0:
                max_force = np.max(np.linalg.norm(forces, axis=1))
                if max_force < 1e-6:
                    break
        
        return pos
    
    def is_valid(positions, r):
        """Check if the configuration is valid for radius r."""
        n = len(positions)
        
        # Check boundary
        radii = np.linalg.norm(positions, axis=1)
        if np.any(radii > 1.0 - r + 1e-8):
            return False
        
        # Check pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 2 * r - 1e-8:
                    return False
        
        return True
    
    def make_initial_configuration():
        """Create an initial hexagonal-like configuration."""
        points = []
        scale = 0.15
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * scale + 0.5 * scale * (j % 2)
                y = j * scale * np.sqrt(3) / 2
                if np.sqrt(x * x + y * y) < 1.0:
                    points.append([x, y])
        points = np.array(points)
        
        if len(points) > N:
            # Sort by distance from center and take closest N
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)[:N]
            points = points[indices]
        elif len(points) < N:
            while len(points) < N:
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, 0.9)
                points = np.vstack([points, [radius * np.cos(angle), radius * np.sin(angle)]])
        
        return points
    
    np.random.seed(42)
    
    best_r = 0.0
    best_positions = make_initial_configuration()
    
    # Start with a small radius and increase
    r = 0.05
    positions = best_positions.copy()
    
    # First, relax at initial radius
    positions = relax(positions, r, num_steps=2000)
    
    # Try to increase radius
    step_size = 0.002
    max_steps = 100
    
    for step in range(max_steps):
        # Relax at current radius
        positions = relax(positions, r, num_steps=3000)
        
        # Check if valid
        if is_valid(positions, r):
            best_r = r
            best_positions = positions.copy()
            # Try to increase radius
            r += step_size
        else:
            # Not valid, try to reduce step and relax more
            # Try with smaller step
            r_test = r - step_size / 2
            positions = relax(positions, r_test, num_steps=5000)
            if is_valid(positions, r_test):
                best_r = r_test
                best_positions = positions.copy()
                r = r_test + step_size / 4  # smaller step for next attempt
            else:
                # Can't even do half step, break
                break
    
    # Final polish: try to squeeze a bit more
    for _ in range(20):
        r_test = best_r + 0.0005
        positions_test = relax(best_positions.copy(), r_test, num_steps=3000)
        if is_valid(positions_test, r_test):
            best_r = r_test
            best_positions = positions_test
        else:
            break
    
    return best_positions
# EVOLVE-BLOCK-END
