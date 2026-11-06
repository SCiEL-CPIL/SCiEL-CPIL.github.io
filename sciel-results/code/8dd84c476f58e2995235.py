"""Physics-based relaxation for packing 50 equal circles in a unit disk."""
import numpy as np

N = 50


def construct():
    """
    Place N=50 equal circles inside the unit disk (radius 1, centered at origin)
    to maximize the minimum radius.
    
    Uses a physics-based relaxation:
    - Repulsive potential between circles (inverse power law)
    - Soft wall potential to keep circles inside the unit disk
    - Gradient descent to minimize energy
    """
    # Initial configuration: random points inside the unit disk
    # Use multiple restarts to find a good configuration
    best_centres = None
    best_radius = -1.0

    rng = np.random.default_rng(42)

    for attempt in range(20):
        # Random initial positions inside unit disk
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0, 0.95, N)
        centres = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        # Ensure no initial overlaps by spreading them out
        # Add some structured initialization mixed with random
        if attempt == 0:
            # Use a hexagonal-like grid as starting point
            spacing = 0.2
            points = []
            for j in range(-5, 6):
                for i in range(-5, 6):
                    x = i * spacing + (0.5 * spacing if j % 2 else 0)
                    y = j * spacing * np.sqrt(3) / 2
                    if x**2 + y**2 <= 0.81:  # radius 0.9
                        points.append([x, y])
            points = np.array(points)
            if len(points) >= N:
                # Pick N points most central
                r = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(r)[:N]
                centres = points[idx].copy()
            else:
                # Fall back to random
                angles = rng.uniform(0, 2 * np.pi, N)
                radii = rng.uniform(0, 0.95, N)
                centres = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif attempt % 3 == 1:
            # Fibonacci spiral initialization
            golden_angle = np.pi * (3 - np.sqrt(5))
            for i in range(N):
                r = 0.9 * np.sqrt((i + 0.5) / N)
                theta = i * golden_angle
                centres[i] = [r * np.cos(theta), r * np.sin(theta)]
        # else: random as set above

        # Energy minimization via gradient descent
        # Energy = sum of pairwise repulsion + wall potential
        # Use a strong repulsion to push circles apart
        
        positions = centres.copy()
        lr = 0.01
        n_steps = 5000
        momentum = np.zeros_like(positions)
        beta = 0.9  # momentum coefficient

        for step in range(n_steps):
            # Compute forces
            forces = np.zeros_like(positions)
            
            # Pairwise repulsion: F_ij = k * (1/d^p - 1) * (r_i - r_j)/d for d < 1
            # Use a soft repulsion that is strong when overlapping
            for i in range(N):
                for j in range(i + 1, N):
                    diff = positions[i] - positions[j]
                    d = np.linalg.norm(diff)
                    if d < 1e-10:
                        # Random nudge if coincident
                        forces[i] += rng.normal(0, 0.01, 2)
                        forces[j] -= rng.normal(0, 0.01, 2)
                        continue
                    # Repulsive force: strong inverse power
                    # Use d^-3 to make it very strong when close
                    force_mag = 100.0 / (d ** 3) if d < 1.0 else 10.0 / (d ** 5)
                    force_dir = diff / d
                    forces[i] += force_mag * force_dir
                    forces[j] -= force_mag * force_dir
            
            # Wall potential: push inward if too close to boundary
            for i in range(N):
                r = np.linalg.norm(positions[i])
                if r > 0.9:
                    # Force pushing toward center
                    force_mag = 50.0 * (r - 0.9) ** 2
                    if r > 1e-10:
                        forces[i] -= force_mag * positions[i] / r
                # Also a hard constraint: if outside, push back strongly
                if r > 1.0:
                    forces[i] -= 1000.0 * positions[i] / r
            
            # Gradient descent with momentum
            momentum = beta * momentum + (1 - beta) * forces
            positions += lr * momentum
            
            # Project back inside unit disk (hard constraint)
            for i in range(N):
                r = np.linalg.norm(positions[i])
                if r > 1.0:
                    positions[i] = positions[i] / r * 0.999

            # Decay learning rate
            lr = 0.01 * (0.999 ** step)

        # Compute the actual radius for this configuration
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    min_dist = d
        min_wall = np.inf
        for i in range(N):
            r = np.linalg.norm(positions[i])
            dist_to_wall = 1.0 - r
            if dist_to_wall < min_wall:
                min_wall = dist_to_wall
        
        radius = min(min_dist / 2.0, min_wall)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = positions.copy()

    # Final verification and return
    # Ensure all centres are inside the unit disk
    for i in range(N):
        r = np.linalg.norm(best_centres[i])
        if r > 1.0:
            best_centres[i] = best_centres[i] / r * 0.999

    return best_centres
