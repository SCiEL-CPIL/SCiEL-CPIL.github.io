"""Physics-based relaxation for packing 50 equal circles in a unit disk."""
import numpy as np

N = 50


def construct():
    """
    Place N=50 equal circles inside the unit circle (radius 1, centred at origin)
    to maximize the common radius.
    
    Uses a physics-based relaxation:
    - Circles repel each other via a strong inverse-power potential
    - A soft wall potential keeps them inside the unit disk
    - Gradient descent on a smooth energy landscape
    
    Returns: array of shape (N, 2) with the centres.
    """
    np.random.seed(42)
    
    # Initialize positions in a ring-like pattern with some randomness
    # Start with a hexagonal-ish arrangement perturbed by noise
    positions = np.zeros((N, 2))
    
    # Use a multi-ring initialization
    rings = [0, 6, 12, 18, 14]  # approximate counts per ring
    ring_radii = [0.0, 0.25, 0.45, 0.65, 0.85]
    
    idx = 0
    for ring_idx, (count, r) in enumerate(zip(rings, ring_radii)):
        if count == 0:
            continue
        if ring_idx == 0:
            positions[idx] = [0.0, 0.0]
            idx += 1
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + np.random.uniform(-0.1, 0.1, size=count)
            for i in range(count):
                if idx >= N:
                    break
                positions[idx] = [r * np.cos(angles[i]), r * np.sin(angles[i])]
                idx += 1
    
    # Ensure we have exactly N positions
    if idx < N:
        # Fill remaining with random points
        remaining = N - idx
        angles = np.linspace(0, 2 * np.pi, remaining, endpoint=False) + np.random.uniform(-0.1, 0.1, size=remaining)
        for i in range(remaining):
            positions[idx] = [0.7 * np.cos(angles[i]), 0.7 * np.sin(angles[i])]
            idx += 1
    positions = positions[:N]
    
    # Physics-based relaxation
    def compute_forces(pos, dt=0.01):
        """Compute forces on each particle."""
        n = len(pos)
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion: strong inverse-power potential
        # Use a soft repulsion that's strong when circles overlap
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random tiny perturbation
                    forces[i] += np.random.normal(0, 0.01, 2)
                    forces[j] -= np.random.normal(0, 0.01, 2)
                    continue
                
                # Strong repulsion: force ~ 1/dist^2 scaled
                # We want to push apart when close
                repulsion_strength = 10.0 / (dist ** 2)
                direction = diff / dist
                forces[i] += repulsion_strength * direction
                forces[j] -= repulsion_strength * direction
        
        # Wall constraint: soft potential that pushes inward when too close to boundary
        # The circles should have radius r, so center must be at distance <= 1 - r from origin
        # We don't know r yet, so use a soft wall that keeps points well inside
        for i in range(n):
            dist_from_origin = np.linalg.norm(pos[i])
            if dist_from_origin > 0.9:
                # Push inward with increasing force as we approach the boundary
                wall_force = 50.0 * (dist_from_origin - 0.9) ** 2
                if dist_from_origin > 1e-10:
                    direction = -pos[i] / dist_from_origin
                else:
                    direction = np.array([0.0, 0.0])
                forces[i] += wall_force * direction
        
        return forces
    
    # Gradient descent with adaptive step size
    lr = 0.005
    momentum = 0.5
    velocity = np.zeros_like(positions)
    
    # Run relaxation for a number of iterations
    n_iterations = 20000
    for it in range(n_iterations):
        forces = compute_forces(positions)
        
        # Update velocity with momentum
        velocity = momentum * velocity + lr * forces
        positions = positions + velocity
        
        # Hard constraint: ensure all points are strictly inside the unit disk
        # with some margin
        for i in range(N):
            dist = np.linalg.norm(positions[i])
            if dist > 0.98:
                positions[i] = positions[i] * (0.98 / dist)
        
        # Adaptive learning rate: reduce if forces are large
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 10.0:
            lr *= 0.9
        elif max_force < 0.1:
            lr = min(lr * 1.01, 0.02)
    
    # Final cleanup: ensure all points are inside with a small margin
    for i in range(N):
        dist = np.linalg.norm(positions[i])
        if dist > 0.99:
            positions[i] = positions[i] * (0.99 / dist)
    
    return positions
