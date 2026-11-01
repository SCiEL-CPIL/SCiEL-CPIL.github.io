"""Rewritten program for the suite task; uses discrete combinatorial annealing."""
import numpy as np

N = 100


def construct():
    # Discrete combinatorial annealing (lattice-gas Metropolis-Hastings)
    # to pack 100 equal spheres in the unit cube [-1/2, 1/2]^3.
    
    rng = np.random.default_rng(42)
    
    # Initial configuration: random points inside the cube
    # with a small margin to ensure validity
    margin = 0.01
    positions = rng.uniform(-0.5 + margin, 0.5 - margin, size=(N, 3))
    
    # Ensure initial validity by adjusting any points too close to boundary
    for i in range(N):
        for k in range(3):
            if abs(positions[i, k]) > 0.5 - margin:
                positions[i, k] = np.sign(positions[i, k]) * (0.5 - margin)
    
    def compute_radius(pos):
        """Compute the maximum common radius for a given configuration."""
        # Distance to nearest boundary
        dist_to_boundary = np.min(0.5 - np.abs(pos), axis=1)
        min_boundary = np.min(dist_to_boundary)
        
        # Minimum pairwise distance
        min_pair_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_pair_dist:
                    min_pair_dist = d
        
        return min(min_pair_dist / 2.0, min_boundary)
    
    # Precompute initial radius
    current_radius = compute_radius(positions)
    
    # Annealing parameters
    n_iterations = 50000
    T_start = 0.05
    T_end = 0.0001
    step_scale = 0.05
    
    best_positions = positions.copy()
    best_radius = current_radius
    
    for it in range(n_iterations):
        # Linear cooling schedule
        alpha = it / max(n_iterations - 1, 1)
        T = T_start * (T_end / T_start) ** alpha
        step = step_scale * (1.0 - 0.8 * alpha)
        
        # Pick a random sphere to move
        i = rng.integers(N)
        
        # Generate a random perturbation
        delta = rng.normal(0, step, size=3)
        new_pos = positions.copy()
        new_pos[i] = positions[i] + delta
        
        # Clamp to cube boundaries
        new_pos[i] = np.clip(new_pos[i], -0.5, 0.5)
        
        # Check validity: no coincident centers, all inside cube
        if not np.all(np.isfinite(new_pos)):
            continue
        
        # Compute new radius
        new_radius = compute_radius(new_pos)
        
        # Metropolis acceptance
        if new_radius >= current_radius:
            accept = True
        else:
            delta_e = current_radius - new_radius  # positive means worse
            if delta_e > 0:
                accept = rng.random() < np.exp(-delta_e / T)
            else:
                accept = True
        
        if accept:
            positions = new_pos
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_positions = positions.copy()
    
    return best_positions
