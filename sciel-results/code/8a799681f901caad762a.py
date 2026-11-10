"""New program: Lattice Initiation + Multi-Start Relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r))(exp(rho(1-r)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    x = rho * (1.0 - r)
    iu = np.triu_indices(n, k=1)
    xij = x[iu]
    e = np.exp(xij) * (np.exp(xij) - 2.0)
    return float(np.sum(e))


def _enforce_min_distance(pos, min_dist=0.5, max_iter=100):
    """Push atoms apart to satisfy minimum distance constraint."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    if d < 1e-10:
                        # Random push
                        pos[i] += np.random.normal(0, 0.1, 3)
                        pos[j] -= np.random.normal(0, 0.1, 3)
                    else:
                        # Push apart along the line connecting them
                        diff = pos[i] - pos[j]
                        diff /= np.linalg.norm(diff)
                        push = (min_dist - d) / 2.0
                        pos[i] += diff * push
                        pos[j] -= diff * push
                    moved = True
        if not moved:
            break
    return pos


def _generate_lattice_seed(n, rng, noise_scale=0.2):
    """
    Generate a compact cluster configuration using a lattice-based approach.
    1. Estimate radius.
    2. Create a grid of candidate points.
    3. Select n closest to center.
    4. Add noise.
    """
    # Estimate radius of a sphere containing n atoms with density ~1
    # Volume = n, R = (3n/4pi)^(1/3)
    R = (3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0)
    # Add some buffer
    R_grid = R * 1.5
    
    # Create a grid of points
    # Number of points per dimension
    # We want enough points to pick n from. Let's generate 3*n points.
    n_points_target = max(100, 3 * n)
    
    # Generate random points in a sphere of radius R_grid
    # This is simpler than a regular grid and effectively samples the "lattice" of a disordered solid
    # To make it more "lattice-like" and structured, we can use a jittered grid.
    
    # Let's stick to a jittered grid for better structure
    # Grid spacing
    # We want roughly n points in a volume of ~R^3.
    # Let's create a grid in a cube of side 2*R_grid
    # Number of points per axis
    k = int(np.cbrt(n_points_target)) + 1
    if k < 3: k = 3
    
    x = np.linspace(-R_grid, R_grid, k)
    y = np.linspace(-R_grid, R_grid, k)
    z = np.linspace(-R_grid, R_grid, k)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    
    # Filter points inside sphere of radius R
    dists = np.linalg.norm(points, axis=1)
    mask = dists < R
    points = points[mask]
    
    # If we don't have enough points, increase grid resolution or radius
    if len(points) < n:
        # Fallback to random points in sphere
        while len(points) < n:
            p = rng.normal(0, R, 3)
            if np.linalg.norm(p) < R:
                points = np.vstack([points, p.reshape(1,3)])
        
    # Sort by distance to center and take the n closest
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    # Add noise to break symmetry
    noise = rng.normal(0, noise_scale, points.shape)
    points = points + noise
    
    # Center the cluster
    points -= np.mean(points, axis=0)
    
    # Enforce min distance
    points = _enforce_min_distance(points, 0.5)
    
    return points


def construct(instance):
    """Lattice Initiation + Multi-Start Relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter
    
    start_time = time.time()
    time_budget = 45.0  # seconds per instance
    
    rng = np.random.default_rng(42 + size)
    
    best_energy = np.inf
    best_positions = None
    
    # Determine number of attempts
    # Relax is fast, so we can try many
    n_attempts = 50
    
    for i in range(n_attempts):
        if time.time() - start_time > time_budget:
            break
            
        # Generate seed
        # Vary noise scale to explore different structural regimes
        noise_scale = 0.1 + 0.3 * (i / float(n_attempts))
        seed = _generate_lattice_seed(size, rng, noise_scale)
        
        # Relax
        try:
            energy, pos = relax(seed, 'morse', rho)
            # Check validity
            if energy < best_energy:
                # Ensure min distance on final result
                pos = _enforce_min_distance(pos, 0.5)
                # Re-evaluate energy after enforcement to be safe
                # Note: _morse_energy is O(N^2), which is fine for N<=80
                final_energy = _morse_energy(pos, rho)
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_positions = pos
        except Exception:
            continue
            
    # If no valid position found (unlikely), return a valid random one
    if best_positions is None:
        # Fallback
        R = (3.0 * size / (4.0 * np.pi)) ** (1.0 / 3.0)
        best_positions = rng.normal(0, R, (size, 3))
        best_positions = _enforce_min_distance(best_positions, 0.5)
        
    return best_positions
# EVOLVE-BLOCK-END
