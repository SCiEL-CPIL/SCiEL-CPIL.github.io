import numpy as np
from scipy.optimize import minimize

N = 38

# EVOLVE-BLOCK-START
def construct():
    # 1. Define the FCC lattice parameters
    # The minimum of the LJ potential is at r = 2^(1/6)
    r_min = 2.0 ** (1.0 / 6.0)
    # In an FCC lattice, the nearest neighbor distance is a / sqrt(2)
    # So, a = r_min * sqrt(2)
    a = r_min * np.sqrt(2.0)
    
    # 2. Generate candidate FCC lattice points
    # We need to generate enough points to ensure we have at least 38 close to the center.
    # A truncated octahedron of 38 atoms is roughly a sphere of radius ~3-4 lattice units.
    # Let's generate a cubic block of integer coordinates from -5 to 5.
    coords = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                # FCC condition: sum of indices must be even
                if (i + j + k) % 2 == 0:
                    # Scale to physical coordinates
                    x = i * a
                    y = j * a
                    z = k * a
                    coords.append((x, y, z))
    
    # Convert to numpy array
    coords = np.array(coords)
    
    # 3. Select the 38 atoms closest to the origin
    # Calculate squared distance to origin for sorting (faster than sqrt)
    dists = np.sum(coords**2, axis=1)
    # Get indices of the 38 smallest distances
    idx = np.argsort(dists)[:N]
    positions = coords[idx].copy()
    
    # 4. Center the cluster
    # The origin in our generation might not be exactly the center of mass of the selected subset
    # due to the discrete nature of the lattice and the cutoff. Centering helps optimization stability.
    positions -= np.mean(positions, axis=0)
    
    # 5. Local relaxation using L-BFGS-B
    # Define the energy function
    def lj_energy(pos):
        # pos is (N, 3)
        # Compute all pair distances
        # To avoid N^2 memory blowup for small N, we can use broadcasting
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero (diagonal elements)
        # We only care about i < j, so we can mask the upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        dist_sq = dist_sq[mask]
        
        # Calculate LJ energy: 4 * (r^-12 - r^-6)
        # r^-6 = (r^2)^-3
        r_inv6 = 1.0 / (dist_sq**3)
        r_inv12 = r_inv6 * r_inv6
        energy = 4.0 * np.sum(r_inv12 - r_inv6)
        return energy
    
    # Check initial energy
    initial_energy = lj_energy(positions)
    
    # Optimization
    # L-BFGS-B is robust for this kind of smooth potential
    result = minimize(
        lj_energy, 
        positions, 
        method='L-BFGS-B', 
        options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15, 'disp': False}
    )
    
    # Use the optimized positions
    final_positions = result.x
    
    # Final check: ensure atoms are at least 0.5 apart
    # Calculate min distance
    diff = final_positions[:, np.newaxis, :] - final_positions[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    min_dist = np.sqrt(np.min(dist_sq[mask]))
    
    if min_dist < 0.5:
        # This should not happen with FCC, but if it does, we might need to rescale or reject
        # For safety, if invalid, return the pre-optimization positions which are valid
        return positions
        
    return final_positions
# EVOLVE-BLOCK-END
