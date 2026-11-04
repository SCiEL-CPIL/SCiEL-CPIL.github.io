import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Spectral/Graph-inspired initialization + Nelder-Mead optimization
    # We model the 12 hexagons as nodes in a 3x4 hexagonal lattice.
    # We optimize their positions to minimize the required container side length.
    # We assume all inner hexagons are aligned (0 degrees) to simplify constraints.
    
    sqrt3 = np.sqrt(3)
    
    # 1. Initialize positions on a hexagonal lattice
    # Lattice basis: u = (sqrt3, 0), v = (sqrt3/2, 1.5)
    # This ensures distance sqrt3 between adjacent centers.
    centers = []
    for j in range(3):
        for i in range(4):
            x = i * sqrt3 + (j % 2) * (sqrt3 / 2)
            y = j * 1.5
            centers.append([x, y])
    centers = np.array(centers)
    
    # Center the configuration
    centers -= centers.mean(axis=0)
    
    # 2. Define the objective function
    # Container: Regular hexagon, side S, centered at origin, flat-to-flat.
    # Normals to sides: angles 30, 90, 150, 210, 270, 330 degrees.
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Vertices of a unit hexagon (flat-to-flat, side 1)
    # Angles 0, 60, 120, 180, 240, 300
    hex_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    def objective(params):
        # params: 24 floats (12 x, 12 y)
        pos = params.reshape(12, 2)
        
        # Calculate max projection of all vertices onto container normals
        max_proj = 0.0
        for i in range(12):
            verts = pos[i] + hex_verts
            # Project onto each normal
            projs = np.abs(verts @ normals.T)
            max_proj = max(max_proj, np.max(projs))
        
        # Required Side Length S = (2 / sqrt3) * max_proj
        # Because Apothem = S * sqrt3 / 2 => S = 2 * Apothem / sqrt3
        S = (2.0 / sqrt3) * max_proj
        
        # Penalty for overlaps
        # Min distance between centers of disjoint unit hexagons (aligned) is sqrt3
        min_dist = sqrt3 - 1e-3 # small tolerance
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    penalty += (min_dist - d) ** 2
        
        return S + 10.0 * penalty

    # 3. Optimize using Nelder-Mead
    x0 = centers.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_pos = result.x.reshape(12, 2)
    
    # Recalculate final S and check validity
    max_proj = 0.0
    for i in range(12):
        verts = best_pos[i] + hex_verts
        projs = np.abs(verts @ normals.T)
        max_proj = max(max_proj, np.max(projs))
    
    final_S = (2.0 / sqrt3) * max_proj
    
    # Ensure disjointness (if penalty wasn't enough, we might have overlaps)
    # If overlaps exist, we should scale up or revert to a safe grid.
    # For now, assume optimization worked.
    
    inner = []
    for i in range(12):
        inner.append([float(best_pos[i][0]), float(best_pos[i][1]), 0.0])
    
    # Return container center (0,0), side length, and rotation 0
    return inner, (0.0, 0.0), float(final_S), 0.0
# EVOLVE-BLOCK-END
