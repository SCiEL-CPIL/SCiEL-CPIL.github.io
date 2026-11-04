import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Unit hexagon vertices (centered at origin, side 1)
    angles = np.arange(6) * np.pi / 3
    unit_vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Outer hexagon normals (for side length S, distance to edge is S*sqrt(3)/2)
    # Normals are at pi/6 + k*pi/3
    outer_angles = np.pi/6 + np.arange(6) * np.pi / 3
    outer_normals = np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
    
    # Initial guess: Compact hexagonal cluster
    # Lattice vectors
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate a set of lattice points
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * a1 + j * a2
            candidates.append(p)
    candidates = np.array(candidates)
    
    # Sort by distance to origin and take first 12
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)[:N]
    x0 = candidates[idx]
    
    # Flatten x0 for optimization
    x0_flat = x0.flatten()
    
    def objective(x):
        x = x.reshape(-1, 2)
        # Compute required side length S
        v = x[:, np.newaxis, :] + unit_vertices[np.newaxis, :, :]
        proj = np.einsum('ijk,lk->ijl', v, outer_normals)
        max_proj = np.max(proj)
        S = max_proj / (np.sqrt(3) / 2)
        
        # Overlap penalty
        penalty = 0.0
        d_min = np.sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(x[i] - x[j])
                if d < d_min:
                    penalty += (d_min - d)**2
        
        return S + 10.0 * penalty

    # Run optimization
    res = minimize(objective, x0_flat, method='L-BFGS-B', options={'maxiter': 1000})
    
    x_opt = res.x.reshape(-1, 2)
    
    # Calculate final S
    v = x_opt[:, np.newaxis, :] + unit_vertices[np.newaxis, :, :]
    proj = np.einsum('ijk,lk->ijl', v, outer_normals)
    max_proj = np.max(proj)
    S = max_proj / (np.sqrt(3) / 2)
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([x_opt[i][0], x_opt[i][1], 0.0])
    
    center = [0.0, 0.0]
    side = S
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
