import numpy as np
from scipy.optimize import differential_evolution

N = 11

def construct():
    # Define the rigid cluster of 11 hexagons
    # Using a lattice of flat-top hexagons
    # Lattice vectors:
    a = np.array([np.sqrt(3), 0])
    b = np.array([np.sqrt(3)/2, 1.5])
    
    # 11 positions in lattice coordinates (i, j)
    # (0,0)
    # (1,0), (-1,0)
    # (0,1), (0,-1)
    # (1,1), (1,-1)
    # (-1,1), (-1,-1)
    # (2,0)
    # (0,2)
    lattice_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (1, -1),
        (-1, 1), (-1, -1),
        (2, 0),
        (0, 2)
    ]
    
    # Calculate actual centers
    centers = []
    for i, j in lattice_coords:
        c = i * a + j * b
        centers.append(c)
        
    centers = np.array(centers) # shape (11, 2)
    
    # Pre-compute vertices for each hexagon in the local frame (angle 0)
    # Unit hexagon vertices (flat-top)
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    local_vx = np.cos(angles)
    local_vy = np.sin(angles)
    local_vertices = np.column_stack((local_vx, local_vy)) # shape (6, 2)
    
    # All vertices for all 11 hexagons in local frame
    # shape (11, 6, 2)
    all_vertices = centers[:, np.newaxis, :] + local_vertices[np.newaxis, :, :]
    
    # Flatten for easier manipulation if needed, but we can work with (11, 6, 2)
    # We need to rotate and translate these points.
    
    def cost(params):
        dx, dy, theta = params
        
        # Rotation matrix
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate all vertices
        # all_vertices shape (11, 6, 2)
        # R shape (2, 2)
        # Use einsum or matrix multiplication
        rotated = np.einsum('ij,klj->kli', R, all_vertices)
        
        # Translate
        translated = rotated + np.array([dx, dy])
        
        # Calculate max projection on 3 axes of the outer hexagon
        # Axes: 0, pi/3, 2pi/3
        axes = np.array([
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2]
        ])
        
        # Project translated vertices onto each axis
        # translated shape (11, 6, 2)
        # axes shape (3, 2)
        # projections shape (11, 6, 3)
        projections = np.einsum('kl,ij->kli', translated, axes)
        
        # Find max absolute projection
        max_proj = np.max(np.abs(projections))
        
        # Side length S
        # Distance from center to side is S * sqrt(3) / 2
        # So max_proj <= S * sqrt(3) / 2
        S = max_proj * 2 / np.sqrt(3)
        
        return S

    # Optimization
    # Bounds: dx, dy in [-5, 5], theta in [0, pi/3]
    bounds = [(-5, 5), (-5, 5), (0, np.pi/3)]
    
    # Use differential evolution for robustness, though gradient-based might work too
    result = differential_evolution(cost, bounds, maxiter=200, popsize=15, tol=1e-6, seed=42, polish=True)
    
    if result.success or result.fun < 10: # Check if we found a reasonable solution
        dx, dy, theta = result.x
    else:
        # Fallback: just center at 0, theta 0
        dx, dy, theta = 0, 0, 0
        
    # Calculate final S
    S = cost([dx, dy, theta])
    
    # Construct the inner hexagons
    inner = []
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    
    for i in range(N):
        # Rotate center
        cx, cy = centers[i]
        new_cx = c * cx - s * cy + dx
        new_cy = s * cx + c * cy + dy
        # Angle is theta
        inner.append([new_cx, new_cy, np.degrees(theta)])
        
    return inner, (0.0, 0.0), S, 0.0

