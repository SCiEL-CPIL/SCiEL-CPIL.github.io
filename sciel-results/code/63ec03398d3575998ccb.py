import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate 11 unit hexagons in a hexagonal lattice pattern
    # Axial coordinates (q, r)
    # Center: (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: 4 selected neighbors (side-middles for compactness)
    
    axial_coords = [
        (0, 0),      # Center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1), # Ring 1
        (2, -1), (1, -1), (-2, 1), (-1, 1) # 4 from Ring 2 (selected for symmetry/compactness)
    ]
    
    # Ensure we have exactly N hexagons
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    elif len(axial_coords) < N:
        # Fallback: add more from Ring 2 if needed
        extra = [(2, 0), (2, -2), (0, -2), (-2, 0), (-2, 2), (0, 2), (1, 1), (-1, -1)]
        for c in extra:
            if len(axial_coords) >= N: break
            if c not in axial_coords:
                axial_coords.append(c)
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    centers = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Inner hexagons: (x, y, angle_degrees)
    # All aligned at 0 degrees
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Get vertices of all inner hexagons to define the shape to be enclosed
    # A unit hexagon centered at (cx, cy) with angle 0 has vertices:
    # (cx + cos(60*i), cy + sin(60*i)) for i=0..5
    vertices = []
    for cx, cy, _ in inner:
        for i in range(6):
            angle = np.deg2rad(60 * i)
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            vertices.append((vx, vy))
    
    vertices = np.array(vertices)
    
    def calculate_min_side(params):
        cx, cy, theta_rad = params
        center = np.array([cx, cy])
        
        # Normals to the 3 pairs of opposite sides of the outer hexagon
        # The sides are perpendicular to angles theta, theta+60, theta+120
        angles = [theta_rad, theta_rad + np.pi/3, theta_rad + 2*np.pi/3]
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
        
        # Vector from center to all vertices
        diffs = vertices - center
        
        # Projection of vertices onto normals
        # proj[i, k] = dot(v_i - c, n_k)
        projs = diffs @ normals.T
        
        # The maximum absolute projection for each normal
        max_projs = np.max(np.abs(projs), axis=0)
        
        # The side length S must satisfy:
        # max_proj_k <= S * sqrt(3) / 2
        # S >= max_proj_k / (sqrt(3) / 2)
        # S >= 2 * max_proj_k / sqrt(3)
        
        min_sides = max_projs / (np.sqrt(3) / 2.0)
        return np.max(min_sides)
    
    # Initial guess: centroid of vertices, angle 0
    cx_init = np.mean(vertices[:, 0])
    cy_init = np.mean(vertices[:, 1])
    theta_init = 0.0
    
    res = minimize(calculate_min_side, [cx_init, cy_init, theta_init], method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_cx, best_cy, best_theta = res.x
    best_side = res.fun
    
    # Final check: ensure it's valid (it should be by construction)
    # Return inner, (center_x, center_y), side, angle_degrees
    return inner, (float(best_cx), float(best_cy)), float(best_side), float(np.rad2deg(best_theta))
# EVOLVE-BLOCK-END
