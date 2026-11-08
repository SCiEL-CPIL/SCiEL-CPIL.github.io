import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def axial_to_xy(q, r, s=1.0):
    """Convert axial coordinates to Cartesian for flat-top hexagons of side s."""
    x = 1.5 * s * q
    y = np.sqrt(3) * s * (r + 0.5 * q)
    return x, y

def construct():
    start_time = time.time()
    
    # 1. Define the 7 central hexagons (Ring 0 + Ring 1)
    # Axial coordinates (q, r)
    central_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)
    ]
    
    # 2. Define the 12 Ring 2 hexagons
    ring2_axial = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1),
        (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)
    ]
    
    best_side = float('inf')
    best_centers = None
    best_outer_center = (0, 0)
    best_outer_angle = 0.0
    
    # Pre-compute central centers
    central_centers = [axial_to_xy(q, r) for q, r in central_axial]
    
    # Iterate over all combinations of 4 from Ring 2
    for combo in combinations(range(12), 4):
        # Check time
        if time.time() - start_time > 130:
            break
            
        extra_centers = [axial_to_xy(ring2_axial[i][0], ring2_axial[i][1]) for i in combo]
        all_centers = central_centers + extra_centers
        
        # Compute all vertices
        all_verts = []
        for cx, cy in all_centers:
            verts = hex_vertices(cx, cy, 1.0, 0.0)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Find minimal enclosing regular hexagon
        # Try a few rotations. Due to symmetry of the hexagon, 0-30 is enough,
        # but the cluster might be asymmetric, so we check a range.
        # The optimal rotation is likely to align with the symmetry of the cluster.
        # We'll check 0, 15, 30, 45, 60.
        
        min_side_for_config = float('inf')
        best_c_for_config = (0,0)
        best_theta_for_config = 0.0
        
        # Optimization for center (x, y) given a fixed rotation theta
        def objective(C, vertices, normals):
            # C is [x, y]
            dx = vertices[:, 0] - C[0]
            dy = vertices[:, 1] - C[1]
            # Project onto normals
            # normals shape (6, 2)
            projs = np.dot(vertices - C, normals.T) # (N_verts, 6)
            # Max absolute projection
            return np.max(np.abs(projs))
        
        # We want to minimize max_proj. Then S = 2/sqrt(3) * min_max_proj
        # Actually, S = 2/sqrt(3) * max_proj is the side length if max_proj is the distance to the side.
        # Wait. Distance to side is apothem = S * sqrt(3)/2.
        # So S = 2/sqrt(3) * apothem.
        # The constraint is proj <= S * sqrt(3)/2.
        # So S >= 2/sqrt(3) * max_proj.
        
        for theta_deg in [0, 15, 30, 45, 60]:
            theta = np.radians(theta_deg)
            # Normals for outer hexagon
            # Face normals are at theta + 30 + i*60
            normals = np.array([
                [np.cos(theta + np.pi/6 + i * np.pi/3), np.sin(theta + np.pi/6 + i * np.pi/3)]
                for i in range(6)
            ])
            
            # Initial guess for center: mean of vertices
            x0 = all_verts[:, 0].mean()
            y0 = all_verts[:, 1].mean()
            x0_y0 = [x0, y0]
            
            # Optimize
            try:
                res = minimize(
                    lambda C: objective(C, all_verts, normals),
                    x0_y0,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                min_max_proj = res.fun
                side = (2.0 / np.sqrt(3)) * min_max_proj
                
                if side < min_side_for_config:
                    min_side_for_config = side
                    best_c_for_config = res.x
                    best_theta_for_config = theta_deg
                    
            except Exception:
                pass
        
        if min_side_for_config < best_side:
            best_side = min_side_for_config
            best_centers = all_centers
            best_outer_center = best_c_for_config
            best_outer_angle = best_theta_for_config

    # If no valid config found (shouldn't happen), fallback to a simple one
    if best_centers is None:
        # Just use the central 7 and 4 arbitrary
        extra_centers = [axial_to_xy(ring2_axial[i][0], ring2_axial[i][1]) for i in range(4)]
        best_centers = central_centers + extra_centers
        all_verts = np.vstack([hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in best_centers])
        best_outer_center = [all_verts[:, 0].mean(), all_verts[:, 1].mean()]
        best_outer_angle = 0.0
        # Calculate side
        normals = np.array([
            [np.cos(np.pi/6 + i * np.pi/3), np.sin(np.pi/6 + i * np.pi/3)]
            for i in range(6)
        ])
        dx = all_verts[:, 0] - best_outer_center[0]
        dy = all_verts[:, 1] - best_outer_center[1]
        projs = np.stack([dx * normals[i,0] + dy * normals[i,1] for i in range(6)], axis=1)
        max_proj = np.max(np.abs(projs))
        best_side = (2.0 / np.sqrt(3)) * max_proj

    # Build return values
    inner = [[float(cx), float(cy), 0.0] for cx, cy in best_centers]
    # Add small safety margin to side length to avoid floating point issues in evaluator
    best_side *= 1.0001
    
    return inner, (float(best_outer_center[0]), float(best_outer_center[1])), float(best_side), float(best_outer_angle)
