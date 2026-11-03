"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    sqrt3 = np.sqrt(3)
    
    # 1. Generate a set of candidate compact lattice arrangements for 12 hexagons.
    # We use axial coordinates (q, r).
    # Distance between centers in hex grid:
    # dx = 1.5 * (q1 - q2)
    # dy = sqrt3 * (r1 - r2 + 0.5 * (q1 - q2))
    # To avoid overlap, we need distance >= 2.0 (touching is allowed).
    # In axial coords, valid non-overlapping pairs satisfy specific distance metrics.
    # A simple check: convert to pixel and check dist >= 2.0 - eps.
    
    def axial_to_pixel(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y

    def is_valid_arrangement(coords):
        pixels = [axial_to_pixel(q, r) for q, r in coords]
        for i in range(len(pixels)):
            for j in range(i+1, len(pixels)):
                dx = pixels[i][0] - pixels[j][0]
                dy = pixels[i][1] - pixels[j][1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 2.0 - 1e-9:
                    return False
        return True

    # Candidate 1: The "Flower" + 6 surrounding (Center + Ring 1 + Ring 2 partial)
    # Center (0,0)
    # Ring 1 (6 hexes): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # That's 7 hexes. We need 5 more.
    # Ring 2 neighbors of Ring 1 that don't overlap each other or existing.
    # Let's just generate random valid clusters or use known shapes.
    
    # Known compact shape for 12:
    # Row 0: 4 hexes
    # Row 1: 3 hexes (offset)
    # Row -1: 3 hexes (offset)
    # Row 2: 2 hexes
    # Let's define a few specific topologies.
    
    topologies = []
    
    # Topology A: 4-3-3-2 (as in original code)
    coords_A = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, -1), (0, -1), (1, -1),
        (0, 2), (1, 2)
    ]
    if is_valid_arrangement(coords_A):
        topologies.append(coords_A)

    # Topology B: 3-4-3-2?
    coords_B = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, -1), (0, -1), (1, -1),
        (0, 2), (1, 2)
    ]
    if is_valid_arrangement(coords_B):
        topologies.append(coords_B)
        
    # Topology C: 2-3-4-3-? No, 12 items.
    # Let's try a more symmetric "hexagon of hexagons" minus corners?
    # A full hexagon of radius 2 has 19 hexes.
    # A full hexagon of radius 1 has 7 hexes.
    # 12 is 7 + 5.
    
    # Let's add a random search for better topologies if time permits, 
    # but for now, we will optimize the placement of these fixed topologies 
    # within the outer hexagon.
    
    best_side = np.inf
    best_inner = None
    best_center = (0,0)
    best_angle = 0.0
    
    # For each topology, we will:
    # 1. Center the topology at (0,0).
    # 2. Search for the optimal outer hexagon center and angle.
    #    Since the inner shape is fixed (up to translation/rotation of the whole cluster, 
    #    but we fixed lattice orientation), we can just translate the cluster.
    #    Actually, we can rotate the entire cluster? 
    #    The task allows inner hexagons to rotate. 
    #    However, if we fix them to lattice orientation (angle 0), we lose the ability 
    #    to rotate the cluster. 
    #    BUT, a regular hexagon has 6-fold symmetry. Rotating the cluster by 60 deg 
    #    is equivalent to rotating the outer hexagon by 60 deg. 
    #    So we can fix inner angles to 0, and search outer angle in [0, 60).
    
    for coords in topologies:
        # Convert to pixel and center
        pixels = np.array([axial_to_pixel(q, r) for q, r in coords])
        center = np.mean(pixels, axis=0)
        pixels -= center # Center at origin
        
        # Now optimize outer hexagon.
        # Variables: outer_cx, outer_cy, outer_angle
        # Objective: min side length such that all inner hexes are inside.
        
        def calc_side(outer_cx, outer_cy, outer_angle_deg):
            oa = np.radians(outer_angle_deg)
            max_inradius = 0.0
            for px, py in pixels:
                # Check all 6 vertices of this inner hex? 
                # No, for a convex polygon inside another, it suffices to check 
                # the vertices of the inner polygon against the half-planes of the outer.
                # Inner hex vertices relative to its center (px, py) with angle 0:
                # v_k = (px + cos(k*pi/3), py + sin(k*pi/3))
                # Wait, the inner hex is centered at (px, py).
                # Its vertices are at distance 1 from (px, py).
                # We need to check if these vertices are inside the outer hex.
                
                # Outer hex face normals:
                # theta_k = oa + k*pi/3 + pi/6
                for k in range(6):
                    theta = oa + k * np.pi / 3 + np.pi / 6
                    nx = np.cos(theta)
                    ny = np.sin(theta)
                    
                    # Project all 6 vertices of inner hex onto this normal
                    # Vertex j of inner hex:
                    # vx = px + cos(j*pi/3)
                    # vy = py + sin(j*pi/3)
                    # Projection = (px + cos_j) * nx + (py + sin_j) * ny
                    # = px*nx + py*ny + cos_j*nx + sin_j*ny
                    # We need max projection over all j and all inner hexes.
                    
                    # Precompute cos/sin for inner hex vertices (angle 0)
                    # c = [1, 0.5, -0.5, -1, -0.5, 0.5]
                    # s = [0, 0.866, 0.866, 0, -0.866, -0.866]
                    
                    # Max over j of (cos_j*nx + sin_j*ny) is the support function of the inner hex in direction (nx, ny).
                    # For a unit hexagon, the support function h(theta) is:
                    # If theta is aligned with a face normal, h = sqrt(3)/2 (inradius)
                    # If theta is aligned with a vertex, h = 1 (circumradius)
                    # Generally, h(theta) = cos(theta - nearest_vertex_angle) * ... 
                    # Actually, simpler: just compute the 6 dots.
                    
                    proj = px * nx + py * ny
                    max_v = -np.inf
                    for j in range(6):
                        vx = np.cos(j * np.pi / 3)
                        vy = np.sin(j * np.pi / 3)
                        val = proj + vx * nx + vy * ny
                        if val > max_v:
                            max_v = val
                    if max_v > max_inradius:
                        max_inradius = max_v
            
            # Side = 2 * inradius / sqrt(3)
            return 2.0 * max_inradius / sqrt3

        # Optimization
        # Use Nelder-Mead on (cx, cy, angle)
        # Initial guess: (0,0,0)
        
        x0 = [0.0, 0.0, 0.0]
        
        def obj(p):
            return calc_side(p[0], p[1], p[2])
        
        try:
            result = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_side:
                best_side = result.fun
                best_inner = [[float(pixels[i][0]), float(pixels[i][1]), 0.0] for i in range(N)]
                best_center = (float(result.x[0]), float(result.x[1]))
                best_angle = float(result.x[2])
        except Exception:
            pass

        # Also try a few fixed angles to ensure we don't miss global minima due to local optima
        for ang in [0, 30, 15, 45]:
             # For fixed angle, optimize cx, cy
             def obj_xy(p):
                 return calc_side(p[0], p[1], ang)
             x0_xy = [0.0, 0.0]
             res_xy = minimize(obj_xy, x0_xy, method='Nelder-Mead', options={'maxiter': 200})
             if res_xy.fun < best_side:
                 best_side = res_xy.fun
                 best_inner = [[float(pixels[i][0]), float(pixels[i][1]), 0.0] for i in range(N)]
                 best_center = (float(res_xy.x[0]), float(res_xy.x[1]))
                 best_angle = float(ang)

    # If no topology was valid or found, return a fallback
    if best_inner is None:
        # Fallback: simple grid
        coords = [(-1, 0), (0, 0), (1, 0), (2, 0),
                  (-1, 1), (0, 1), (1, 1),
                  (-1, -1), (0, -1), (1, -1),
                  (0, 2), (1, 2)]
        pixels = np.array([axial_to_pixel(q, r) for q, r in coords])
        center = np.mean(pixels, axis=0)
        pixels -= center
        best_inner = [[float(pixels[i][0]), float(pixels[i][1]), 0.0] for i in range(N)]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = calc_side(0, 0, 0)

    return best_inner, best_center, float(best_side), float(best_angle)

# EVOLVE-BLOCK-END
