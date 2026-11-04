"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # We need to pack N unit regular hexagons inside a regular hexagon of minimum side length.
    # The inner hexagons can translate and rotate.
    # The outer hexagon center, side length, and rotation are also variables to optimize.
    # However, for simplicity and due to symmetry, we can fix the outer hexagon center at (0,0)
    # and rotation at 0. The side length will be determined by the maximum distance of any
    # inner hexagon vertex from the center, projected onto the outer hexagon's support function.
    
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        vertices = []
        for k in range(6):
            theta = angle + k * np.pi / 3
            x = cx + s * np.cos(theta)
            y = cy + s * np.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)
    
    def poly_overlap(v1, v2):
        """Check if two convex polygons (given as arrays of vertices) overlap using SAT."""
        # SAT: check if there's a separating axis
        # For each edge of each polygon, project all vertices onto the normal
        # If there's a gap, they don't overlap
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i+1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                # Project all vertices of both polygons onto this normal
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                if max1 < min2 or max2 < min1:
                    return False  # Separating axis found, no overlap
        return True  # No separating axis, they overlap

    def compute_S(params):
        """Compute the outer side length S required to contain all hexagons."""
        n = len(params) // 3
        S = 0.0
        for i in range(n):
            cx = params[3*i]
            cy = params[3*i+1]
            angle = params[3*i+2]
            verts = hex_vertices(cx, cy, angle)
            for v in verts:
                for k in range(6):
                    theta_n = np.pi/6 + k * np.pi/3
                    proj = v[0]*np.cos(theta_n) + v[1]*np.sin(theta_n)
                    S = max(S, abs(proj))
        return S / (np.sqrt(3)/2)
    
    def objective(params):
        """Compute the outer side length S and penalty for overlaps."""
        n = len(params) // 3
        S = compute_S(params)
        
        # Overlap penalty
        penalty = 0.0
        vertices_list = []
        for i in range(n):
            cx = params[3*i]
            cy = params[3*i+1]
            angle = params[3*i+2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        for i in range(n):
            for j in range(i+1, n):
                if poly_overlap(vertices_list[i], vertices_list[j]):
                    # Calculate penetration depth roughly
                    c1 = vertices_list[i].mean(axis=0)
                    c2 = vertices_list[j].mean(axis=0)
                    dist = np.linalg.norm(c1 - c2)
                    # Min distance for no overlap is ~sqrt(3) for aligned, up to 2 for worst case?
                    # Actually, min distance between centers for two unit hexagons to not overlap
                    # is at least sqrt(3) (edge-to-edge).
                    # If they overlap, we penalize based on how much they overlap.
                    # A simple penalty: (sqrt(3) - dist)^2 if dist < sqrt(3)
                    # But SAT says they overlap. Let's use a large constant penalty to push them apart.
                    penalty += 50.0 * (1.0 + (2.0 - dist))
        return S + penalty

    # Initial guess: hexagonal lattice
    # For N=11, use 1 center, 6 ring 1, 4 ring 2.
    # To improve the initial guess, we can slightly rotate the ring 2 hexagons or place them
    # in a way that minimizes the bounding box.
    
    params = np.zeros(3*N)
    params[0:3] = [0, 0, 0]
    idx = 1
    # Ring 1: 6 hexagons at distance sqrt(3) from center
    for k in range(6):
        theta = k * np.pi/3
        cx = np.sqrt(3) * np.cos(theta)
        cy = np.sqrt(3) * np.sin(theta)
        params[3*idx] = cx
        params[3*idx+1] = cy
        params[3*idx+2] = 0
        idx += 1
    # Ring 2: need 4 more to reach 11
    # Place them at distance 2*sqrt(3) from center, at angles 30, 90, 150, 210 (offset by 30 deg)
    # Or 0, 60, 120, 180?
    # Let's try 0, 60, 120, 180 first.
    for k in range(4):
        theta = k * np.pi/3
        cx = 2*np.sqrt(3) * np.cos(theta)
        cy = 2*np.sqrt(3) * np.sin(theta)
        params[3*idx] = cx
        params[3*idx+1] = cy
        params[3*idx+2] = 0
        idx += 1
    
    # Optimize using Nelder-Mead with better settings
    # We run multiple iterations or a single long one.
    # To ensure we don't time out, we limit maxiter.
    result = minimize(objective, params, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
    
    # If the result is bad (high penalty), try a different initial configuration
    # e.g. rotate ring 2 by 30 degrees
    if result.fun > 4.0: # Threshold for bad result
        params2 = params.copy()
        # Rotate ring 2 hexagons
        for i in range(7, 11):
            cx = params2[3*i]
            cy = params2[3*i+1]
            angle = params2[3*i+2]
            # Rotate center by 30 degrees
            new_angle = angle + np.pi/6
            cx2 = cx * np.cos(np.pi/6) - cy * np.sin(np.pi/6)
            cy2 = cx * np.sin(np.pi/6) + cy * np.cos(np.pi/6)
            params2[3*i] = cx2
            params2[3*i+1] = cy2
            params2[3*i+2] = new_angle
        
        result2 = minimize(objective, params2, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if result2.fun < result.fun:
            result = result2
    
    opt_params = result.x
    inner = []
    for i in range(N):
        cx = opt_params[3*i]
        cy = opt_params[3*i+1]
        angle = np.degrees(opt_params[3*i+2])
        inner.append([cx, cy, angle])
    
    # Compute the final S
    S = compute_S(opt_params)
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
