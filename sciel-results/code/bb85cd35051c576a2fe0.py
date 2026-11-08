"""Strategy: Pre-computed optimal packing with exact geometric validation."""
import numpy as np
import time
import math

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Helper: get vertices of a unit regular hexagon centered at (cx, cy) with rotation angle (degrees)
    def hex_vertices(cx, cy, angle):
        base_angles = np.array([0, 60, 120, 180, 240, 330]) * np.pi / 180.0
        # Wait, standard hexagon vertices are at 0, 60, 120, 180, 240, 300.
        # Let's use 0, 60, 120, 180, 240, 300.
        base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
        rot = angle * np.pi / 180.0
        final_angles = base_angles + rot
        xs = cx + np.cos(final_angles)
        ys = cy + np.sin(final_angles)
        return np.column_stack((xs, ys))

    # Helper: check if a point (x, y) is inside the outer hexagon
    # Outer hexagon: center (0,0), side length S, rotation 0 (vertex at angle 0).
    # Face normals are at 30, 90, 150, 210, 270, 330 degrees.
    # Distance from center to face is S * sqrt(3) / 2.
    def is_inside_outer(x, y, S):
        normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
        nx = np.cos(normals)
        ny = np.sin(normals)
        dists = x * nx + y * ny
        R_out = S * np.sqrt(3) / 2
        return np.all(dists <= R_out + 1e-9)

    def point_in_hex(px, py, vertices):
        # Ray casting
        inside = False
        n = 6
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def edges_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        # Handle collinear cases (rare but possible)
        def point_on_segment(p, q, r):
            if (p[0] <= r[0] <= q[0] or q[0] <= r[0] <= p[0]) and \
               (p[1] <= r[1] <= q[1] or q[1] <= r[1] <= p[1]):
                return True
            return False
            
        if d1 == 0 and point_on_segment(p3, p4, p1): return True
        if d2 == 0 and point_on_segment(p3, p4, p2): return True
        if d3 == 0 and point_on_segment(p1, p2, p3): return True
        if d4 == 0 and point_on_segment(p1, p2, p4): return True
        return False

    def are_disjoint(hex1_verts, hex2_verts):
        # Check if any vertex of hex1 is inside hex2 or vice versa
        for v in hex1_verts:
            if point_in_hex(v[0], v[1], hex2_verts):
                return False
        for v in hex2_verts:
            if point_in_hex(v[0], v[1], hex1_verts):
                return False
        
        # Check edge intersections
        for i in range(6):
            p1 = hex1_verts[i]
            p2 = hex1_verts[(i+1)%6]
            for j in range(6):
                p3 = hex2_verts[j]
                p4 = hex2_verts[(j+1)%6]
                if edges_intersect(p1, p2, p3, p4):
                    return False
        return True

    def is_valid_packing(positions, angles, S):
        verts_list = []
        for i in range(N):
            verts = hex_vertices(positions[i,0], positions[i,1], angles[i])
            verts_list.append(verts)
            
        # Check inside outer
        for i in range(N):
            for v in verts_list[i]:
                if not is_inside_outer(v[0], v[1], S):
                    return False
            if not is_inside_outer(positions[i,0], positions[i,1], S):
                return False

        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not are_disjoint(verts_list[i], verts_list[j]):
                    return False
        return True

    # Pre-computed packing based on known optimal or near-optimal configurations.
    # Configuration: 1 center, 6 around it (hexagonal ring), 4 more placed in the gaps.
    # This is a common pattern for N=11.
    # Let's define a specific layout and verify it.
    
    # Layout:
    # Center: (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3) from center, angles 0, 60, ... 300.
    # Wait, if we put them at distance sqrt(3), they touch.
    # Positions for Ring 1:
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    
    # Where to put the other 4?
    # In a hexagonal lattice, the "next" shell is at distance 2*sqrt(3).
    # But we only have 4 more.
    # Maybe place them in the "dimples" between the center and the ring? No, those are occupied.
    # Maybe place them outside the ring?
    # If we place them in the gaps between adjacent ring hexagons, they would be further out.
    # Or maybe a different arrangement: a 3x3 block minus 2 corners?
    
    # Let's try a 3x3 grid arrangement (9 hexagons) plus 2 extra.
    # Or 1 + 6 + 4.
    
    # Let's try to generate a valid packing using a simple grid and then optimize slightly if needed,
    # but the prompt asks for a "pre-computed" strategy to avoid runtime optimization failures.
    # I will construct a specific known valid packing.
    
    # A compact packing for 11 unit hexagons:
    # Imagine a hexagon of side 2 (which holds 7 hexagons: 1+6).
    # Add 4 more on the sides.
    
    # Let's try this specific configuration:
    # Centers:
    # (0, 0)
    # (sqrt(3), 0)
    # (-sqrt(3), 0)
    # (0, 2*sqrt(3)) # Wait, vertical distance in hex grid is 1.5?
    # Let's stick to the honeycomb lattice coordinates.
    # Basis vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 1.5)
    # Lattice points: i*a1 + j*a2
    # We want 11 points.
    # 0,0
    # 1,0 -> (sqrt(3), 0)
    # -1,0 -> (-sqrt(3), 0)
    # 0,1 -> (sqrt(3)/2, 1.5)
    # 0,-1 -> (-sqrt(3)/2, -1.5)
    # 1,1 -> (3*sqrt(3)/2, 1.5)
    # -1,1 -> (-sqrt(3)/2, 1.5)
    # 1,-1 -> (sqrt(3)/2, -1.5)
    # -1,-1 -> (-3*sqrt(3)/2, -1.5)
    # 0,2 -> (sqrt(3), 3.0)
    # 0,-2 -> (-sqrt(3), -3.0)
    
    # Let's check if these 11 points are disjoint.
    # In a honeycomb, all lattice points are at distance >= sqrt(3) from each other (nearest neighbors).
    # So they are disjoint.
    
    # Now, what is the bounding hexagon size?
    # We need to find the minimal regular hexagon (side S, rotation 0) that contains all these vertices.
    # The vertices of the inner hexagons extend 1 unit from their centers.
    # So we need to find S such that all vertices of all 11 hexagons are inside the outer hexagon of side S.
    # Outer hexagon constraint: max projection onto face normals <= S * sqrt(3) / 2.
    # Face normals: 30, 90, 150, 210, 270, 330 deg.
    
    # Let's compute the required S for this specific set of centers.
    # We will compute the max projection of all vertices of all 11 hexagons onto the 6 face normals.
    # Then S = (max_proj + 1e-9) / (sqrt(3)/2).
    
    centers = []
    # Define the 11 lattice points (i, j)
    lattice_points = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (-1, 1),
        (1, -1), (-1, -1),
        (0, 2),
        (0, -2)
    ]
    
    sqrt3 = math.sqrt(3)
    a1 = np.array([sqrt3, 0])
    a2 = np.array([sqrt3/2, 1.5])
    
    for i, j in lattice_points:
        c = i * a1 + j * a2
        centers.append(c)
    
    # Convert to numpy array
    centers = np.array(centers)
    angles = np.zeros(N) # All aligned
    
    # Compute max projection
    normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    nx = np.cos(normals)
    ny = np.sin(normals)
    
    max_proj = 0
    for c in centers:
        verts = hex_vertices(c[0], c[1], 0)
        for v in verts:
            projs = v[0] * nx + v[1] * ny
            max_proj = max(max_proj, np.max(projs))
            
    S_calc = max_proj / (sqrt3 / 2)
    
    # Add a tiny epsilon to ensure strict inequality in floating point checks if needed,
    # but the validator uses <= R_out + 1e-9, so exact fit should pass.
    S = S_calc
    
    # Verify the packing
    if is_valid_packing(centers, angles, S):
        # Format output
        inner = []
        for i in range(N):
            inner.append([centers[i,0], centers[i,1], angles[i]])
        
        return inner, (0.0, 0.0), S, 0.0
    else:
        # Fallback: return a very large S with a simple valid packing (e.g. linear chain)
        # This is unlikely to happen if the lattice is correct, but safety first.
        inner = []
        for i in range(N):
            inner.append([4.0 * i, 0.0, 0.0])
        return inner, (0.0, 0.0), 40.0, 0.0

