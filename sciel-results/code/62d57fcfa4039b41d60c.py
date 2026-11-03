"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    
    # Vertices of a unit regular hexagon (flat-top, side 1)
    # Side length 1.
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees?
    # Flat-top usually has vertices at 0, 60... relative to center?
    # If side is 1, distance from center to vertex is 1.
    verts_base = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def get_hex_verts(cx, cy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return verts_base @ R.T + np.array([cx, cy])

    def segments_intersect(p1, p2, p3, p4):
        # Check if segment p1-p2 intersects p3-p4
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def are_hexagons_disjoint(v1, v2):
        # v1, v2 are (6,2) arrays of vertices
        # Check if any edge of v1 intersects any edge of v2
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    def point_in_hex(p, v):
        # Check if point p is inside hexagon v (convex)
        # Cross product of edge vector and point vector must have same sign
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1)%6]
            cross1 = np.cross(p2 - p1, p - p1)
            if i == 0:
                sign = np.sign(cross1)
                if sign == 0: continue
            else:
                if np.sign(cross1) != sign:
                    return False
        return True

    def check_overlap(hexes):
        # hexes: list of (cx, cy, angle)
        verts_list = [get_hex_verts(h[0], h[1], h[2]) for h in hexes]
        for i in range(N):
            for j in range(i+1, N):
                if not are_hexagons_disjoint(verts_list[i], verts_list[j]):
                    return True
                # Also check containment (unlikely in packing but safe)
                if point_in_hex(hexes[i][:2], verts_list[j]) or point_in_hex(hexes[j][:2], verts_list[i]):
                    return True
        return False

    def calc_S(hexes, cx=0.0, cy=0.0, outer_angle=0.0):
        # Calculate minimum S for outer hexagon centered at (cx, cy) with angle outer_angle
        # Outer hexagon constraints for a point (x,y):
        # Let u = x - cx, v = y - cy
        # Rotate by -outer_angle to align with standard frame
        # Standard frame constraints for side S:
        # |u| <= S
        # |v| <= S * sqrt3/2
        # |sqrt3*u + v| <= S * sqrt3
        # |-sqrt3*u + v| <= S * sqrt3
        # Wait, the vertices of standard flat-top hexagon side S are:
        # (S, 0), (S/2, S*sqrt3/2), (-S/2, S*sqrt3/2), (-S, 0), (-S/2, -S*sqrt3/2), (S/2, -S*sqrt3/2)
        # The edges are:
        # x = S (vertical? No, flat top means flat edge on top? No, "flat-top" usually means horizontal top edge.
        # My verts_base above has vertices at 0, 60... which is "pointy-top" if 0 is x-axis?
        # Verts: (1,0), (0.5, 0.866)...
        # Edge from (1,0) to (0.5, 0.866). Midpoint (0.75, 0.433).
        # This is a hexagon with a vertex on the x-axis. This is "pointy-side" or "flat-bottom"?
        # It has vertical sides? No.
        # Let's stick to the definition:
        # A regular hexagon can be defined by its apothem (distance center to side) $a = S \frac{\sqrt{3}}{2}$.
        # The support function $h(\theta) = a / \cos(\theta \pmod{\pi/3})$? No.
        # Easier: The hexagon is the intersection of 6 strips.
        # For a hexagon with vertices at angles $0, \pi/3, \dots$, the normal vectors to the edges are at angles $\pi/6, \pi/2, \dots$.
        # Distance from center to each edge is $S \frac{\sqrt{3}}{2}$.
        # So for any point $P$, let $d_k$ be the projection of $P-C$ onto the normal vector of edge $k$.
        # We need $d_k \le S \frac{\sqrt{3}}{2}$ for all 6 edges.
        # Thus $S \ge \frac{2}{\sqrt{3}} \max_k | \text{proj}_{n_k}(P-C) |$.
        
        # Normals for "vertex at 0" hexagon (which is what verts_base is):
        # Edges are between (1,0)-(0.5, 0.866) etc.
        # Normal to edge 0-1: Vector (0.5, 0.866) - (1,0) = (-0.5, 0.866). Normal is (0.866, 0.5). Angle 30 deg.
        # So normals are at 30, 90, 150, 210, 270, 330 degrees.
        # i.e., $\pi/6 + k \pi/3$.
        
        max_proj = 0.0
        for h in hexes:
            verts = get_hex_verts(h[0], h[1], h[2])
            for v in verts:
                # Rotate relative to outer center and angle
                dx = v[0] - cx
                dy = v[1] - cy
                # Rotate by -outer_angle
                oa = np.deg2rad(outer_angle)
                cos_oa = np.cos(oa)
                sin_oa = np.sin(oa)
                u = dx * cos_oa + dy * sin_oa
                v_rot = -dx * sin_oa + dy * cos_oa
                
                # Project onto 6 normals
                # Normals angles: 30, 90, 150, 210, 270, 330
                for k in range(6):
                    angle = np.pi/6 + k * np.pi/3
                    nx = np.cos(angle)
                    ny = np.sin(angle)
                    proj = abs(u * nx + v_rot * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        S = max_proj * 2.0 / sqrt3
        return S

    # Initial positions: Hexagonal grid
    def hex_to_cart(q, r):
        return (sqrt3*(q + r/2), 1.5*r)
    
    positions_hex = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
                      (2,0), (1,-2), (0,-2), (-1,-1)]
    
    hexes = []
    for q, r in positions_hex:
        x, y = hex_to_cart(q, r)
        hexes.append([x, y, 0.0])
        
    best_S = calc_S(hexes)
    best_hexes = [h.copy() for h in hexes]
    
    # Simple Local Search
    current_S = best_S
    T = 5.0 # Temperature
    step_size = 1.0
    
    while time.time() < deadline - 5.0:
        # Pick a random hexagon and perturb it
        i = np.random.randint(N)
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        dtheta = np.random.normal(0, 0.1) # Radians
        
        old_x, old_y, old_t = hexes[i]
        hexes[i][0] += dx
        hexes[i][1] += dy
        hexes[i][2] += np.rad2deg(dtheta)
        
        # Check overlap
        if check_overlap(hexes):
            # Revert
            hexes[i] = [old_x, old_y, old_t]
        else:
            new_S = calc_S(hexes)
            if new_S < current_S or np.random.rand() < np.exp(-(new_S - current_S)/T):
                current_S = new_S
                if current_S < best_S:
                    best_S = current_S
                    best_hexes = [h.copy() for h in hexes]
            else:
                # Revert
                hexes[i] = [old_x, old_y, old_t]
                
        # Cool down
        T *= 0.99
        if T < 0.1:
            T = 0.1
            step_size *= 0.9

    # Return best found
    # The task asks for inner rows (x, y, angle_degrees)
    inner = []
    for h in best_hexes:
        inner.append([h[0], h[1], h[2]])
        
    # Center and Angle of outer hexagon
    # We assumed (0,0) and 0.0 in calc_S.
    # We can try to optimize the outer center/angle for the best inner configuration to squeeze S down further.
    # For now, return (0,0) and 0.0.
    
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
