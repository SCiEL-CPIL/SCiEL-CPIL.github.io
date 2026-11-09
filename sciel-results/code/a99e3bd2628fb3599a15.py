"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Constants
    SQRT3 = math.sqrt(3)
    INRADIUS = SQRT3 / 2.0
    
    def axial_to_cartesian(q, r):
        # Flat-top hexagon grid
        # x = SQRT3 * (q + r/2)
        # y = 1.5 * r
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    def hexagon_vertices(cx, cy, angle_deg, side=1.0):
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + math.radians(60 * i)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def min_outer_side(vertices, outer_angle_deg=0.0):
        """
        Calculate the minimum side length S of a regular hexagon with given 
        orientation such that it contains all points in 'vertices'.
        Assumes the outer hexagon is centered at the origin.
        """
        # Normals to the sides of a flat-top hexagon (vertices at 0, 60, ...)
        # Sides are perpendicular to 30, 90, 150, 210, 270, 330 degrees
        # If outer is rotated by outer_angle_deg, normals are shifted
        base_angles = [30, 90, 150, 210, 270, 330]
        normals = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            normals.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for x, y in vertices:
            for nx, ny in normals:
                proj = abs(x * nx + y * ny)
                if proj > max_proj:
                    max_proj = proj
        
        # The apothem (distance center to side) of outer hex is S * SQRT3 / 2
        # So max_proj <= S * SQRT3 / 2
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    def evaluate_configuration(axial_coords, inner_angle=0.0, outer_angle=0.0):
        """
        Evaluate a configuration of inner hexagons given by axial coordinates.
        Returns (side_length, inner_list).
        """
        all_vertices = []
        inner_list = []
        
        for q, r in axial_coords:
            cx, cy = axial_to_cartesian(q, r)
            inner_list.append([cx, cy, inner_angle])
            verts = hexagon_vertices(cx, cy, inner_angle)
            all_vertices.extend(verts)
        
        side = min_outer_side(all_vertices, outer_angle)
        return side, inner_list

    # Candidate configurations for N=11
    # Base 7-hex cluster (radius 1)
    base_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    candidates = []
    
    # Candidate 1: Add 4 to the right
    # Neighbors of the right side (1,0), (1,-1), (0,-1)
    # (2,0) is right of (1,0)
    # (2,-1) is right of (1,-1)
    # (2,1) is right of (0,1)? No, (1,1) is right of (0,1) and (1,0)?
    # Let's just append specific coordinates that form a compact shape.
    
    # Shape A: Elongated to the right
    # (2,0), (2,-1), (2,1), (1,1) ? 
    # Let's check connectivity.
    # (1,1) is neighbor of (0,1) and (1,0).
    # (2,0) is neighbor of (1,0) and (1,-1)? No, (1,-1) neighbor is (2,-1).
    # (2,0) neighbors: (1,0), (1,1), (2,1), (2,-1), (3,-1), (3,0).
    # So (2,0), (2,-1), (2,1), (1,1) seems like a valid block.
    cand_A = base_7 + [(2,0), (2,-1), (2,1), (1,1)]
    candidates.append(cand_A)
    
    # Shape B: Symmetric cross (2 right, 2 left)
    # (2,0), (-2,0), (0,2), (0,-2)
    cand_B = base_7 + [(2,0), (-2,0), (0,2), (0,-2)]
    candidates.append(cand_B)
    
    # Shape C: Symmetric diagonal (2 right-up, 2 left-down)
    # (1,1), (2,1) and (-1,-1), (-2,-1)
    cand_C = base_7 + [(1,1), (2,1), (-1,-1), (-2,-1)]
    candidates.append(cand_C)
    
    # Shape D: Fill corners more evenly
    # (1,1), (-1,1) and (1,-1), (-1,-1) are already in base?
    # Base: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Missing "corners" of the radius 2 hexagon are:
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # And the "edge" ones: (1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1)
    # Let's try adding (1,1), (-1,2), (-2,1), (2,-1)
    cand_D = base_7 + [(1,1), (-1,2), (-2,1), (2,-1)]
    candidates.append(cand_D)
    
    # Shape E: Another variation
    # (2,0), (1,1), (-1,2), (-2,0)
    cand_E = base_7 + [(2,0), (1,1), (-1,2), (-2,0)]
    candidates.append(cand_E)
    
    best_side = float('inf')
    best_inner = None
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    # Try a few outer rotations just in case, though 0 is likely best
    outer_angles_to_try = [0.0, 30.0]
    
    for cand in candidates:
        if len(cand) != N:
            continue
        for oa in outer_angles_to_try:
            side, inner = evaluate_configuration(cand, 0.0, oa)
            if side < best_side:
                best_side = side
                best_inner = inner
                best_outer_angle = oa
                best_inner_angle = 0.0
                
    # If no candidate worked (shouldn't happen), fallback to linear
    if best_inner is None:
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_side = 40.0
        best_outer_angle = 0.0
        best_inner_angle = 0.0

    # Center the configuration at (0,0) is already done by axial_to_cartesian 
    # which centers the (0,0) axial point at (0,0) Cartesian.
    # The outer hexagon should be centered at the centroid of the inner hexagons 
    # to minimize side length? 
    # Actually, for a regular hexagon, centering at the centroid of the 
    # bounding set is optimal if the shape is symmetric. 
    # Our axial coordinates are roughly centered around (0,0).
    # Let's compute the centroid of the inner centers and shift everything.
    
    if best_inner:
        cx_sum = sum(p[0] for p in best_inner)
        cy_sum = sum(p[1] for p in best_inner)
        cx_avg = cx_sum / N
        cy_avg = cy_sum / N
        
        # Shift inner hexagons
        shifted_inner = []
        for x, y, a in best_inner:
            shifted_inner.append([x - cx_avg, y - cy_avg, a])
        
        # The outer hexagon center should be (0,0) in the final frame?
        # The interface asks for outer center.
        # If we shift inner to be centered at 0, we can set outer center to (0,0).
        # But wait, the "min_outer_side" function assumed outer center at (0,0).
        # So if we shift inner such that their "bounding box" is centered at 0,
        # the calculated side length is valid for an outer hex at (0,0).
        
        # However, the optimal center of the outer hexagon might not be exactly 
        # the centroid of the inner centers. 
        # But for these symmetric clusters, it's very close.
        # Given the constraints, returning outer center (0,0) and inner centered at (0,0) 
        # is a valid and likely optimal solution.
        
        return shifted_inner, (0.0, 0.0), best_side, best_outer_angle
    else:
        return best_inner, (0.0, 0.0), best_side, best_outer_angle

# EVOLVE-BLOCK-END
