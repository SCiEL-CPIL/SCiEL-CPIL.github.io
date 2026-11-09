"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    sqrt3 = math.sqrt(3.0)
    
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon in CCW order."""
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def point_in_convex(px, py, poly):
        """Check if point is inside convex polygon (CCW order)."""
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross < -1e-10:
                return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        c1 = cross(p3, p4, p1)
        c2 = cross(p3, p4, p2)
        c3 = cross(p1, p2, p3)
        c4 = cross(p1, p2, p4)

        if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
           ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
            return True
        return False

    def polygons_intersect(poly1, poly2):
        """Check if two convex polygons intersect."""
        for p in poly1:
            if point_in_convex(p[0], p[1], poly2):
                return True
        for p in poly2:
            if point_in_convex(p[0], p[1], poly1):
                return True
        for i in range(len(poly1)):
            p1 = poly1[i]
            p2 = poly1[(i + 1) % len(poly1)]
            for j in range(len(poly2)):
                p3 = poly2[j]
                p4 = poly2[(j + 1) % len(poly2)]
                if segments_intersect(p1, p2, p3, p4):
                    return True
        return False

    def check_overlap(centers, angles):
        """Check if any two inner hexagons overlap."""
        for i in range(N):
            vi = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for j in range(i + 1, N):
                vj = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                if polygons_intersect(vi, vj):
                    return False
        return True

    def distance_point_to_polygon_boundary(px, py, poly):
        """Calculate distance from point to the closest edge of the polygon."""
        min_dist = float('inf')
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            # Vector of the line
            dx = x2 - x1
            dy = y2 - y1
            # Length squared
            length_sq = dx*dx + dy*dy
            if length_sq == 0: continue
            # Project point onto line
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = math.hypot(px - proj_x, py - proj_y)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def overlap_penalty(centers, angles):
        """Calculate a penalty for overlaps."""
        penalty = 0.0
        for i in range(N):
            vi = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for j in range(i + 1, N):
                vj = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                # Check vertices of i in j
                for p in vi:
                    if point_in_convex(p[0], p[1], vj):
                        penalty += distance_point_to_polygon_boundary(p[0], p[1], vj)
                # Check vertices of j in i
                for p in vj:
                    if point_in_convex(p[0], p[1], vi):
                        penalty += distance_point_to_polygon_boundary(p[0], p[1], vi)
        return penalty

    def enclosing_side_length(centers, angles, outer_angle=0.0):
        """Compute minimal side length of outer hexagon containing all inner hexagons."""
        normals = []
        for k in range(6):
            angle = math.radians(outer_angle + 30 + k * 60)
            normals.append((math.cos(angle), math.sin(angle)))

        max_extent = 0.0
        for i in range(N):
            cx, cy = centers[i]
            inner_angle_rad = math.radians(angles[i])

            for nk in range(6):
                nx, ny = normals[nk]
                n_angle = math.atan2(ny, nx)

                max_dot = -2.0
                for j in range(6):
                    v_angle = inner_angle_rad + j * math.pi / 3.0
                    dot_val = math.cos(v_angle - n_angle)
                    if dot_val > max_dot:
                        max_dot = dot_val

                extent = cx * nx + cy * ny + max_dot
                if extent > max_extent:
                    max_extent = extent

        return 2.0 * max_extent / sqrt3

    def objective(params):
        if time.time() > deadline:
            return 1e10
        centers_opt = []
        angles_opt = []
        for i in range(N):
            centers_opt.append((params[i * 3], params[i * 3 + 1]))
            angles_opt.append(params[i * 3 + 2])
        outer_angle = params[36]

        L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
        pen = overlap_penalty(centers_opt, angles_opt)
        
        # If there is any overlap, return a large value
        if pen > 1e-6:
            return 1000.0 + L + 10.0 * pen
            
        return L

    # Initial configuration: 13-hex grid (remove 1 to make 12)
    # Grid of side 3 has 19 hexes.
    # We use Center + Ring 1 (6) + Ring 2 Edges (6). Total 13.
    # We remove one Ring 2 Edge to make 12.
    
    def grid_to_cart(q, r):
        x = sqrt3 * q + sqrt3 / 2 * r
        y = 1.5 * r
        return (x, y)

    # Axial coordinates for the 13-hex configuration
    # (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 Edges: (2,-1), (1,2), (-1,2), (-2,1), (-1,-1), (0,-2)
    # Wait, let's verify Ring 2 Edges.
    # The 19 grid Ring 2 has 12 hexes.
    # Corners: (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1) -> Dist 2*sqrt(3)
    # Edges: (2,-1), (1,2), (-1,2), (-2,1), (-1,0)?? No.
    # Let's list all q,r with max(|q|,|r|,|q+r|) == 2.
    # (2,0) Corner
    # (2,-1) Edge
    # (1,1) Corner
    # (1,2) Edge
    # (0,2) Corner
    # (-1,2) Edge
    # (-1,1) Corner
    # (-2,1) Edge
    # (-2,0) Corner
    # (-1,-1) Corner
    # (0,-1) Edge? No, (0,-1) is Ring 1.
    # (1,-1) is Ring 1.
    # Let's check (0,-2). |q|=0, |r|=2, |q+r|=2. Yes.
    # (-1,0). |q|=1, |r|=0, |q+r|=1. Ring 1.
    # (1,0). Ring 1.
    
    # So Ring 2 Edges are:
    # (2,-1), (1,2), (-1,2), (-2,1), (-1,-1)?? No (-1,-1) is Corner.
    # Let's re-evaluate Corners vs Edges.
    # Corners are (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1).
    # Edges are (2,-1), (1,2), (-1,2), (-2,1), (-1,0)?? No.
    # (0,-2) is an edge.
    # (1,-1) is Ring 1.
    # (-1,0) is Ring 1.
    
    # Let's just use the 19 grid and pick the 12 with smallest max distance?
    # No, just use the 13-hex config and drop one.
    
    init_grid_coords_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 2), (-1, 2), (-2, 1), (0, -2), (-1, -1) # Wait, (-1,-1) is a corner.
    ]
    # Let's replace (-1,-1) with another edge?
    # The 6 edges are: (2,-1), (1,2), (-1,2), (-2,1), (0,-2), (1,-1)?? No (1,-1) is R1.
    # Let's check (1,-1) dist. x=sqrt3(0.5), y=-1.5. Dist sqrt(0.75+2.25)=sqrt(3). R1.
    # The 6 edges of R2 are:
    # (2,-1), (1,2), (-1,2), (-2,1), (0,-2), and... (1,-1) is R1.
    # What is the 6th edge?
    # Symmetry: (2,-1) -> rotate 60 deg -> (1,2) -> ( -1, 2) -> (-2, 1) -> (-1, -1)??
    # Let's check (-1,-1). q=-1, r=-1. |q|=1, |r|=1, |q+r|=2. It is on the boundary of the hexagon of side 2?
    # No, the hexagon of side 2 contains centers with max(|q|,|r|,|q+r|) <= 1.
    # So R2 is max == 2.
    # (-1,-1) is a Corner.
    # The 6 Edges are:
    # (2,-1), (1,2), (-1,2), (-2,1), (0,-2), and (1,-1) is NOT it.
    # It must be (1,-1) rotated?
    # Let's list all 12 R2:
    # (2,0) C
    # (2,-1) E
    # (1,1) C
    # (1,2) E
    # (0,2) C
    # (-1,2) E
    # (-1,1) C
    # (-2,1) E
    # (-2,0) C
    # (-1,-1) C
    # (0,-1) R1
    # (1,-1) R1
    # Wait, I missed some.
    # q=2: r=-1, 0. (2,-1) E, (2,0) C.
    # q=1: r=-1, 0, 1, 2. (1,-1) R1, (1,0) R1, (1,1) C, (1,2) E.
    # q=0: r=-2, -1, 0, 1, 2. (0,-2) E, (0,-1) R1, (0,0) C, (0,1) R1, (0,2) C.
    # q=-1: r=-2, -1, 0, 1, 2. (-1,-2) E, (-1,-1) C, (-1,0) R1, (-1,1) C, (-1,2) E.
    # q=-2: r=-1, 0. (-2,-1) E, (-2,0) C.
    
    # So the 6 Edges are:
    # (2,-1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2).
    
    init_grid_coords_12 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 2), (-1, 2), (-2, 1)
    ]
    # This is 12 hexes.
    # Max dist:
    # (2,-1) -> x=1.5sqrt3, y=-1.5. Dist 3.
    # (1,2) -> x=2.5sqrt3/2? x=sqrt3(1+1)=2sqrt3, y=3. Dist 5?
    # Wait.
    # (1,2): x = sqrt3(1 + 1) = 2sqrt3. y = 3.
    # Dist = sqrt(12 + 9) = sqrt(21) approx 4.58.
    # This is bad.
    
    # Let's use the 13-hex config with max dist 3.
    # The 6 edges with dist 3 are:
    # (2,-1) -> 3
    # (1,2) -> sqrt(12+9)=sqrt(21) != 3.
    # My previous calculation for (1,1) was dist 3?
    # (1,1) is a Corner.
    # Let's re-calculate dist for (1,1).
    # x = sqrt3(1.5) = 1.5sqrt3. y = 1.5.
    # Dist = sqrt(6.75 + 2.25) = 3.
    # So (1,1) is at dist 3.
    # (2,0) is at dist 2sqrt3 approx 3.46.
    
    # So the 6 "mid-edge" hexes of the side-3 grid are actually the ones at dist 3?
    # No, (1,1) is a corner of the R2 ring?
    # In a hex grid, the "corners" of the bounding hexagon are (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1).
    # Their distances:
    # (2,0): 3.46
    # (1,1): 3.0
    # (0,2): 3.46
    # (-1,1): 3.0
    # (-2,0): 3.46
    # (-1,-1): 3.0
    
    # So 3 of the corners are at 3.0 and 3 are at 3.46.
    # If we pick the 3 at 3.0 and 3 of the edges?
    
    # Let's just use the optimizer.
    
    centers = [grid_to_cart(q, r) for q, r in init_grid_coords_12]
    angles = [0.0] * N
    
    # If overlap, fix it
    if not check_overlap(centers, angles):
        # Fallback to random
        rng = random.Random(42)
        centers = [(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(N)]
        angles = [rng.uniform(-180, 180) for _ in range(N)]

    best_L = enclosing_side_length(centers, angles, 0.0)
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    best_outer_angle = 0.0

    init_params = []
    for i in range(N):
        init_params.append(centers[i][0])
        init_params.append(centers[i][1])
        init_params.append(angles[i])
    init_params.append(0.0)

    # Local Search
    res = minimize(objective, init_params, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    if res.fun < best_L and check_overlap([ (res.x[i*3], res.x[i*3+1]) for i in range(N) ], [res.x[i*3+2] for i in range(N)]):
        best_L = res.fun
        best_centers = [ (res.x[i*3], res.x[i*3+1]) for i in range(N) ]
        best_angles = [res.x[i*3+2] for i in range(N)]
        best_outer_angle = res.x[36]

    # Multi-start Local Search
    rng = random.Random(123)
    for _ in range(5):
        if time.time() > deadline - 20.0:
            break
        # Perturb
        perturbed = init_params[:]
        for i in range(37):
            perturbed[i] += rng.uniform(-1.0, 1.0)
        
        res = minimize(objective, perturbed, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        c_opt = [ (res.x[i*3], res.x[i*3+1]) for i in range(N) ]
        a_opt = [res.x[i*3+2] for i in range(N)]
        
        if check_overlap(c_opt, a_opt):
            L = enclosing_side_length(c_opt, a_opt, res.x[36])
            if L < best_L:
                best_L = L
                best_centers = c_opt
                best_angles = a_opt
                best_outer_angle = res.x[36]

    # Final validation
    if not check_overlap(best_centers, best_angles):
        # Revert to initial
        best_centers = [grid_to_cart(q, r) for q, r in init_grid_coords_12]
        best_angles = [0.0] * N
        best_L = enclosing_side_length(best_centers, best_angles, 0.0)
        best_outer_angle = 0.0

    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])

    center = (0.0, 0.0)
    side = best_L
    angle = best_outer_angle

    return inner, center, side, angle
# EVOLVE-BLOCK-END
