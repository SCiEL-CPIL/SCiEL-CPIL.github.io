"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg):
        """
        Computes the minimum side length of the outer hexagon centered at (outer_cx, outer_cy)
        with rotation outer_angle_deg to contain all inner hexagons.
        """
        outer_center = np.array([outer_cx, outer_cy])
        outer_angle_rad = math.radians(outer_angle_deg)
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = inner_centers[i]
            ang = inner_angles[i]
            verts = hex_vertices(cx, cy, ang)
            rel_verts = verts - outer_center
            for k in range(6):
                normal_angle = outer_angle_rad + k * math.pi / 3
                normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                proj = np.dot(rel_verts, normal)
                max_proj = max(max_proj, np.max(proj))
        
        return 2.0 * max_proj / math.sqrt(3)

    def objective(params):
        global_rot = params[0]
        outer_cx = params[1]
        outer_cy = params[2]
        outer_angle_deg = params[3]
        
        # Rotate centers
        rot_rad = math.radians(global_rot)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        
        rotated_centers = []
        for cx, cy in base_centers:
            rx = cx * cos_r - cy * sin_r
            ry = cx * sin_r + cy * cos_r
            rotated_centers.append((rx, ry))
            
        # Inner angles are fixed at 0 for this lattice approach
        # But we can also rotate the hexagons themselves? 
        # Usually aligning hexagons with the lattice is best. 
        # If we rotate the lattice, the hexagons (fixed at 0) are now misaligned with the "flow" of the lattice?
        # Actually, a honeycomb lattice of hexagons with angle 0 is invariant under rotation by 60 degrees.
        # So if we rotate the centers by 30 degrees, the hexagons (still 0) will fit into the "gaps" differently?
        # No, the hexagons are rigid. 
        # Let's keep inner angles at 0.
        
        side = compute_outer_side(rotated_centers, [0.0]*N, outer_cx, outer_cy, outer_angle_deg)
        return side

    # Generate lattice points
    d = math.sqrt(3) + 0.01 # Slight expansion to ensure disjointness
    
    # Basis vectors
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    # Helper to get lattice point
    def lattice_point(i, j):
        return i * u + j * v

    # Generate several candidate shapes of 12 points
    shapes = []
    
    # Shape 1: 1 center, 6 ring, 5 from ring 2
    # Ring 0: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in basis (i,j)
    # Note: The basis vectors u, v are 60 deg apart.
    # Neighbors of (0,0):
    # (1,0) -> u
    # (0,1) -> v
    # (-1,1) -> -u+v
    # (-1,0) -> -u
    # (0,-1) -> -v
    # (1,-1) -> u-v
    
    pts1 = [
        lattice_point(0,0),
        lattice_point(1,0), lattice_point(0,1), lattice_point(-1,1),
        lattice_point(-1,0), lattice_point(0,-1), lattice_point(1,-1),
        # Ring 2 candidates (distance 2 from center in lattice steps, but Euclidean distance varies)
        # Let's just pick 5 close ones from Ring 2
        lattice_point(2,0), lattice_point(1,1), lattice_point(0,2),
        lattice_point(-2,0), lattice_point(-1,-1)
    ]
    shapes.append(pts1)
    
    # Shape 2: 2 rows of 6?
    # Row 0: (0,0) to (5,0)
    # Row 1: (0,1) to (5,1)
    pts2 = []
    for i in range(6):
        pts2.append(lattice_point(i, 0))
        pts2.append(lattice_point(i, 1))
    shapes.append(pts2)
    
    # Shape 3: 3 rows of 4?
    pts3 = []
    for i in range(4):
        pts3.append(lattice_point(i, 0))
        pts3.append(lattice_point(i, 1))
        pts3.append(lattice_point(i, 2))
        # Need 12, so 3*4 = 12.
        # Wait, lattice_point(i, 2) is not directly above lattice_point(i, 0) in a rectangular grid sense, 
        # but in the lattice it's fine.
    shapes.append(pts3)
    
    # Shape 4: A more compact "blob"
    # Center (0,0)
    # Ring 1 (6)
    # Ring 2: Pick the 5 that form the most circular shape
    # Ring 2 points in basis:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Distances from origin:
    # (2,0): 2d
    # (1,1): |u+v| = d (since angle 60, |u+v|^2 = d^2+d^2+2d^2 cos60 = 3d^2 -> d*sqrt(3))
    # Actually, let's just compute Euclidean distances and pick closest 5.
    ring2_candidates = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i == 0 and j == 0: continue
            # Check if it's in Ring 1
            # Ring 1 basis coords: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
            if (i,j) in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                continue
            # Check if it's in Ring 2 (Manhattan-like distance 2? No, just check if it's a neighbor of Ring 1)
            # Simpler: just generate all points with max(|i|,|j|,|i-j|) <= 2?
            # Let's just use the specific list of Ring 2 neighbors of Ring 1.
            pass
            
    # Let's manually define a good Ring 2 set for Shape 4
    # We want to fill the "corners" of the hexagon formed by Ring 1.
    # The gaps in Ring 1 are at 30, 90, 150, 210, 270, 330 degrees?
    # No, Ring 1 forms a hexagon.
    # Let's just use the 5 closest points from a larger set.
    all_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            all_pts.append((i, j))
            
    # Distance from origin in Euclidean space
    def dist_from_origin(i, j):
        p = lattice_point(i, j)
        return np.linalg.norm(p)
        
    # Sort all lattice points by distance
    sorted_pts = sorted(all_pts, key=lambda p: dist_from_origin(p[0], p[1]))
    
    # Exclude origin
    sorted_pts = [p for p in sorted_pts if p != (0,0)]
    
    # Take first 11
    pts4 = [lattice_point(0,0)]
    for i in range(11):
        pts4.append(lattice_point(sorted_pts[i][0], sorted_pts[i][1]))
    shapes.append(pts4)
    
    best_side = float('inf')
    best_result = None
    
    for idx, shape_pts in enumerate(shapes):
        # Center the shape
        center = np.mean(shape_pts, axis=0)
        base_centers = [(p[0] - center[0], p[1] - center[1]) for p in shape_pts]
        
        # Initial guess for outer
        # Bounding box of the shape
        xs = [c[0] for c in base_centers]
        ys = [c[1] for c in base_centers]
        max_x = max(xs)
        min_x = min(xs)
        max_y = max(ys)
        min_y = min(ys)
        width = max_x - min_x
        height = max_y - min_y
        initial_side = max(width, height) * 1.5 # Rough estimate
        
        x0 = [0.0, 0.0, 0.0, 0.0] # global_rot, outer_cx, outer_cy, outer_angle
        
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'maxfev': 2000,
                    'xatol': 1e-5,
                    'fatol': 1e-5,
                    'adaptive': True
                }
            )
            
            if res.fun < best_side:
                best_side = res.fun
                best_result = (shape_pts, res.x)
                
        except Exception as e:
            pass
            
        if time.time() > deadline - 20:
            break
            
    if best_result is None:
        # Fallback to simple grid
        inner = [[i * (math.sqrt(3) + 0.01), 0, 0] for i in range(-5, 7)]
        # This is a line, might be wide. 
        # Let's use the first shape as fallback
        shape_pts = shapes[0]
        center = np.mean(shape_pts, axis=0)
        base_centers = [(p[0] - center[0], p[1] - center[1]) for p in shape_pts]
        inner = [[float(c[0]), float(c[1]), 0.0] for c in base_centers]
        # Compute side
        side = compute_outer_side(base_centers, [0.0]*N, 0.0, 0.0, 0.0)
        return inner, (0.0, 0.0), float(side * 1.01), 0.0
        
    shape_pts, params = best_result
    global_rot, outer_cx, outer_cy, outer_angle = params
    
    # Reconstruct final inner list with rotation applied
    rot_rad = math.radians(global_rot)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    
    center = np.mean(shape_pts, axis=0)
    base_centers = [(p[0] - center[0], p[1] - center[1]) for p in shape_pts]
    
    inner = []
    for cx, cy in base_centers:
        rx = cx * cos_r - cy * sin_r
        ry = cx * sin_r + cy * cos_r
        inner.append([float(rx), float(ry), 0.0])
        
    final_side = best_side * 1.01 + 1e-4
    
    return inner, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle)
# EVOLVE-BLOCK-END
