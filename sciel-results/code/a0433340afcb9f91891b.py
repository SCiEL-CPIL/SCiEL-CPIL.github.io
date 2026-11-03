"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    sqrt3 = np.sqrt(3)
    
    # Triangular lattice basis vectors for unit hexagons (flat-top)
    # Distance between centers of adjacent hexagons is sqrt(3)
    v1 = np.array([1.5, 0.0])
    v2 = np.array([-0.5, sqrt3])
    
    # Generate a set of candidate axial coordinates for 12 hexagons
    # We want compact clusters. 
    # Instead of brute forcing all subsets of a grid (too slow), 
    # we define a few specific compact topologies manually or generate 
    # "hexagonal numbers" based clusters.
    
    # A very compact shape for 12 is the "flower" of 7 plus a ring? No, 7 is too big.
    # 12 hexagons can form a 3x4 rectangle on the lattice, or other shapes.
    # Let's generate all subsets of a small grid that are connected? 
    # That might be too many. 
    # Alternative: Use a few known good topologies.
    
    # Topology 1: 2-3-4-3 rows (as in seed)
    # Axial coords (q, r)
    # Row r=0: q = -1, 0, 1, 2
    # Row r=1: q = -1, 0, 1
    # Row r=-1: q = -1, 0, 1
    # Row r=2: q = 0, 1
    topologies = []
    
    # T1: 4-3-3-2
    t1 = [(-1,0), (0,0), (1,0), (2,0),
          (-1,1), (0,1), (1,1),
          (-1,-1), (0,-1), (1,-1),
          (0,2), (1,2)]
    topologies.append(t1)
    
    # T2: 3-4-4-1? No.
    # Let's try a 3x4 rectangle.
    # q in [-1, 1], r in [-1, 1] is 9. Need 3 more.
    # Let's just generate all subsets of a 4x4 grid that have size 12?
    # Grid points: q in [-2, 1], r in [-2, 1]
    # Total points = 16. Subsets of 12 is C(16,12) = 1820.
    # For each subset, we check if it's connected (optional but good for compactness).
    # Then we optimize the outer hexagon rotation.
    # This is a discrete search over ~1820 shapes.
    # For each shape, we sweep outer angle 0..60 deg in steps of 0.1 deg (600 steps).
    # 1820 * 600 = 1,092,000 evaluations.
    # Each evaluation: compute bounding box of 12 hexagons.
    # This should be fast in Python if vectorized or efficient loops.
    
    # Let's refine the grid to ensure we capture good shapes.
    # A 4x4 grid of axial coordinates:
    grid_points = [(q, r) for q in range(-2, 2) for r in range(-2, 2)]
    
    # Precompute pixel coordinates for grid points
    def axial_to_pixel(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y
    
    pixel_coords = [axial_to_pixel(q, r) for q, r in grid_points]
    
    # We need to choose 12 points out of 16.
    # To avoid duplicates (rotations/reflections of the same shape), we could canonicalize,
    # but for 1820 shapes, it's fast enough to just evaluate them all.
    
    best_side = np.inf
    best_config = None
    
    # Sweep outer hexagon angles
    # The outer hexagon has 6-fold symmetry, so we only need to sweep 0 to 60 degrees.
    # However, the inner cluster is fixed relative to the lattice.
    # Actually, the inner cluster can be rotated too? 
    # The task says "each inner hexagon may translate and rotate".
    # In this discrete strategy, we assume all inner hexagons are axis-aligned (angle=0).
    # This is a constraint. Is it safe?
    # If the optimal solution requires rotated inner hexagons, this will fail.
    # However, for hexagonal packings, axis-aligned is often optimal or near-optimal.
    # Also, we can rotate the *entire cluster* by sweeping the outer angle?
    # No, rotating the outer hexagon relative to a fixed cluster is equivalent to rotating the cluster relative to a fixed outer hexagon.
    # So sweeping outer_angle covers all relative orientations.
    
    outer_angles = np.linspace(0, 60, 601)
    
    # To speed up, we can precompute the vertices of the unit hexagon
    unit_hex_verts = np.array([
        [np.cos(np.radians(a)), np.sin(np.radians(a))] for a in [0, 60, 120, 180, 240, 300]
    ])
    
    # Function to compute min outer side length for a given set of centers and outer angle
    def calc_side(centers, outer_angle_deg):
        oa = np.radians(outer_angle_deg)
        # Face normals of outer hexagon
        # Outer hexagon side s, inradius r = s * sqrt(3)/2
        # Vertices of outer hexagon at distance R = s from center? No, side length is s.
        # Circumradius R = s. Inradius r = s * sqrt(3)/2.
        # The condition for a point p to be inside is:
        # max_k | dot(p - c, n_k) | <= r
        # where n_k are unit normals to the 6 faces.
        
        # Normals are at angles: oa + 30 + k*60 ?
        # Let's check.
        # If outer hex is flat-top (angle 0), vertices are at 0, 60, ...
        # Faces are centered at 30, 90, ...
        # So normals are at 30 + k*60.
        
        max_proj = 0
        for cx, cy in centers:
            # Relative to center (0,0)
            # We can translate the whole cluster so its centroid is at (0,0) to simplify?
            # The outer hexagon center is also a variable.
            # To minimize side length, the outer hexagon center should be chosen optimally.
            # For a fixed set of points, the minimal enclosing hexagon center is not trivially the centroid.
            # However, we can fix the outer center to the centroid of the cluster for this discrete search.
            # This is an approximation.
            
            # Project all vertices of each inner hexagon onto the 6 normals.
            # Vertices of inner hex i at center (cx, cy):
            # v = (cx, cy) + unit_hex_verts
            
            for vx, vy in unit_hex_verts:
                px = cx + vx
                py = cy + vy
                
                for k in range(6):
                    theta = oa + np.radians(30 + k * 60)
                    nx = np.cos(theta)
                    ny = np.sin(theta)
                    d = px * nx + py * ny
                    if d > max_proj:
                        max_proj = d
        
        # inradius = max_proj
        # side = 2 * inradius / sqrt(3)
        return 2.0 * max_proj / sqrt3

    # Iterate over all subsets of 12 points from the 16 grid points
    # Indices: 0..15
    # We use combinations
    from itertools import combinations
    
    # To save time, we can filter out subsets that are not "compact"
    # e.g., bounding box too large.
    
    count = 0
    for combo in combinations(range(16), 12):
        if time.time() > deadline - 5:
            break
            
        centers = [pixel_coords[i] for i in combo]
        
        # Quick check: bounding box size
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        
        # A 3x4 rectangle has width ~ 4.5, height ~ 4.3
        # If it's too spread out, skip
        if w > 6.0 or h > 6.0:
            continue
            
        # Center the cluster at origin for calculation
        cx_mean = np.mean(xs)
        cy_mean = np.mean(ys)
        centered_centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
        
        # Sweep outer angle
        # Optimization: The function is periodic and smooth. 
        # We can do a coarse sweep first.
        for oa in outer_angles:
            side = calc_side(centered_centers, oa)
            if side < best_side:
                best_side = side
                # Store best config
                # We need to reconstruct the full config
                best_config = {
                    'centers': centers,
                    'cx_mean': cx_mean,
                    'cy_mean': cy_mean,
                    'oa': oa
                }
                
        count += 1
        if count % 100 == 0:
            # print(f"Processed {count} shapes, best: {best_side}")
            pass

    if best_config is None:
        # Fallback to seed config
        return _fallback()

    # Refine the best discrete solution with a local continuous optimization
    # We take the best discrete configuration and allow small translations/rotations?
    # No, the discrete strategy assumes axis-aligned.
    # We can refine the outer angle and outer center.
    
    # Let's refine the outer angle around the best discrete angle
    best_oa = best_config['oa']
    centers = best_config['centers']
    
    # Fine sweep of outer angle
    fine_angles = np.linspace(best_oa - 2, best_oa + 2, 201)
    best_fine_side = best_side
    best_fine_oa = best_oa
    
    cx_mean = best_config['cx_mean']
    cy_mean = best_config['cy_mean']
    centered_centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
    
    for oa in fine_angles:
        side = calc_side(centered_centers, oa)
        if side < best_fine_side:
            best_fine_side = side
            best_fine_oa = oa
    
    # The outer center in the global frame is (cx_mean, cy_mean)
    # But we can also translate the outer hexagon independently.
    # In calc_side, we assumed outer center is at (0,0) relative to centered cluster.
    # So outer center in global frame is (cx_mean, cy_mean).
    
    outer_cx = cx_mean
    outer_cy = cy_mean
    outer_angle = best_fine_oa
    side = best_fine_side
    
    # Construct inner list
    inner = []
    for x, y in centers:
        inner.append([float(x), float(y), 0.0])
        
    return inner, (float(outer_cx), float(outer_cy)), float(side), float(outer_angle)

def _fallback():
    # Return the seed program's initial guess if search fails
    sqrt3 = np.sqrt(3)
    axial_coords = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, -1), (0, -1), (1, -1),
        (0, 2), (1, 2)
    ]
    
    def axial_to_pixel(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y
    
    init_x = []
    init_y = []
    for q, r in axial_coords:
        x, y = axial_to_pixel(q, r)
        init_x.append(x)
        init_y.append(y)
        
    cx0 = np.mean(init_x)
    cy0 = np.mean(init_y)
    init_x = [x - cx0 for x in init_x]
    init_y = [y - cy0 for y in init_y]
    
    inner = [[float(x), float(y), 0.0] for x, y in zip(init_x, init_y)]
    
    # Estimate side
    # Just return a reasonable guess
    side = 5.0
    return inner, (0.0, 0.0), side, 0.0

# EVOLVE-BLOCK-END
