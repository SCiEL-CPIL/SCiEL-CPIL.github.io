"""Unit hexagons in regular hexagon - Rigid Cluster Search + Local Rotation Refinement."""
import time
import math
import numpy as np
import itertools
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Base vertices of unit hexagon (pointy-right orientation, rotation 0).
    # Vertices at angles 0, 60, 120, 180, 240, 300.
    base_vertices = np.array([
        [math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))] for k in range(6)
    ])

    # --- Geometry / validity helpers ---
    def hex_vertices(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        return [(cx + side * math.cos(a + k * math.pi / 3.0),
                 cy + side * math.sin(a + k * math.pi / 3.0)) for k in range(6)]

    def point_in_convex(p, poly):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-12:
                return False
        return True

    def polygon_area_signed(verts):
        area = 0.0
        for i in range(len(verts)):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % len(verts)]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def line_intersect(p1, p2, p3, p4):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-15:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def polygon_overlap_area(v1, v2):
        def clip_polygon(subject, clip):
            output = list(subject)
            for i in range(len(clip)):
                if not output:
                    break
                inp = output
                output = []
                S = inp[-1]
                for E in inp:
                    if point_in_convex(E, clip):
                        if not point_in_convex(S, clip):
                            output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                        output.append(E)
                    elif point_in_convex(S, clip):
                        output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                    S = E
            return output
        clipped = clip_polygon(v1, v2)
        if len(clipped) < 3:
            return 0.0
        return abs(polygon_area_signed(clipped))

    def is_valid(inner_params, outer_center, outer_side, outer_angle):
        cx, cy = outer_center
        outer_verts = hex_vertices(cx, cy, outer_side, outer_angle)
        # containment
        for (ix, iy, iang) in inner_params:
            if not all(point_in_convex(v, outer_verts) for v in hex_vertices(ix, iy, 1.0, iang)):
                return False
        # non-overlap
        verts_list = [hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner_params]
        for i in range(N):
            for j in range(i + 1, N):
                if polygon_overlap_area(verts_list[i], verts_list[j]) > 1e-8:
                    return False
        return True

    # --- Fast Rigid Cluster Side Calculation ---
    def compute_side_rigid(centers, theta_out_deg):
        theta_out = math.radians(theta_out_deg)
        normals = np.array([
            [math.cos(math.radians(30 + k * 60) + theta_out),
             math.sin(math.radians(30 + k * 60) + theta_out)] for k in range(6)
        ])
        max_proj = 0.0
        for c in centers:
            verts = base_vertices + c
            dots = verts @ normals.T
            max_proj = max(max_proj, float(np.max(np.abs(dots))))
        return 2.0 * max_proj / math.sqrt(3)

    # --- Candidate Cluster Generators ---
    def make_cluster(r1, r2, ring1_angles, ring2_angles):
        centers = [[0.0, 0.0]]
        for a in ring1_angles:
            rad = math.radians(a)
            centers.append([r1 * math.cos(rad), r1 * math.sin(rad)])
        for a in ring2_angles:
            rad = math.radians(a)
            centers.append([r2 * math.cos(rad), r2 * math.sin(rad)])
        return np.array(centers)

    # Generate diverse lattice-based clusters
    candidates = []
    
    # Topology 1: Standard Flower (1-6-4)
    # Ring 1: 6 neighbors at dist sqrt(3), angles 0, 60, ...
    # Ring 2: 4 neighbors at dist 3, angles 30, 90, 150, 210 (or similar gaps)
    ring1_0 = [0, 60, 120, 180, 240, 300]
    ring1_30 = [30, 90, 150, 210, 270, 330]
    
    # Standard flower has ring1 at 0,60.. and ring2 at 30,90.. (gaps)
    # Let's enumerate subsets of the "second ring" directions
    # In a hex lattice, the 12 neighbors at distance 2 (in lattice units, which is 3 in side units? No, dist 2 is 2*sqrt(3)? No.)
    # Lattice points:
    # Center (0,0)
    # Ring 1 (dist sqrt(3)): 6 points.
    # Ring 2 (dist 3): 6 points (at 30, 90, etc) + 6 points (at 0, 60, etc but further out? No, dist 3 is the next ring in the "star" sense? 
    # Actually, in a hex lattice of side 1:
    # Neighbors at dist sqrt(3) are the 6 adjacent hexagons.
    # The next nearest are at dist 3 (angle 30, 90...) and dist 2*sqrt(3) (angle 0, 60... but that's 2 steps away).
    # Let's stick to the known compact configurations.
    
    # Config A: 1 center, 6 at dist sqrt(3) (angles 0..300), 4 at dist 3 (angles 30, 90, 150, 210)
    for r1 in [math.sqrt(3), math.sqrt(3) + 1e-9]:
        for r2 in [3.0, 3.0 + 2e-9]:
            # Vary which 4 of the 6 gap directions are used
            gap_dirs = [30, 90, 150, 210, 270, 330]
            for combo in itertools.combinations(gap_dirs, 4):
                candidates.append(make_cluster(r1, r2, ring1_0, list(combo)))
                
            # Also try rotating the whole ring1 set
            for combo in itertools.combinations(gap_dirs, 4):
                 # Rotate ring1 by 30 deg? Then gaps are at 0, 60...
                 # If ring1 is at 30, 90..., gaps are at 0, 60...
                 gap_dirs_rot = [0, 60, 120, 180, 240, 300]
                 candidates.append(make_cluster(r1, r2, ring1_30, list(combo)))

    # Config B: Different ring distances?
    # Maybe 1 center, 5 at dist sqrt(3), 5 at dist 3? Unlikely to be better than 6+4.
    
    # --- Grid Search over Outer Rotation for Rigid Clusters ---
    best_score = float('inf')
    best_centers = None
    best_theta = 0.0
    best_inner_rots = np.zeros(N) # All 0 for rigid

    coarse = np.linspace(0.0, 60.0, 600) # Coarse grid

    for centers in candidates:
        if time.time() > deadline - 20:
            break
        local_best = float('inf')
        local_theta = 0.0
        for t in coarse:
            s = compute_side_rigid(centers, t)
            if s < local_best:
                local_best = s
                local_theta = t
        if local_best >= best_score:
            continue
        # Refine 1D
        lo = max(0.0, local_theta - 10.0)
        hi = min(60.0, local_theta + 10.0)
        fine = np.linspace(lo, hi, 1000)
        for t in fine:
            s = compute_side_rigid(centers, t)
            if s < local_best:
                local_best = s
                local_theta = t
        
        if local_best < best_score:
            best_score = local_best
            best_centers = centers
            best_theta = local_theta

    # --- Local Refinement with Individual Rotations ---
    # Use scipy.optimize to refine [theta_out, phi_0, ..., phi_10]
    # Objective: Minimize outer side length + penalty for overlap/containment
    
    def objective(params):
        theta_out = params[0]
        phis = params[1:]
        
        # Compute vertices for all inner hexagons
        # phi_i is rotation in radians
        verts_list = []
        for i in range(N):
            c, s = np.cos(phis[i]), np.sin(phis[i])
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + best_centers[i]
            verts_list.append(verts)
            
        # Compute required outer side length
        theta_out_rad = theta_out
        normals = np.array([
            [math.cos(math.radians(30 + k * 60) + theta_out_rad),
             math.sin(math.radians(30 + k * 60) + theta_out_rad)] for k in range(6)
        ])
        
        max_proj = 0.0
        for verts in verts_list:
            dots = verts @ normals.T
            max_proj = max(max_proj, float(np.max(np.abs(dots))))
        
        side = 2.0 * max_proj / math.sqrt(3)
        
        # Calculate Penalty for Overlap and Containment
        penalty = 0.0
        
        # Containment Check:
        # Project vertices onto normals. If any projection exceeds side * sqrt(3)/2, it's outside.
        # Actually, the condition for a point (x,y) to be inside hexagon centered at 0 with side L and rotation theta is:
        # max_k |dot(v, n_k)| <= L * sqrt(3) / 2
        # We already computed max_proj. The limit is side * sqrt(3) / 2.
        # So if max_proj > side * sqrt(3) / 2, it's outside.
        # But side is derived from max_proj. So containment is automatically satisfied by the definition of side?
        # Yes, the side length is defined as the minimum to contain the projections.
        # So we only need to penalize OVERLAP.
        
        # Overlap Check:
        # For each pair, compute overlap area or penetration depth.
        # Area is expensive. Penetration depth is cheaper but harder to compute exactly for polygons.
        # Let's use a simple check: distance between centers vs min distance.
        # Min distance between two unit hexagons is 0 (touching).
        # If we use area, it's smooth-ish? No, area is 0 until they touch, then grows.
        # We can use a "soft" overlap: sum of squared distances of vertices inside the other polygon?
        
        # Simpler: Use the polygon_overlap_area function but it's slow.
        # For 11 hexagons, 55 pairs.
        # Let's try to optimize using a simpler metric:
        # If we assume the rigid configuration is already non-overlapping, small rotations might cause overlap.
        # We can use a penalty based on the minimum distance between edges?
        
        # Alternative: Just check if the configuration is valid using the exact `is_valid` logic?
        # That's boolean. Not good for optimization.
        
        # Let's use a penalty: Sum of areas of intersection.
        # To speed this up, we can skip the calculation if the bounding boxes don't intersect.
        
        for i in range(N):
            for j in range(i + 1, N):
                # Quick AABB check
                v1 = verts_list[i]
                v2 = verts_list[j]
                min_x1, max_x1 = np.min(v1[:,0]), np.max(v1[:,0])
                min_y1, max_y1 = np.min(v1[:,1]), np.max(v1[:,1])
                min_x2, max_x2 = np.min(v2[:,0]), np.max(v2[:,0])
                min_y2, max_y2 = np.min(v2[:,1]), np.max(v2[:,1])
                
                if max_x1 < min_x2 - 1e-9 or max_x2 < min_x1 - 1e-9 or max_y1 < min_y2 - 1e-9 or max_y2 < min_y1 - 1e-9:
                    continue
                
                # Compute overlap area
                # Convert to lists of tuples for the helper function
                v1_list = [(float(x), float(y)) for x, y in v1]
                v2_list = [(float(x), float(y)) for x, y in v2]
                area = polygon_overlap_area(v1_list, v2_list)
                if area > 1e-8:
                    penalty += area * 1000.0 # Heavy penalty
        
        return side + penalty

    # Initial guess: best rigid solution
    x0 = np.array([best_theta] + list(np.zeros(N)))
    
    # Optimize
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_score:
                # Check validity of the optimized solution
                theta_out_opt = res.x[0]
                phis_opt = res.x[1:]
                
                inner_params = []
                for i in range(N):
                    inner_params.append([float(best_centers[i][0]), float(best_centers[i][1]), float(np.degrees(phis_opt[i]))])
                
                # The side length from the optimizer includes penalty, so we need to recompute the true side length
                # and verify validity.
                # Recompute true side length for these rotations
                theta_out_rad = theta_out_opt
                normals = np.array([
                    [math.cos(math.radians(30 + k * 60) + theta_out_rad),
                     math.sin(math.radians(30 + k * 60) + theta_out_rad)] for k in range(6)
                ])
                max_proj = 0.0
                for i in range(N):
                    c, s = np.cos(phis_opt[i]), np.sin(phis_opt[i])
                    rot = np.array([[c, -s], [s, c]])
                    verts = base_vertices @ rot.T + best_centers[i]
                    dots = verts @ normals.T
                    max_proj = max(max_proj, float(np.max(np.abs(dots))))
                true_side = 2.0 * max_proj / math.sqrt(3)
                
                if is_valid(inner_params, (0.0, 0.0), true_side, theta_out_opt):
                    best_score = true_side
                    best_theta = theta_out_opt
                    best_inner_rots = phis_opt
        except Exception:
            pass

    # Final Result Construction
    if best_centers is None:
        # Fallback to standard flower
        centers = make_cluster(math.sqrt(3) + 1e-9, 3.0 + 2e-9, [0, 60, 120, 180, 240, 300], [30, 90, 150, 210])
        best_centers = centers
        best_theta = 0.0
        best_score = compute_side_rigid(centers, 0.0)
        best_inner_rots = np.zeros(N)

    inner = []
    for i in range(N):
        x = float(best_centers[i][0])
        y = float(best_centers[i][1])
        angle = float(np.degrees(best_inner_rots[i]))
        inner.append([x, y, angle])

    outer_side = best_score
    outer_angle = float(best_theta)

    # Final validity check
    if not is_valid(inner, (0.0, 0.0), outer_side, outer_angle):
        # Fallback to rigid best
        inner = [[float(best_centers[i][0]), float(best_centers[i][1]), 0.0] for i in range(N)]
        outer_side = compute_side_rigid(best_centers, best_theta)
        if not is_valid(inner, (0.0, 0.0), outer_side, outer_angle):
             # Hard fallback
             centers = make_cluster(math.sqrt(3) + 1e-9, 3.0 + 2e-9, [0, 60, 120, 180, 240, 300], [30, 90, 150, 210])
             inner = [[float(centers[i][0]), float(centers[i][1]), 0.0] for i in range(N)]
             outer_side = compute_side_rigid(centers, 0.0)
             outer_angle = 0.0

    return inner, (0.0, 0.0), outer_side, outer_angle
# EVOLVE-BLOCK-END
