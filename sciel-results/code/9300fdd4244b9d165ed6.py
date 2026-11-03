import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
EPS = 1e-6

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Unit hexagon vertices (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 300
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def min_dist_polygons(p1, p2):
        """
        Calculate the minimum distance between two convex polygons.
        If they intersect, return 0.
        """
        # Check vertex in polygon for intersection
        def point_in_poly(pt, poly):
            x, y = pt
            inside = False
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                    inside = not inside
            return inside

        # Check edge intersections
        def segments_intersect(p1, p2, p3, p4):
            def ccw(a, b, c):
                return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

        # Check for intersection
        for pt in p1:
            if point_in_poly(pt, p2):
                return 0.0
        for pt in p2:
            if point_in_poly(pt, p1):
                return 0.0
                
        for i in range(6):
            p1a = p1[i]
            p1b = p1[(i+1)%6]
            for j in range(6):
                p2a = p2[j]
                p2b = p2[(j+1)%6]
                if segments_intersect(p1a, p1b, p2a, p2b):
                    return 0.0

        # No intersection, calculate min distance
        min_d = float('inf')
        # Distance between all pairs of edges
        for i in range(6):
            a1 = p1[i]
            a2 = p1[(i+1)%6]
            for j in range(6):
                b1 = p2[j]
                b2 = p2[(j+1)%6]
                
                # Distance from point to line segment
                def point_seg_dist(p, a, b):
                    ab = b - a
                    ap = p - a
                    ab_sq = np.dot(ab, ab)
                    if ab_sq < EPS:
                        return np.linalg.norm(p - a)
                    t = np.dot(ap, ab) / ab_sq
                    t = np.clip(t, 0, 1)
                    proj = a + t * ab
                    return np.linalg.norm(p - proj)

                d1 = point_seg_dist(a1, b1, b2)
                d2 = point_seg_dist(a2, b1, b2)
                d3 = point_seg_dist(b1, a1, a2)
                d4 = point_seg_dist(b2, a1, a2)
                
                d = min(d1, d2, d3, d4)
                if d < min_d:
                    min_d = d
        return min_d

    def objective(params):
        if time.time() > deadline:
            return 1e6

        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Calculate outer side
        outer_side = calc_min_outer_side(positions, angles, 0.0)
        
        # Calculate penalty
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_polygons(polys[i], polys[j])
                if d < EPS:
                    penalty += (EPS - d) ** 2
                # We want disjoint, so distance > 0 is good.
                # If d < 0 (overlap), penalty is huge.
                # But min_dist returns 0 on overlap.
                # So we need to detect overlap to add penalty.
                # Actually, min_dist returns 0 if intersecting.
                # So if d == 0, they are touching or overlapping.
                # We need to distinguish touching (valid) from overlapping (invalid).
                # For optimization, we want to push them apart.
                # If they are touching, d=0. If overlapping, d=0.
                # This makes the gradient zero at the boundary, which is bad for SLSQP.
                # Better: use a penalty that is active when d < 0? But we can't get d < 0 easily.
                # Alternative: Check if vertices are inside. If yes, penalty.
                # Let's stick to the simple penalty: if d < threshold, add penalty.
                # But touching is allowed. So we only penalize if they are "too close" or overlapping.
                # Since we can't distinguish touching from overlapping with min_dist=0,
                # we will add a small penalty for d < 0.1 to encourage separation,
                # and a huge penalty if they are actually overlapping (detected by point-in-poly).
                
                # Re-evaluate overlap
                overlap = False
                for pt in polys[i]:
                    # Quick check: is pt inside polys[j]?
                    # Using a simplified point-in-poly for speed in inner loop?
                    # No, min_dist_polygons already did the heavy lifting.
                    # If d == 0, they touch or overlap.
                    # If they overlap, the "distance" is 0.
                    # To make the objective smooth, we can use a penalty based on d.
                    # If d < 0.01, penalty = (0.01 - d)^2.
                    # This encourages them to be at least 0.01 apart, which is effectively touching.
                    if d < 0.01:
                        penalty += (0.01 - d) ** 2
        
        return outer_side + 100.0 * penalty

    # Initial guess: Honeycomb lattice
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
    
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    best_score = float('inf')
    best_x = x0.copy()
    
    # Run SLSQP
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            options={'maxiter': 500, 'ftol': 1e-6, 'disp': False}
        )
        
        if res.fun < best_score:
            # Verify validity
            new_positions = res.x[:2*N].reshape(N, 2)
            new_angles = res.x[2*N:]
            
            polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = min_dist_polygons(polys[i], polys[j])
                    # If d < EPS, they might be overlapping or touching.
                    # We need to ensure they are not overlapping.
                    # min_dist returns 0 for both.
                    # We need a strict overlap check.
                    # Let's reuse the logic from min_dist_polygons but just for overlap.
                    def check_overlap(p1, p2):
                        def point_in_poly(pt, poly):
                            x, y = pt
                            inside = False
                            n = len(poly)
                            for i in range(n):
                                x1, y1 = poly[i]
                                x2, y2 = poly[(i + 1) % n]
                                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                                    inside = not inside
                            return inside
                        for pt in p1:
                            if point_in_poly(pt, p2):
                                return True
                        for pt in p2:
                            if point_in_poly(pt, p1):
                                return True
                        def segments_intersect(p1, p2, p3, p4):
                            def ccw(a, b, c):
                                return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
                            return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
                        for i in range(6):
                            p1a = p1[i]
                            p1b = p1[(i+1)%6]
                            for j in range(6):
                                p2a = p2[j]
                                p2b = p2[(j+1)%6]
                                if segments_intersect(p1a, p1b, p2a, p2b):
                                    return True
                        return False

                    if check_overlap(polys[i], polys[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                score = calc_min_outer_side(new_positions, new_angles, 0.0)
                if score < best_score:
                    best_score = score
                    best_x = res.x
    except Exception:
        pass

    # Fallback to initial if optimization failed or was invalid
    if best_score == float('inf'):
        best_positions = init_positions
        best_angles = init_angles
        best_score = calc_min_outer_side(best_positions, best_angles, 0.0)
    else:
        best_positions = best_x[:2*N].reshape(N, 2)
        best_angles = best_x[2*N:]

    final_positions = best_positions
    final_angles = best_angles
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

