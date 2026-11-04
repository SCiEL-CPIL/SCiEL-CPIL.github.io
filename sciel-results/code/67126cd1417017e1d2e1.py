import numpy as np
import time
import itertools

N = 11
DEADLINE = 140.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_convex_polygon(pt, poly, tol=1e-9):
    """Check if point is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > tol:
            if sign < 0:
                return False
            sign = 1
        elif cross < -tol:
            if sign > 0:
                return False
            sign = -1
    return True

def _verify_packing(inner, outer_center, side, outer_angle):
    """
    Verify that all inner hexagons are inside the outer hexagon and don't overlap.
    """
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_angle, side)
    
    inner_verts_list = []
    for x, y, a in inner:
        verts = _hex_vertices(x, y, a, 1.0)
        inner_verts_list.append(verts)
        for v in verts:
            if not _point_in_convex_polygon(v, outer_verts, tol=1e-8):
                return False
    
    for i in range(len(inner_verts_list)):
        for j in range(i + 1, len(inner_verts_list)):
            v1 = inner_verts_list[i]
            v2 = inner_verts_list[j]
            for v in v1:
                if _point_in_convex_polygon(v, v2, tol=1e-8):
                    return False
            for v in v2:
                if _point_in_convex_polygon(v, v1, tol=1e-8):
                    return False
    return True

def _min_enclosing_hexagon_side(centers, inner_angle=30.0):
    """
    Calculate the minimal side length of a regular hexagon enclosing the cluster
    of hexagons defined by centers and inner_angle.
    The outer hexagon center is fixed at the centroid of the cluster (or 0,0 if symmetric).
    We optimize the rotation of the outer hexagon.
    """
    # Assume cluster is centered at (0,0) for simplicity in this helper
    # The actual centers should be shifted to be centered at (0,0) before calling this
    # Or we pass the centers as is and assume the outer hex is centered at the centroid
    
    # Calculate vertices of all inner hexagons
    all_verts = []
    for x, y in centers:
        verts = _hex_vertices(x, y, inner_angle, 1.0)
        all_verts.extend(verts)
    all_verts = np.array(all_verts)
    
    # The centroid of the cluster
    centroid = np.mean(all_verts, axis=0)
    
    # We want to find the minimal S such that there exists an angle theta
    # where all points are inside the hexagon of side S, angle theta, centered at centroid.
    
    # The condition for a point p to be inside a regular hexagon of side S, angle theta, centered at c:
    # The hexagon is defined by 6 inequalities:
    # n_i . (p - c) <= S * apothem
    # where n_i are the normal vectors to the edges.
    # Apothem = S * sqrt(3) / 2.
    # So n_i . (p - c) <= S * sqrt(3) / 2
    # S >= 2 / sqrt(3) * max_i (n_i . (p - c))
    
    # The normals for a hexagon with angle theta (vertex angle) are at angles theta + 30 + k*60.
    # Wait, if vertices are at theta, theta+60..., then edges are perpendicular to normals at theta+30...
    
    def side_length_for_angle(theta):
        # theta is the rotation of the outer hexagon (vertex angle)
        normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + theta)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Project all vertices onto normals
        # diffs = all_verts - centroid
        # projs = diffs @ normals.T
        # max_proj = max(projs, axis=0)
        # max_dist = np.max(max_proj)
        # S = 2 * max_dist / sqrt(3)
        
        diffs = all_verts - centroid
        projs = diffs @ normals.T
        max_projs = np.max(projs, axis=0)
        max_dist = np.max(max_projs)
        
        return 2 * max_dist / np.sqrt(3)
    
    # Optimize theta using ternary search
    # The function is periodic with period 60 degrees.
    # We search in [0, 60]
    
    lo = 0.0
    hi = 60.0
    for _ in range(50):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        f1 = side_length_for_angle(m1)
        f2 = side_length_for_angle(m2)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
    
    optimal_theta = (lo + hi) / 2
    optimal_side = side_length_for_angle(optimal_theta)
    
    return optimal_side, optimal_theta, centroid

def construct():
    start_time = time.time()
    
    # Candidate 1: The "Flower" packing from the current program
    # 1 center, 6 ring 1, 4 ring 2
    centers_flower = []
    centers_flower.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers_flower.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers_flower.append((x, y))
    
    inner_angle = 30.0
    
    side1, theta1, centroid1 = _min_enclosing_hexagon_side(centers_flower, inner_angle)
    
    # Candidate 2: Try a slightly different arrangement?
    # Maybe rotate the ring 2?
    # The current ring 2 is at 30, 90, 150, 270.
    # What if we put them at 0, 60, 120, 180?
    centers_flower2 = []
    centers_flower2.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers_flower2.append((x, y))
    for k in range(4):
        angle = np.deg2rad(k * 60) # 0, 60, 120, 180
        r = 3.0
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers_flower2.append((x, y))
        
    side2, theta2, centroid2 = _min_enclosing_hexagon_side(centers_flower2, inner_angle)
    
    # Choose the best
    if side1 < side2:
        best_side = side1
        best_theta = theta1
        best_centers = centers_flower
        best_centroid = centroid1
    else:
        best_side = side2
        best_theta = theta2
        best_centers = centers_flower2
        best_centroid = centroid2
        
    # Shift centers so that the centroid is at (0,0) for the final output
    # The outer hexagon will be centered at (0,0)
    # The inner centers need to be shifted by -centroid
    
    inner = []
    for x, y in best_centers:
        inner.append([x - best_centroid[0], y - best_centroid[1], inner_angle])
        
    outer_center = (0.0, 0.0)
    outer_angle = best_theta
    side = best_side + 1e-9
    
    # Verify
    if time.time() - start_time < DEADLINE - 5:
        if not _verify_packing(inner, outer_center, side, outer_angle):
            side += 1e-6
            if not _verify_packing(inner, outer_center, side, outer_angle):
                side += 1e-4
    
    return inner, outer_center, side, outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
