import numpy as np
import time
import math

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _support_hex(angle_deg, side, dx, dy):
    d = np.array([dx, dy])
    dn = np.linalg.norm(d)
    if dn < 1e-12:
        return 0.0
    d = d / dn
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    vs = side * np.array([np.cos(angles), np.sin(angles)]).T
    dots = vs @ d
    return np.max(dots)

def _min_enclosing_hex_side(centers, angles, outer_angle_deg):
    outer_angle = np.radians(outer_angle_deg)
    normals = np.array([
        [np.cos(outer_angle + np.radians(60 * i)),
         np.sin(outer_angle + np.radians(60 * i))]
        for i in range(6)
    ])
    
    max_ext = 0.0
    for k in range(len(centers)):
        cx, cy = centers[k]
        ang = angles[k]
        for ni in range(6):
            dx = normals[ni, 0]
            dy = normals[ni, 1]
            h = _support_hex(ang, 1.0, dx, dy)
            dot_center = cx * dx + cy * dy
            needed = dot_center + h
            if needed > max_ext:
                max_ext = needed
    
    S = 2.0 * max_ext / np.sqrt(3.0)
    return S

def _check_validity(centers, angles, S, outer_angle_deg, tol=1e-6):
    outer_verts = _hex_verts(0.0, 0.0, S, outer_angle_deg)
    inner_verts_list = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        iv = _hex_verts(cx, cy, 1.0, ang)
        inner_verts_list.append(iv)
        for pt in iv:
            inside = True
            for j in range(6):
                v1 = outer_verts[j]
                v2 = outer_verts[(j+1)%6]
                edge = v2 - v1
                rel = pt - v1
                cross = edge[0]*rel[1] - edge[1]*rel[0]
                if cross < -tol:
                    inside = False
                    break
            if not inside:
                return False
    
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0:
                continue
            
            def point_in_poly(pt, poly):
                n = len(poly)
                for idx in range(n):
                    v1 = poly[idx]
                    v2 = poly[(idx+1)%n]
                    edge = v2 - v1
                    rel = pt - v1
                    cross = edge[0]*rel[1] - edge[1]*rel[0]
                    if cross < -1e-9:
                        return False
                return True

            def segments_intersect(p1, p2, p3, p4):
                def cross(o, a, b):
                    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                
                d1 = cross(p3, p4, p1)
                d2 = cross(p3, p4, p2)
                d3 = cross(p1, p2, p3)
                d4 = cross(p1, p2, p4)
                
                if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                   ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                    return True
                return False

            overlap = False
            for pt in vi:
                if point_in_poly(pt, vj):
                    overlap = True
                    break
            if not overlap:
                for pt in vj:
                    if point_in_poly(pt, vi):
                        overlap = True
                        break
            if not overlap:
                for idx1 in range(6):
                    p1 = vi[idx1]
                    p2 = vi[(idx1+1)%6]
                    for idx2 in range(6):
                        p3 = vj[idx2]
                        p4 = vj[(idx2+1)%6]
                        if segments_intersect(p1, p2, p3, p4):
                            overlap = True
                            break
                    if overlap:
                        break
            
            if overlap:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    u = np.array([1.5, np.sqrt(3.0)/2.0])
    v = np.array([-1.5, np.sqrt(3.0)/2.0])
    
    positions = []
    for i in range(4):
        for j in range(3):
            pos = i * u + j * v
            positions.append(pos)
    
    pos_array = np.array(positions, dtype=float)
    center = np.mean(pos_array, axis=0)
    pos_array = pos_array - center
    
    angles = np.zeros(N)
    
    S_init = _min_enclosing_hex_side(pos_array, angles, 0.0)
    valid_init = _check_validity(pos_array, angles, S_init, 0.0)
    
    if valid_init:
        best_centers = pos_array.copy()
        best_angles = angles.copy()
        best_S = S_init
        best_theta = 0.0
        
        np.random.seed(42)
        
        for step in range(2000):
            if time.time() > deadline:
                break
                
            idx = np.random.randint(N)
            delta = np.random.uniform(-0.5, 0.5, 2)
            new_centers = best_centers.copy()
            new_centers[idx] += delta
            
            if np.random.rand() < 0.3:
                new_angles = best_angles.copy()
                new_angles[idx] = np.random.uniform(0, 60)
            else:
                new_angles = best_angles.copy()
                
            best_candidate_S = float('inf')
            best_candidate_theta = 0.0
            
            for theta_try in [0.0, 30.0]:
                S_try = _min_enclosing_hex_side(new_centers, new_angles, theta_try)
                if S_try < best_candidate_S:
                    if _check_validity(new_centers, new_angles, S_try, theta_try):
                        best_candidate_S = S_try
                        best_candidate_theta = theta_try
                        
            if best_candidate_S < best_S:
                best_S = best_candidate_S
                best_centers = new_centers
                best_angles = new_angles
                best_theta = best_candidate_theta
                
        if _check_validity(best_centers, best_angles, best_S, best_theta):
            inner = [[float(cx), float(cy), float(a)] for (cx, cy), a in zip(best_centers, best_angles)]
            return inner, (0.0, 0.0), float(best_S), float(best_theta)
        else:
            inner = [[float(cx), float(cy), 0.0] for cx, cy in pos_array]
            return inner, (0.0, 0.0), float(S_init), 0.0
    else:
        inner = []
        for i in range(N):
            ang = 2 * np.pi * i / N
            r = 5.0
            x = r * np.cos(ang)
            y = r * np.sin(ang)
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0
