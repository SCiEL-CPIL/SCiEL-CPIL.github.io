"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses lattice initialization and local refinement with fixed angles.
"""
N = 12

import numpy as np
import time
import math
from scipy.optimize import minimize

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _support_hex(center, angle_deg, side, dx, dy):
    """Support function of a regular hexagon in direction (dx, dy)."""
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

def _min_enclosing_hex_side(centers, angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length of a regular hexagon with given outer angle
    that contains all inner hexagons.
    """
    outer_angle = np.radians(outer_angle_deg)
    # The 6 outward normals of the outer hexagon
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
            h = _support_hex(np.array([cx, cy]), ang, 1.0, dx, dy)
            dot_center = cx * dx + cy * dy
            needed = dot_center + h
            if needed > max_ext:
                max_ext = needed
    
    # The apothem of the outer hexagon must be at least max_ext
    # Apothem = S * sqrt(3) / 2
    # So S = 2 * max_ext / sqrt(3)
    S = 2.0 * max_ext / np.sqrt(3.0)
    return S

def _check_validity(centers, angles, S, outer_angle_deg=0.0, tol=1e-6):
    """
    Check if all inner hexagons are inside the outer hexagon and disjoint.
    """
    # Check containment
    outer_verts = _hex_verts(0.0, 0.0, S, outer_angle_deg)
    inner_verts_list = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        iv = _hex_verts(cx, cy, 1.0, ang)
        inner_verts_list.append(iv)
        for pt in iv:
            # Check if point is inside outer hex
            # Using cross product method for convex polygon
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
    
    # Check disjointness
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]
            
            def intersect(v1, v2):
                def point_in_poly(pt, poly):
                    n = len(poly)
                    for i in range(n):
                        v1 = poly[i]
                        v2 = poly[(i+1)%n]
                        edge = v2 - v1
                        rel = pt - v1
                        cross = edge[0]*rel[1] - edge[1]*rel[0]
                        if cross < -1e-9:
                            return False
                    return True

                for pt in v1:
                    if point_in_poly(pt, v2):
                        return True
                for pt in v2:
                    if point_in_poly(pt, v1):
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

                for i in range(6):
                    p1 = vi[i]
                    p2 = vi[(i+1)%6]
                    for j in range(6):
                        p3 = vj[j]
                        p4 = vj[(j+1)%6]
                        if segments_intersect(p1, p2, p3, p4):
                            return True
                return False
            
            if intersect(vi, vj):
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial lattice configuration
    u = np.array([1.5, np.sqrt(3.0)/2.0])
    v = np.array([-1.5, np.sqrt(3.0)/2.0])
    scale = 1.02
    
    inner = []
    for i in range(4):
        for j in range(3):
            pos = scale * (i * u + j * v)
            inner.append([float(pos[0]), float(pos[1]), 0.0])
            
    cx_mean = np.mean([p[0] for p in inner])
    cy_mean = np.mean([p[1] for p in inner])
    inner = [[p[0]-cx_mean, p[1]-cy_mean, p[2]] for p in inner]
    
    best_inner = [p[:] for p in inner]
    best_S = _min_enclosing_hex_side(
        np.array([[p[0], p[1]] for p in best_inner]), 
        np.array([0.0]*N), 0.0)
    
    # Optimization parameters: [x0, y0, x1, y1, ...] (angles fixed at 0)
    x0 = np.array([p[0] for p in inner])
    y0 = np.array([p[1] for p in inner])
    x0_full = np.concatenate([x0, y0])
    
    def objective(params):
        centers = np.array([params[2*i], params[2*i+1]] for i in range(N))
        
        # Calculate S
        S = _min_enclosing_hex_side(centers, np.zeros(N), 0.0)
        
        # Penalty for overlap
        penalty = 0.0
        min_dist = np.sqrt(3.0)
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_dist:
                    penalty += (min_dist - dist)**2 * 1000.0
                elif dist < 2.0:
                    penalty += (2.0 - dist)**2 * 10.0
        
        return S + penalty
    
    # Nelder-Mead optimization
    try:
        res = minimize(objective, x0_full, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_S:
            best_params = res.x
            best_centers = np.array([best_params[2*i], best_params[2*i+1]] for i in range(N))
            best_S = _min_enclosing_hex_side(best_centers, np.zeros(N), 0.0)
            
            # Verify validity
            if _check_validity(best_centers, np.zeros(N), best_S, 0.0):
                best_inner = [[float(cx), float(cy), 0.0] for cx, cy in best_centers]
                
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 5.0:
        pass
    else:
        # Try a few different initial scales and optimize
        for trial_scale in [1.0, 1.01, 1.05]:
            if time.time() > deadline - 10.0:
                break
                
            inner_trial = []
            for i in range(4):
                for j in range(3):
                    pos = trial_scale * (i * u + j * v)
                    inner_trial.append([float(pos[0]), float(pos[1]), 0.0])
                    
            cx_mean = np.mean([p[0] for p in inner_trial])
            cy_mean = np.mean([p[1] for p in inner_trial])
            inner_trial = [[p[0]-cx_mean, p[1]-cy_mean, p[2]] for p in inner_trial]
            
            x0_t = np.array([p[0] for p in inner_trial])
            y0_t = np.array([p[1] for p in inner_trial])
            x0_full_t = np.concatenate([x0_t, y0_t])
            
            try:
                res_t = minimize(objective, x0_full_t, method='Nelder-Mead', 
                               options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res_t.fun < best_S:
                    best_params = res_t.x
                    best_centers = np.array([best_params[2*i], best_params[2*i+1]] for i in range(N))
                    S_t = _min_enclosing_hex_side(best_centers, np.zeros(N), 0.0)
                    
                    if _check_validity(best_centers, np.zeros(N), S_t, 0.0):
                        best_inner = [[float(cx), float(cy), 0.0] for cx, cy in best_centers]
                        best_S = S_t
                        
            except Exception:
                pass
    
    return best_inner, (0.0, 0.0), float(best_S), 0.0
