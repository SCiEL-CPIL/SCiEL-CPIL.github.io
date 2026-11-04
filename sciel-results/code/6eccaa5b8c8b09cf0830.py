import numpy as np
from scipy.optimize import minimize
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    angles_norm = np.deg2rad([30 + 60*k for k in range(6)])
    normals = np.array([np.cos(angles_norm), np.sin(angles_norm)]).T
    
    def get_vertices(cx, cy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        verts = []
        for j in range(6):
            theta = angle_rad + j * np.pi / 3
            vx = cx + np.cos(theta)
            vy = cy + np.sin(theta)
            verts.append((vx, vy))
        return np.array(verts)

    def point_in_polygon(pt, poly):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            v1 = p2 - p1
            v2 = pt - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if cross < -1e-9:
                return False
        return True

    def min_dist_point_to_edge(pt, e1, e2):
        # Distance from point to line segment
        px, py = pt
        x1, y1 = e1
        x2, y2 = e2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))
        qx = x1 + t * dx
        qy = y1 + t * dy
        return math.hypot(px - qx, py - qy)

    def min_dist_point_to_poly(pt, poly):
        min_d = float('inf')
        for i in range(6):
            d = min_dist_point_to_edge(pt, poly[i], poly[(i+1)%6])
            if d < min_d:
                min_d = d
        return min_d

    def overlap_penalty(verts_list):
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                vi = verts_list[i]
                vj = verts_list[j]
                
                # Check vertices of i inside j
                for v in vi:
                    if point_in_polygon(v, vj):
                        d = min_dist_point_to_poly(v, vj)
                        pen += d * 10.0
                
                # Check vertices of j inside i
                for v in vj:
                    if point_in_polygon(v, vi):
                        d = min_dist_point_to_poly(v, vi)
                        pen += d * 10.0
                        
                # Check edge intersections (approximate penalty)
                # If edges intersect, penalty is significant
                for k in range(6):
                    e1_start = vi[k]
                    e1_end = vi[(k+1)%6]
                    for m in range(6):
                        e2_start = vj[m]
                        e2_end = vj[(m+1)%6]
                        if segments_intersect(e1_start, e1_end, e2_start, e2_end):
                            pen += 5.0
        return pen

    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def objective(params):
        verts_list = []
        max_proj = 0.0
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            verts = get_vertices(x, y, a)
            verts_list.append(verts)
            projs = np.dot(verts, normals)
            max_proj = max(max_proj, np.max(projs))
        
        S = max_proj * 2.0 / np.sqrt(3.0)
        pen = overlap_penalty(verts_list)
        return S + pen

    # Initial guess: Hexagonal lattice
    lattice_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            lattice_points.append((q, r))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    selected = lattice_points[:N]
    
    x0 = []
    for q, r in selected:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        x0.extend([x, y, 0.0])
    
    best_res = None
    best_fun = float('inf')
    
    # Run optimization
    # Use multiple restarts to avoid local minima
    starts = [np.array(x0)]
    for _ in range(3):
        random_pts = []
        for _ in range(N):
            ang = np.random.uniform(0, 2*np.pi)
            rad = np.random.uniform(0, 4)
            x = rad * np.cos(ang)
            y = rad * np.sin(ang)
            a = np.random.uniform(0, 60)
            random_pts.extend([x, y, a])
        starts.append(np.array(random_pts))
        
    for start in starts:
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_fun:
                best_fun = res.fun
                best_res = res
        except Exception:
            continue
            
    if best_res is None:
        # Fallback to initial
        best_res = minimize(objective, np.array(x0), method='Nelder-Mead', options={'maxiter': 500})

    params = best_res.x
    inner = []
    for i in range(N):
        inner.append([float(params[i*3]), float(params[i*3+1]), float(params[i*3+2])])
        
    # Final S calculation
    verts_list = []
    max_proj = 0.0
    for i in range(N):
        x, y, a = inner[i]
        verts = get_vertices(x, y, a)
        verts_list.append(verts)
        projs = np.dot(verts, normals)
        max_proj = max(max_proj, np.max(projs))
    
    S = float(max_proj * 2.0 / np.sqrt(3.0))
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
