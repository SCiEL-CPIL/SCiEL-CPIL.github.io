import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    def get_hex_vertices(cx, cy, angle_deg):
        # Vertices of a unit hexagon (side length 1)
        # Side length 1 means distance from center to vertex is 1?
        # "Unit regular hexagons of side length 1"
        # Distance from center to vertex = 1.
        # Distance from center to side = sqrt(3)/2.
        angles = [math.radians(angle_deg + 60*k) for k in range(6)]
        return [(cx + math.cos(a), cy + math.sin(a)) for a in angles]

    def point_in_polygon(pt, poly):
        x, y = pt
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if (y > p1y) != (y > p2y):
                xint = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
                if x < xint:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def segments_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def polygons_disjoint(v1, v2):
        for v in v1:
            if point_in_polygon(v, v2):
                return False
        for v in v2:
            if point_in_polygon(v, v1):
                return False
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1)%n1]
            for j in range(n2):
                p3 = v2[j]
                p4 = v2[(j+1)%n2]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    def point_polygon_distance(pt, poly):
        min_dist = float('inf')
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                d = math.hypot(pt[0]-p1[0], pt[1]-p1[1])
            else:
                t = ((pt[0]-p1[0])*dx + (pt[1]-p1[1])*dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))
                proj_x = p1[0] + t*dx
                proj_y = p1[1] + t*dy
                d = math.hypot(pt[0]-proj_x, pt[1]-proj_y)
            if d < min_dist:
                min_dist = d
        return min_dist

    def get_min_L(vertices_list):
        max_proj = 0.0
        normals = [math.radians(30 + 60*k) for k in range(6)]
        for v in vertices_list:
            for ang in normals:
                proj = v[0]*math.cos(ang) + v[1]*math.sin(ang)
                if proj > max_proj:
                    max_proj = proj
        return max_proj / (sqrt3 / 2.0)

    def compute_penalty(params):
        penalty = 0.0
        for i in range(N):
            xi, yi, thi = params[3*i], params[3*i+1], params[3*i+2]
            vi = get_hex_vertices(xi, yi, thi)
            for j in range(i+1, N):
                xj, yj, thj = params[3*j], params[3*j+1], params[3*j+2]
                vj = get_hex_vertices(xj, yj, thj)
                
                min_dist = float('inf')
                
                # Check containment
                for v in vi:
                    if point_in_polygon(v, vj):
                        d = point_polygon_distance(v, vj)
                        min_dist = min(min_dist, d)
                
                for v in vj:
                    if point_in_polygon(v, vi):
                        d = point_polygon_distance(v, vi)
                        min_dist = min(min_dist, d)
                
                # If no containment, check edges
                if min_dist == float('inf'):
                    n1 = len(vi)
                    n2 = len(vj)
                    for ii in range(n1):
                        p1 = vi[ii]
                        p2 = vi[(ii+1)%n1]
                        for jj in range(n2):
                            p3 = vj[jj]
                            p4 = vj[(jj+1)%n2]
                            # Distance from p1 to segment p3-p4
                            dx = p4[0] - p3[0]
                            dy = p4[1] - p3[1]
                            if dx == 0 and dy == 0:
                                d = math.hypot(p1[0]-p3[0], p1[1]-p3[1])
                            else:
                                t = ((p1[0]-p3[0])*dx + (p1[1]-p3[1])*dy) / (dx*dx + dy*dy)
                                t = max(0, min(1, t))
                                proj_x = p3[0] + t*dx
                                proj_y = p3[1] + t*dy
                                d = math.hypot(p1[0]-proj_x, p1[1]-proj_y)
                            min_dist = min(min_dist, d)
                            
                            # Distance from p3 to segment p1-p2
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            if dx == 0 and dy == 0:
                                d = math.hypot(p3[0]-p1[0], p3[1]-p1[1])
                            else:
                                t = ((p3[0]-p1[0])*dx + (p3[1]-p1[1])*dy) / (dx*dx + dy*dy)
                                t = max(0, min(1, t))
                                proj_x = p1[0] + t*dx
                                proj_y = p1[1] + t*dy
                                d = math.hypot(p3[0]-proj_x, p3[1]-proj_y)
                            min_dist = min(min_dist, d)
                
                if min_dist < 0.0:
                    penalty += 100.0 * min_dist * min_dist
                else:
                    penalty += 0.01 * max(0, 0.001 - min_dist) ** 2
        
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
        all_vertices = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            th = params[3*i+2]
            verts = get_hex_vertices(x, y, th)
            all_vertices.extend(verts)
        L_req = get_min_L(all_vertices)
        penalty = compute_penalty(params)
        return L_req + penalty

    def objective_strict(params):
        if time.time() > deadline:
            return 1e10
        all_vertices = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            th = params[3*i+2]
            verts = get_hex_vertices(x, y, th)
            all_vertices.extend(verts)
        L_req = get_min_L(all_vertices)
        penalty = compute_penalty(params)
        return L_req + 10.0 * penalty

    # Initializations
    initializations = []
    
    # 1. Hexagonal grid pattern (12 closest to center)
    grid_centers = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q + r) <= 3:
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                grid_centers.append((x, y))
    
    dists = [(math.hypot(x, y), i) for i, (x, y) in enumerate(grid_centers)]
    dists.sort()
    init_grid_idx = [d[1] for d in dists[:12]]
    
    params_grid = []
    for i in init_grid_idx:
        x, y = grid_centers[i]
        params_grid.extend([x, y, 0.0])
    initializations.append(params_grid)
    
    # 2. Random
    np.random.seed(42)
    params_random = []
    for _ in range(N):
        x = np.random.uniform(-3.0, 3.0)
        y = np.random.uniform(-3.0, 3.0)
        th = np.random.uniform(0, 360)
        params_random.extend([x, y, th])
    initializations.append(params_random)

    # 3. Ring
    params_ring = []
    for i in range(12):
        angle = 2 * math.pi * i / 12
        r = 2.5
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        th = angle * 180 / math.pi
        params_ring.extend([x, y, th])
    initializations.append(params_ring)

    best_params = None
    best_score = float('inf')

    # Phase 1: Differential Evolution (Global)
    if time.time() < deadline - 60.0:
        bounds = [(-3.0, 3.0), (-3.0, 3.0), (0.0, 360.0)] * N
        try:
            res = differential_evolution(objective, bounds, maxiter=100, tol=1e-5, 
                                         popsize=15, seed=42, polish=False, init='sohal')
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead on best DE result
    if best_params is not None and time.time() < deadline - 30.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B Polish
    if best_params is not None and time.time() < deadline - 10.0:
        try:
            res = minimize(objective_strict, best_params, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 4: Final Nelder-Mead
    if best_params is not None and time.time() < deadline - 2.0:
        try:
            res = minimize(objective_strict, best_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = params_grid
        
    positions = []
    all_vertices = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        th = best_params[3*i+2]
        positions.append([x, y, th])
        verts = get_hex_vertices(x, y, th)
        all_vertices.extend(verts)
        
    L_final = get_min_L(all_vertices)
    
    # Verify disjointness
    is_valid = True
    verts_list = [get_hex_vertices(pos[0], pos[1], pos[2]) for pos in positions]
    for i in range(N):
        for j in range(i+1, N):
            if not polygons_disjoint(verts_list[i], verts_list[j]):
                is_valid = False
                break
        if not is_valid:
            break
        
    if not is_valid:
        # Fallback to grid
        inner = []
        for i in init_grid_idx:
            x, y = grid_centers[i]
            inner.append([x, y, 0.0])
            
        fb_vertices = []
        for p in inner:
            v = get_hex_vertices(p[0], p[1], p[2])
            fb_vertices.extend(v)
        L_final = get_min_L(fb_vertices)
        
        return inner, (0.0, 0.0), L_final, 0.0

    return positions, (0.0, 0.0), L_final, 0.0
