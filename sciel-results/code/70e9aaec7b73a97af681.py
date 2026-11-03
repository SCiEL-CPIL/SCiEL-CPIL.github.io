import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    """
    Returns penetration depth if overlapping (negative), or separation distance if not.
    """
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        vertices_list = []
        for i in range(N):
            cx, cy, rot = xys_rots[i]
            v = get_hex_vertices(cx, cy, 1.0, rot)
            vertices_list.append(v)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = sat_overlap(vertices_list[i], vertices_list[j])
                if d < 0:
                    penalty += (-d) ** 2
        
        all_points = np.vstack(vertices_list)
        L = outer_hex_side(all_points, outer_rot)
        
        return L + 100.0 * penalty

    def make_hex_grid(rows, cols, rot_offset=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3
        spacing_y = 1.5
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def make_flower():
        pts = []
        sqrt3 = np.sqrt(3)
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
        # 5 more at distance 2*sqrt3? No, let's place them at hex grid positions
        # Positions at dist 2 from center in hex grid:
        # (2,0), (1,1), (0,2), (-1,2), (-2,2) -> 5 points
        # Convert axial to cartesian:
        # x = sqrt3 * (q + r/2), y = 1.5 * r
        # (2,0) -> (2*sqrt3, 0)
        # (1,1) -> (1.5*sqrt3, 1.5)
        # (0,2) -> (sqrt3, 3)
        # (-1,2) -> (0.5*sqrt3, 3)
        # (-2,2) -> (0, 3)
        # This is not symmetric.
        # Let's just use 5 points from the ring of 12 at dist 2*sqrt3?
        # No, dist from center to center of hex at axial (2,0) is 2*sqrt3.
        # Let's pick 5 that are roughly symmetric.
        # (2,0), (0,2), (-2,1), (-1,-1), (1,-2)?
        # Let's just use a grid for simplicity in initial guess, or random.
        # For flower, let's just add 5 random points near the ring 1.
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([2*sqrt3 * np.cos(ang), 2*sqrt3 * np.sin(ang), 0.0])
        
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    best_L = float('inf')
    best_params = None
    
    configs = []
    configs.append(make_hex_grid(4, 3))
    configs.append(make_hex_grid(3, 4))
    configs.append(make_hex_grid(2, 6))
    configs.append(make_hex_grid(6, 2))
    configs.append(make_flower())
    
    # Add rotated variants
    for base_cfg in configs[:2]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
        
    outer_rots = [0.0, 30.0]
    
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check initial
            xys_rots_init = init_params[:3*N].reshape(N, 3)
            outer_rot_init = init_params[3*N]
            vertices_list_init = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots_init]
            
            has_overlap = False
            for i in range(N):
                for j in range(i+1, N):
                    d = sat_overlap(vertices_list_init[i], vertices_list_init[j])
                    if d < -1e-6:
                        has_overlap = True
                        break
                if has_overlap: break
            
            if not has_overlap:
                all_points = np.vstack(vertices_list_init)
                L_val = outer_hex_side(all_points, outer_rot_init)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            # Polish
            try:
                res = minimize(
                    objective, 
                    init_params, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )
                
                xys_rots = res.x[:3*N].reshape(N, 3)
                outer_rot = res.x[3*N]
                
                vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
                
                has_overlap = False
                for i in range(N):
                    for j in range(i+1, N):
                        d = sat_overlap(vertices_list[i], vertices_list[j])
                        if d < -1e-6:
                            has_overlap = True
                            break
                    if has_overlap: break
                
                if not has_overlap:
                    all_points = np.vstack(vertices_list)
                    L_val = outer_hex_side(all_points, outer_rot)
                    
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()
                        
            except Exception:
                pass
    
    # If time remains, try Differential Evolution
    if time.time() - start_time < DEADLINE - 30:
        bounds = [(-10, 10)] * (2*N) + [(0, 180)] * N + [(0, 60)]
        try:
            de_res = differential_evolution(
                objective,
                bounds,
                maxiter=20,
                popsize=10,
                tol=1e-4,
                seed=42,
                polish=False
            )
            
            xys_rots = de_res.x[:3*N].reshape(N, 3)
            outer_rot = de_res.x[3*N]
            
            vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
            
            has_overlap = False
            for i in range(N):
                for j in range(i+1, N):
                    d = sat_overlap(vertices_list[i], vertices_list[j])
                    if d < -1e-6:
                        has_overlap = True
                        break
                if has_overlap: break
            
            if not has_overlap:
                all_points = np.vstack(vertices_list)
                L_val = outer_hex_side(all_points, outer_rot)
                
                if L_val < best_L:
                    best_L = L_val
                    best_params = de_res.x.copy()
                    
                    # Polish DE result
                    if time.time() - start_time < DEADLINE - 10:
                        res = minimize(
                            objective,
                            best_params,
                            method='Nelder-Mead',
                            options={'maxiter': 100, 'adaptive': True}
                        )
                        xys_rots = res.x[:3*N].reshape(N, 3)
                        outer_rot = res.x[3*N]
                        vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
                        has_overlap = False
                        for i in range(N):
                            for j in range(i+1, N):
                                d = sat_overlap(vertices_list[i], vertices_list[j])
                                if d < -1e-6:
                                    has_overlap = True
                                    break
                            if has_overlap: break
                        if not has_overlap:
                            all_points = np.vstack(vertices_list)
                            L_val = outer_hex_side(all_points, outer_rot)
                            if L_val < best_L:
                                best_L = L_val
                                best_params = res.x.copy()
                                
        except Exception:
            pass

    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0

    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
