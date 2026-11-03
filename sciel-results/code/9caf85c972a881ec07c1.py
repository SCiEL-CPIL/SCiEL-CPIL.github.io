import math
import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def _hex_vertices(cx, cy, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        ang = a + k * math.pi / 3.0
        verts.append((cx + math.cos(ang), cy + math.sin(ang)))
    return np.array(verts)

def _separation(verts_a, verts_b):
    # minimum distance between two convex polygons (hexagons)
    min_d = float('inf')
    for va in verts_a:
        for vb in verts_b:
            d = math.hypot(va[0]-vb[0], va[1]-vb[1])
            if d < min_d:
                min_d = d
    # Also check edge-edge proximity approximately via vertex-to-edge
    # For hexagons this vertex-vertex is often sufficient for penalty
    return min_d

def _min_pair_sep(centers, angles):
    """Compute minimum separation between all pairs of unit hexagons."""
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_vertices(centers[i*2], centers[i*2+1], angles[i]))
    min_sep = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = _separation(verts_list[i], verts_list[j])
            if d < min_sep:
                min_sep = d
    return min_sep

def _max_proj(centers, ox, oy, beta):
    """Maximum absolute projection of any hexagon vertex onto outer hexagon normals."""
    normals = []
    for k in range(6):
        ang = beta + math.pi/6.0 + k * math.pi/3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    max_p = 0.0
    for i in range(N):
        cx, cy = centers[i*2], centers[i*2+1]
        a = math.radians(angles[i])
        for k in range(6):
            ang = a + k * math.pi/3.0
            vx = cx + math.cos(ang)
            vy = cy + math.sin(ang)
            for nx, ny in normals:
                p = abs((vx - ox)*nx + (vy - oy)*ny)
                if p > max_p:
                    max_p = p
    return max_p

def _objective(params, penalty_w=100.0, contain_w=1.0):
    """
    params: [x1,y1,ang1, x2,y2,ang2, ..., x11,y11,ang11, ox, oy, beta]
    Returns a smooth scalar to minimize.
    """
    centers = params[:N*2]
    angles = params[N*2:N*2+N]
    ox, oy, beta = params[N*2+N], params[N*2+N+1], params[N*2+N+2]
    
    # Containment: max projection of any vertex onto normals
    max_p = _max_proj(centers, ox, oy, beta)
    # We want max_p <= APOTHEM * S - APOTHEM for some S... 
    # Actually minimize (max_p + APOTHEM)/APOTHEM = S
    S = (max_p + APOTHEM) / APOTHEM
    
    # Separation penalty: we want min_sep >= 0 (non-overlapping)
    # Use a smooth penalty: if min_sep < 0, penalize heavily
    min_sep = _min_pair_sep(centers, angles)
    if min_sep < 0:
        sep_penalty = penalty_w * min_sep * min_sep
    else:
        sep_penalty = 0.0
    
    return S + sep_penalty

def _objective_fast(params, penalty_w=100.0):
    """Faster objective using only vertex-vertex distances for separation."""
    centers = params[:N*2]
    angles = params[N*2:N*2+N]
    ox, oy, beta = params[N*2+N], params[N*2+N+1], params[N*2+N+2]
    
    # Compute all vertices
    all_verts = np.zeros((N, 6, 2))
    for i in range(N):
        cx, cy = centers[i*2], centers[i*2+1]
        a = math.radians(angles[i])
        for k in range(6):
            ang = a + k * math.pi/3.0
            all_verts[i, k, 0] = cx + math.cos(ang)
            all_verts[i, k, 1] = cy + math.sin(ang)
    
    # Max projection
    normals = np.array([
        (math.cos(beta + math.pi/6.0 + k * math.pi/3.0),
         math.sin(beta + math.pi/6.0 + k * math.pi/3.0))
        for k in range(6)
    ])
    
    max_p = 0.0
    for i in range(N):
        for k in range(6):
            vx, vy = all_verts[i, k]
            for j in range(6):
                p = abs((vx - ox)*normals[j,0] + (vy - oy)*normals[j,1])
                if p > max_p:
                    max_p = p
    S = (max_p + APOTHEM) / APOTHEM
    
    # Separation: vertex-vertex min distance
    min_sep = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(6):
                for m in range(6):
                    dx = all_verts[i,k,0] - all_verts[j,m,0]
                    dy = all_verts[i,k,1] - all_verts[j,m,1]
                    d = dx*dx + dy*dy
                    if d < min_sep:
                        min_sep = d
    min_sep = math.sqrt(min_sep)
    
    if min_sep < 0:
        sep_penalty = penalty_w * min_sep * min_sep
    else:
        sep_penalty = 0.0
    
    return S + sep_penalty

def _final_polish(centers, angles):
    """Given centers and angles, find optimal outer hexagon center and rotation via LP."""
    # Collect all vertices
    all_verts = []
    for i in range(N):
        cx, cy = centers[i*2], centers[i*2+1]
        a = math.radians(angles[i])
        for k in range(6):
            ang = a + k * math.pi/3.0
            all_verts.append((cx + math.cos(ang), cy + math.sin(ang)))
    all_verts = np.array(all_verts)
    
    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    
    # Try multiple beta values
    for i in range(61):
        beta = i * (math.pi/6) / 60
        normals = []
        for k in range(6):
            ang = beta + math.pi/6.0 + k * math.pi/3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in all_verts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px*nx + py*ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px*nx + py*ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_center = (res.x[0], res.x[1])
        except:
            pass
    
    # Refine with scalar minimization around best_beta
    if best_center is not None:
        try:
            def obj_beta(b):
                normals = []
                for k in range(6):
                    ang = b + math.pi/6.0 + k * math.pi/3.0
                    normals.append((math.cos(ang), math.sin(ang)))
                
                c_obj = np.array([0.0, 0.0, 1.0])
                A_ub = []
                b_ub = []
                for nx, ny in normals:
                    for px, py in all_verts:
                        A_ub.append([-nx, -ny, -1.0])
                        b_ub.append(-(px*nx + py*ny))
                        A_ub.append([nx, ny, -1.0])
                        b_ub.append(px*nx + py*ny)
                
                A_ub = np.array(A_ub)
                b_ub = np.array(b_ub)
                bounds = [(None, None), (None, None), (0, None)]
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    D = res.x[2]
                    return (D + APOTHEM) / APOTHEM
                return float('inf')
            
            res = minimize(obj_beta, best_beta, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                # Recompute center
                normals = []
                for k in range(6):
                    ang = best_beta + math.pi/6.0 + k * math.pi/3.0
                    normals.append((math.cos(ang), math.sin(ang)))
                c_obj = np.array([0.0, 0.0, 1.0])
                A_ub = []
                b_ub = []
                for nx, ny in normals:
                    for px, py in all_verts:
                        A_ub.append([-nx, -ny, -1.0])
                        b_ub.append(-(px*nx + py*ny))
                        A_ub.append([nx, ny, -1.0])
                        b_ub.append(px*nx + py*ny)
                A_ub = np.array(A_ub)
                b_ub = np.array(b_ub)
                bounds = [(None, None), (None, None), (0, None)]
                res2 = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res2.status == 0:
                    best_center = (res2.x[0], res2.x[1])
        except:
            pass
    
    return best_S, best_beta, best_center

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple grid
        axial = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            pts.append((x, y))
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        max_proj = 0
        for k in range(6):
            ang = math.pi/6.0 + k * math.pi/3.0
            nx, ny = math.cos(ang), math.sin(ang)
            for px, py in pts:
                p = abs((px-cx)*nx + (py-cy)*ny)
                if p > max_proj:
                    max_proj = p
        S = (max_proj + APOTHEM) / APOTHEM
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [cx, cy], float(S), 0.0
    
    best_S = float('inf')
    best_params = None
    
    # Initial configurations to try
    def make_initial(config_type, rng):
        """Generate initial parameter vector."""
        params = np.zeros(N*2 + N + 3)
        
        if config_type == 0:
            # Compact cluster: hexagonal arrangement
            axial = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
            for i, (q, r) in enumerate(axial):
                x = math.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                params[i*2] = x
                params[i*2+1] = y
                params[N*2+i] = 0.0  # angle
        elif config_type == 1:
            # Ring + center
            axial = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,-2)]
            for i, (q, r) in enumerate(axial):
                x = math.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                params[i*2] = x
                params[i*2+1] = y
                params[N*2+i] = 0.0
        elif config_type == 2:
            # Linear-ish
            for i in range(N):
                params[i*2] = (i - 5) * 1.8
                params[i*2+1] = (i % 3 - 1) * 1.5
                params[N*2+i] = 0.0
        elif config_type == 3:
            # Random positions in a disk
            for i in range(N):
                r = rng.uniform(0, 3)
                theta = rng.uniform(0, 2*math.pi)
                params[i*2] = r * math.cos(theta)
                params[i*2+1] = r * math.sin(theta)
                params[N*2+i] = rng.uniform(0, 60)
        elif config_type == 4:
            # Two rows
            for i in range(N):
                row = i // 6
                col = i % 6
                params[i*2] = (col - 2.5) * 1.8
                params[i*2+1] = (row - 0.5) * 1.6
                params[N*2+i] = 0.0
        elif config_type == 5:
            # Diamond
            positions = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(1,1),(-1,1),(1,-1),(-1,-1)]
            for i, (q, r) in enumerate(positions):
                x = math.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                params[i*2] = x
                params[i*2+1] = y
                params[N*2+i] = 0.0
        else:
            # Random
            for i in range(N):
                params[i*2] = rng.uniform(-3, 3)
                params[i*2+1] = rng.uniform(-3, 3)
                params[N*2+i] = rng.uniform(0, 60)
        
        # Outer hexagon initial guess
        params[N*2+N] = 0.0  # ox
        params[N*2+N+1] = 0.0  # oy
        params[N*2+N+2] = 0.0  # beta
        
        return params
    
    # Multiple restarts with different initial configs
    rng = np.random.RandomState(42)
    n_restarts = 8
    configs_per_restart = 3
    
    for restart in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        for cfg in range(configs_per_restart):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            config_type = (restart * configs_per_restart + cfg) % 7
            x0 = make_initial(config_type, rng)
            
            # Add small random perturbation
            x0 = x0 + rng.normal(0, 0.1, size=x0.shape)
            
            try:
                res = minimize(_objective_fast, x0, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8},
                              tol=1e-10)
                
                if res.fun < best_S and res.fun < 20:  # Sanity check
                    best_S = res.fun
                    best_params = res.x.copy()
            except:
                pass
    
    # Also try with different penalty weights
    if best_params is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        for pw in [50.0, 200.0, 500.0]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            try:
                def obj_pw(params, pw=pw):
                    return _objective_fast(params, penalty_w=pw)
                res = minimize(obj_pw, best_params, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x.copy()
            except:
                pass
    
    if best_params is None:
        # Fallback
        axial = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            pts.append((x, y))
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        max_proj = 0
        for k in range(6):
            ang = math.pi/6.0 + k * math.pi/3.0
            nx, ny = math.cos(ang), math.sin(ang)
            for px, py in pts:
                p = abs((px-cx)*nx + (py-cy)*ny)
                if p > max_proj:
                    max_proj = p
        S = (max_proj + APOTHEM) / APOTHEM
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [cx, cy], float(S), 0.0
    
    # Extract centers and angles
    centers = best_params[:N*2]
    angles = best_params[N*2:N*2+N]
    
    # Final polish: optimize outer hexagon placement
    S_final, beta_final, center_final = _final_polish(centers, angles)
    
    if center_final is None or S_final == float('inf'):
        # Use the optimizer's outer hexagon
        ox, oy, beta = best_params[N*2+N], best_params[N*2+N+1], best_params[N*2+N+2]
        S_final = best_S
        beta_final = beta
        center_final = (ox, oy)
    
    cx, cy = center_final
    angle_deg = math.degrees(beta_final)
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(centers[i*2]), float(centers[i*2+1]), float(angles[i])])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
