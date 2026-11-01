import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_support(x, y, angle_deg, u_x, u_y):
    """Support function of a unit regular hexagon centered at (x,y) with given angle."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    verts_x = np.cos(np.arange(6) * np.pi / 3)
    verts_y = np.sin(np.arange(6) * np.pi / 3)
    rx = cos_a * verts_x - sin_a * verts_y
    ry = sin_a * verts_x + cos_a * verts_y
    px = rx + x
    py = ry + y
    return np.max(px * u_x + py * u_y)

def outer_hex_support(cx, cy, S, angle_deg, u_x, u_y):
    """Support function of outer hexagon."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    verts_x = S * np.cos(np.arange(6) * np.pi / 3)
    verts_y = S * np.sin(np.arange(6) * np.pi / 3)
    rx = cos_a * verts_x - sin_a * verts_y
    ry = sin_a * verts_x + cos_a * verts_y
    px = rx + cx
    py = ry + cy
    return np.max(px * u_x + py * u_y)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Honeycomb initial arrangement
    positions = []
    angles = []
    dx = np.sqrt(3)  # horizontal spacing
    dy = 1.5  # vertical spacing
    
    # 4 columns, 3 rows
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append([x, y])
            angles.append(0.0)
    
    positions = np.array(positions)
    positions -= positions.mean(axis=0)
    
    # Initial guess for outer hexagon
    cx, cy = 0.0, 0.0
    S_init = 5.0  # Safe initial guess
    outer_angle = 0.0
    
    def unpack(v):
        pos = v[:36].reshape(12, 3)
        cx = v[36]
        cy = v[37]
        S = v[38]
        outer_angle = v[39]
        return pos, cx, cy, S, outer_angle

    def get_violation(v):
        """Calculate the maximum constraint violation. 0 means feasible."""
        pos, cx, cy, S, outer_angle = unpack(v)
        if S <= 0:
            return 100.0
            
        max_viol = 0.0
        
        # Containment constraints
        oa = np.radians(outer_angle)
        outer_normals = []
        for k in range(6):
            theta = oa + np.pi/6 + k * np.pi / 3
            outer_normals.append((np.cos(theta), np.sin(theta)))
            
        for i in range(12):
            xi, yi, ai = pos[i]
            for (nx, ny) in outer_normals:
                sup_i = hex_support(xi, yi, ai, nx, ny)
                sup_o = outer_hex_support(cx, cy, S, outer_angle, nx, ny)
                viol = sup_i - sup_o
                if viol > max_viol:
                    max_viol = viol
                    
        # Disjointness constraints
        for i in range(12):
            for j in range(i+1, 12):
                xi, yi, ai = pos[i]
                xj, yj, aj = pos[j]
                
                dirs_check = []
                for k in range(6):
                    theta = np.radians(ai) + np.pi/6 + k * np.pi / 3
                    dirs_check.append((np.cos(theta), np.sin(theta)))
                for k in range(6):
                    theta = np.radians(aj) + np.pi/6 + k * np.pi / 3
                    dirs_check.append((np.cos(theta), np.sin(theta)))
                
                for (ux, uy) in dirs_check:
                    sup_i = hex_support(xi, yi, ai, ux, uy)
                    sup_j_neg = hex_support(xj, yj, aj, -ux, -uy)
                    # Constraint: sup_i + sup_j_neg >= 0
                    # Violation if sum < 0
                    viol = -(sup_i + sup_j_neg)
                    if viol > max_viol:
                        max_viol = viol
                        
        return max_viol

    def objective(v):
        pos, cx, cy, S, outer_angle = unpack(v)
        viol = get_violation(v)
        # Penalty for violation
        return S + 10.0 * max(0.0, viol)

    # Initial point
    v0 = np.zeros(40)
    for i in range(12):
        v0[3*i] = positions[i, 0]
        v0[3*i+1] = positions[i, 1]
        v0[3*i+2] = angles[i]
    v0[36] = cx
    v0[37] = cy
    v0[38] = S_init
    v0[39] = outer_angle
    
    # Bounds
    bounds = []
    for i in range(12):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 180))
    bounds.append((-10, 10))
    bounds.append((-10, 10))
    bounds.append((2.0, 10.0))
    bounds.append((0, 60))
    
    best_S = np.inf
    best_v = v0.copy()
    best_feasible_v = None
    best_feasible_S = np.inf
    
    np.random.seed(42)
    
    # Check initial solution
    v0_viol = get_violation(v0)
    if v0_viol < 1e-6:
        best_feasible_v = v0.copy()
        best_feasible_S = v0[38]
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        if restart == 0:
            v_current = v0.copy()
        else:
            # Perturb the best feasible solution if available, else v0
            base_v = best_feasible_v if best_feasible_v is not None else v0
            v_current = base_v + np.random.normal(0, 0.5, 40)
            v_current[38] = np.clip(v_current[38], 2.0, 10.0)
            
        try:
            result = minimize(objective, v_current, method='SLSQP', 
                            bounds=bounds,
                            options={'maxiter': 100, 'ftol': 1e-8})
            
            v_res = result.x
            S_res = v_res[38]
            viol_res = get_violation(v_res)
            
            # Update best overall (for perturbation)
            if result.fun < best_S:
                best_S = result.fun
                best_v = v_res.copy()
                
            # Update best feasible
            if viol_res < 1e-6:
                if S_res < best_feasible_S:
                    best_feasible_S = S_res
                    best_feasible_v = v_res.copy()
                    
        except Exception as e:
            pass
        
        if time.time() > deadline:
            break
            
    # Finalize
    if best_feasible_v is not None:
        final_v = best_feasible_v
    else:
        # Fallback to initial safe configuration
        final_v = v0.copy()
        
    pos, cx, cy, S, outer_angle = unpack(final_v)
    
    inner = []
    for i in range(12):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])])
    
    return inner, (float(cx), float(cy)), float(S), float(outer_angle)
