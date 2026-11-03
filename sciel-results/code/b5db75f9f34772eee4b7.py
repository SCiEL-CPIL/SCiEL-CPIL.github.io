import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    eps = 1e-7
    
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        centers.append((x, y))
        
    r2 = 3.0 + eps
    # Angles 30, 90, 150, 210 correspond to the 4 second-ring hexagons 
    # that are adjacent to the ring-1 hexagons in the standard lattice.
    angles2 = [30, 90, 150, 210]
    for ang in angles2:
        angle = np.deg2rad(ang)
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        centers.append((x, y))

    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])

    # Generate all vertices of inner hexagons
    verts_rel = np.array([
        [np.sqrt(3)/2, 0.5],
        [0, 1],
        [-np.sqrt(3)/2, 0.5],
        [-np.sqrt(3)/2, -0.5],
        [0, -1],
        [np.sqrt(3)/2, -0.5]
    ])
    
    all_verts = []
    for cx, cy, _ in inner:
        for vx, vy in verts_rel:
            all_verts.append([cx + vx, cy + vy])
            
    all_verts = np.array(all_verts)
    
    # Initial guess for outer center: centroid of vertices
    cx0, cy0 = np.mean(all_verts, axis=0)
    
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return 1e9
        cx, cy, S, rot = params
        if S < 0:
            return 1e9
            
        # Outer hexagon is centered at (cx, cy) with side S and rotation rot
        # We need to check if all inner vertices are inside.
        # Transform vertices to outer hexagon's local frame
        dx = all_verts[:, 0] - cx
        dy = all_verts[:, 1] - cy
        
        # Rotate by -rot
        c_rot = np.cos(-rot)
        s_rot = np.sin(-rot)
        lx = dx * c_rot - dy * s_rot
        ly = dx * s_rot + dy * c_rot
        
        # Inequalities for point (lx, ly) to be in hex centered at (0,0) side S (pointy-top)
        # 1. ly <= S
        # 2. -ly <= S
        # 3. (2/sqrt(3))*lx - ly <= S
        # 4. -(2/sqrt(3))*lx - ly <= S
        # 5. (2/sqrt(3))*lx + ly <= S
        # 6. -(2/sqrt(3))*lx + ly <= S
        
        c = 2.0 / np.sqrt(3)
        
        v1 = ly - S
        v2 = -ly - S
        v3 = c * lx - ly - S
        v4 = -c * lx - ly - S
        v5 = c * lx + ly - S
        v6 = -c * lx + ly - S
        
        violations = np.maximum(0, np.maximum(np.maximum(v1, v2), np.maximum(v3, np.maximum(v4, np.maximum(v5, v6)))))
        
        return S + 100 * np.sum(violations**2)

    # Optimize
    x0 = np.array([cx0, cy0, 3.5, 0.0])
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000})
    
    if time.time() - start_time > DEADLINE:
        side = 3.5
        return inner, (0.0, 0.0), side, 0.0
        
    cx_opt, cy_opt, S_opt, rot_opt = res.x
    S_opt += 1e-6 # Safety margin
    
    return inner, (cx_opt, cy_opt), S_opt, np.rad2deg(rot_opt)
