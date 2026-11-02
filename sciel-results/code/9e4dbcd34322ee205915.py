import numpy as np
from scipy.optimize import minimize
import time
import random

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Pre-compute hexagon vertices for unit side, angle 0
    # Vertices are relative to center
    # (1, 0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1, 0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
    base_hex = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    def rotate_hex(angle):
        c = np.cos(angle)
        s = np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        return base_hex @ rot.T

    def get_hex_vertices(cx, cy, angle):
        # Returns 6x2 array
        return cx + cy + rotate_hex(angle) # Wait, cx, cy is 2D? No, cx, cy are scalars.
        # Correction:
        verts = rotate_hex(angle)
        return np.column_stack([verts[:, 0] + cx, verts[:, 1] + cy])

    def penetration_depth(v1, v2):
        # v1, v2: 6x2
        # Separating Axis Theorem for convex polygons
        # For hexagons, we can just check the 6 axes of v1 and 6 axes of v2
        # But since they are regular hexagons, we can be smarter?
        # No, generic SAT is fast enough for N=12 (66 pairs)
        
        # Normals for v1
        edges1 = np.roll(v1, -1, axis=0) - v1
        normals1 = np.column_stack([edges1[:, 1], -edges1[:, 0]])
        normals1 /= (np.linalg.norm(normals1, axis=1)[:, np.newaxis] + 1e-15)
        
        # Normals for v2
        edges2 = np.roll(v2, -1, axis=0) - v2
        normals2 = np.column_stack([edges2[:, 1], -edges2[:, 0]])
        normals2 /= (np.linalg.norm(normals2, axis=1)[:, np.newaxis] + 1e-15)
        
        all_normals = np.vstack([normals1, normals2])
        
        min_overlap = np.inf
        for n in all_normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
                
        return max(0.0, min_overlap)

    def containment_violation(inner_verts, outer_verts):
        # Check if inner_verts is inside outer_verts
        # Project inner vertices onto outer normals
        edges = np.roll(outer_verts, -1, axis=0) - outer_verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        normals /= (np.linalg.norm(normals, axis=1)[:, np.newaxis] + 1e-15)
        
        proj_inner = inner_verts @ normals.T
        proj_outer_max = (outer_verts @ normals.T).max(axis=0)
        
        viol = proj_inner.max(axis=0) - proj_outer_max
        return max(0.0, np.max(viol))

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        # params: [x1, y1, a1, ..., x12, y12, a12, S, theta]
        # 12 * 3 + 2 = 38 params
        
        positions = params[:3*N].reshape(N, 3)
        S = params[3*N]
        theta = params[3*N + 1]
        
        if S < 1.0:
            return 1e6
            
        # Outer hexagon centered at (0,0)
        outer_verts = get_hex_vertices(0, 0, theta)
        
        penalty = 0.0
        inner_verts_list = []
        
        for i in range(N):
            x, y, a = positions[i]
            verts = get_hex_vertices(x, y, a)
            inner_verts_list.append(verts)
            
            # Containment
            viol = containment_violation(verts, outer_verts)
            penalty += viol * viol
            
        # Overlaps
        for i in range(N):
            for j in range(i+1, N):
                pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                penalty += pd * pd
                
        return S + 100.0 * penalty

    def get_initial_config():
        # Flower pattern
        pos = []
        # Center
        pos.append([0.0, 0.0, 0.0])
        # Ring 1 (6 hexes)
        for k in range(6):
            a = np.radians(60 * k)
            # Distance between centers of touching hexes is 2 * apothem = 2 * sqrt(3)/2 * 1 = sqrt(3)
            r = np.sqrt(3)
            pos.append([r * np.cos(a), r * np.sin(a), 0.0])
        # Ring 2 (5 hexes to make 12)
        # Place them in the gaps or further out
        # Let's put them at radius 2*sqrt(3) roughly
        for k in range(5):
            a = np.radians(60 * k + 30) # Offset
            r = 2 * np.sqrt(3)
            pos.append([r * np.cos(a), r * np.sin(a), 0.0])
            
        pos = np.array(pos)
        S_init = 4.0
        theta_init = 0.0
        
        # Add noise
        pos[:, 0] += np.random.normal(0, 0.1, N)
        pos[:, 1] += np.random.normal(0, 0.1, N)
        pos[:, 2] += np.random.uniform(-0.1, 0.1, N)
        
        return np.concatenate([pos.flatten(), [S_init], [theta_init]])

    best_params = None
    best_score = np.inf

    # Run multiple restarts
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        x0 = get_initial_config()
        
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-5,
                    'fatol': 1e-5,
                    'adaptive': True
                }
            )
            
            if res.fun < best_score:
                # Check validity
                params = res.x
                positions = params[:3*N].reshape(N, 3)
                S = params[3*N]
                theta = params[3*N + 1]
                
                outer_verts = get_hex_vertices(0, 0, theta)
                valid = True
                inner_verts_list = []
                for k in range(N):
                    x, y, a = positions[k]
                    verts = get_hex_vertices(x, y, a)
                    inner_verts_list.append(verts)
                    if containment_violation(verts, outer_verts) > 1e-6:
                        valid = False
                        break
                
                if valid:
                    for k in range(N):
                        for l in range(k+1, N):
                            if penetration_depth(inner_verts_list[k], inner_verts_list[l]) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    best_score = S
                    best_params = res.x.copy()
                    
        except Exception:
            pass

    if best_params is None:
        # Fallback
        pos, alphas = [], []
        pos.append([0,0]); alphas.append(0)
        for k in range(6):
            a = np.radians(60*k)
            pos.append([np.sqrt(3)*np.cos(a), np.sqrt(3)*np.sin(a)])
            alphas.append(0)
        for k in range(5):
            a = np.radians(60*k+30)
            pos.append([2*np.sqrt(3)*np.cos(a), 2*np.sqrt(3)*np.sin(a)])
            alphas.append(0)
        S = 5.0
        theta = 0.0
        inner = [[pos[i][0], pos[i][1], np.degrees(alphas[i])] for i in range(N)]
        return inner, (0.0, 0.0), S, np.degrees(theta)

    # Extract results
    final_params = best_params
    positions = final_params[:3*N].reshape(N, 3)
    S = final_params[3*N]
    theta = final_params[3*N + 1]
    
    inner = []
    for i in range(N):
        x, y, a = positions[i]
        inner.append([x, y, np.degrees(a)])
        
    return inner, (0.0, 0.0), S, np.degrees(theta)

