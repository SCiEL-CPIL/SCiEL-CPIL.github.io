import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _hex_face_normals(angle_deg):
    a = np.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + i * np.pi / 3 + np.pi / 6
        normals.append([np.cos(theta), np.sin(theta)])
    return np.array(normals)

def _min_enclosing_hexagon(inner_positions, inner_angles, inner_sides):
    all_verts = []
    for (cx, cy, ang), s in zip(inner_positions, inner_sides):
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_side = float('inf')
    best_center = None
    best_angle = 0.0
    
    for phi_deg in np.linspace(0, 60, 1200):
        phi = np.radians(phi_deg)
        normals = []
        for i in range(6):
            theta = phi + i * np.pi / 3 + np.pi / 6
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        projs = all_verts @ normals.T
        max_projs = projs.max(axis=0)
        min_projs = projs.min(axis=0)
        widths = max_projs - min_projs
        required_apothem = np.max(widths) / 2
        s_candidate = required_apothem * 2 / np.sqrt(3)
        
        lo = max_projs - required_apothem
        hi = min_projs + required_apothem
        
        if np.any(lo > hi + 1e-12):
            continue
        
        mid = (lo + hi) / 2
        c, residuals, rank, sv = np.linalg.lstsq(normals, mid, rcond=None)
        
        c_projs = normals @ c
        if np.any(c_projs < lo - 1e-9) or np.any(c_projs > hi + 1e-9):
            continue
        
        if s_candidate < best_side:
            best_side = s_candidate
            best_center = c
            best_angle = phi_deg
            
    if best_center is None:
        centroid = all_verts.mean(axis=0)
        return centroid[0], centroid[1], 100.0, 0.0
        
    return best_center[0], best_center[1], best_side, best_angle

def _min_enclosing_hexagon_fast(inner_positions, inner_angles, inner_sides, n_steps=20):
    all_verts = []
    for (cx, cy, ang), s in zip(inner_positions, inner_sides):
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_side = float('inf')
    
    for phi_deg in np.linspace(0, 60, n_steps):
        phi = np.radians(phi_deg)
        normals = []
        for i in range(6):
            theta = phi + i * np.pi / 3 + np.pi / 6
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        projs = all_verts @ normals.T
        max_projs = projs.max(axis=0)
        min_projs = projs.min(axis=0)
        widths = max_projs - min_projs
        required_apothem = np.max(widths) / 2
        s_candidate = required_apothem * 2 / np.sqrt(3)
        
        if s_candidate < best_side:
            best_side = s_candidate
            
    return best_side

def _check_overlap(positions, angles, s=1.0, tol=1e-9):
    for i in range(N):
        v1 = _hex_vertices(positions[i][0], positions[i][1], s, angles[i])
        n1 = _hex_face_normals(angles[i])
        for j in range(i+1, N):
            v2 = _hex_vertices(positions[j][0], positions[j][1], s, angles[j])
            n2 = _hex_face_normals(angles[j])
            
            # Check n1
            overlap = False
            for n in n1:
                p1 = v1 @ n
                p2 = v2 @ n
                if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                    overlap = True
                    break
            if overlap:
                continue
                
            # Check n2
            for n in n2:
                p1 = v1 @ n
                p2 = v2 @ n
                if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                    overlap = True
                    break
            if not overlap:
                return False # Overlap detected
    return True # No overlap

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    s = 1.0
    sqrt3 = np.sqrt(3)
    
    best_result = None
    best_score = float('inf')
    
    def honeycomb_positions(count, pattern):
        positions = []
        for j, row_len in enumerate(pattern):
            for i in range(row_len):
                x = (i + 0.5 * (j % 2)) * sqrt3
                y = j * 1.5
                positions.append((x, y, 0.0))
        return positions[:count]
    
    patterns = [
        [3, 4, 3, 2],
        [4, 3, 3, 2],
        [2, 3, 4, 3],
        [3, 3, 4, 2],
        [4, 4, 2, 2],
        [2, 2, 4, 4],
        [5, 3, 2, 2],
        [2, 2, 3, 5],
        [6, 3, 3],
        [3, 3, 6],
        [4, 4, 4],
        [2, 4, 4, 2],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [3, 5, 4],
        [4, 5, 3],
        [5, 4, 3],
        [3, 4, 5],
        [1, 5, 5, 1],
        [5, 1, 5, 1],
        [2, 5, 5],
        [5, 5, 2],
        [1, 1, 10],
        [10, 1, 1],
        [2, 2, 8],
        [8, 2, 2],
    ]
    
    for pattern in patterns:
        if sum(pattern) != 12:
            continue
        if len(pattern) > 6:
            continue
            
        if time.time() > deadline:
            break
            
        positions = honeycomb_positions(12, pattern)
        if len(positions) != 12:
            continue
            
        # Initial params
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        init_params = []
        for (x, y, _) in positions:
            init_params.extend([x - cx, y - cy, 0.0])
            
        inner_sides = [1.0] * 12
        
        # Try optimization
        try:
            def obj(params):
                pos = []
                angs = []
                for i in range(N):
                    pos.append((params[3*i], params[3*i+1]))
                    angs.append(params[3*i+2])
                return _min_enclosing_hexagon_fast(pos, angs, inner_sides, n_steps=20)
                
            res = minimize(obj, init_params, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4})
            
            best_params = res.x
            best_pos = []
            best_angles = []
            for i in range(N):
                best_pos.append((best_params[3*i], best_params[3*i+1]))
                best_angles.append(best_params[3*i+2])
                
            # Check validity
            if _check_overlap(best_pos, best_angles, 1.0):
                # Calculate precise side
                cx_precise, cy_precise, side_precise, angle_precise = _min_enclosing_hexagon(
                    best_pos, best_angles, inner_sides)
                
                if side_precise < best_score:
                    best_score = side_precise
                    inner = [(x, y, ang) for (x, y), ang in zip(best_pos, best_angles)]
                    best_result = (inner, (cx_precise, cy_precise), side_precise, angle_precise)
                    
        except Exception:
            pass

    if best_result is None:
        # Fallback
        scale = 1.001
        pattern = [3, 4, 3, 2]
        positions = []
        for j, row_len in enumerate(pattern):
            for i in range(row_len):
                x = (i + 0.5 * (j % 2)) * sqrt3 * scale
                y = j * 1.5 * scale
                positions.append((x, y, 0.0))
        
        cx, cy, side, angle = _min_enclosing_hexagon(positions, [p[2] for p in positions], [1.0]*12)
        inner = [(x, y, rot) for (x, y, rot) in positions]
        return inner, (cx, cy), side, angle
        
    return best_result
