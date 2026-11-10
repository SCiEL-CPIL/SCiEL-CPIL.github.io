import numpy as np
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600
    
    def hex_vertices(cx, cy, d, angle_deg):
        a = np.radians(angle_deg)
        angles = a + np.radians(60) * np.arange(6)
        return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])
    
    def compute_min_outer_side(inner_positions, inner_angles):
        centroid = np.mean(inner_positions, axis=0)
        centered_positions = inner_positions - centroid
        
        all_vertices = []
        for i in range(len(centered_positions)):
            cx, cy = centered_positions[i]
            ang = inner_angles[i]
            verts = hex_vertices(cx, cy, 1.0, ang)
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices)
        
        normal_angles = np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_s = 0.0
        for normal in normals:
            projections = all_vertices @ normal
            max_proj = np.max(projections)
            min_s = 2.0 / np.sqrt(3.0) * max_proj
            max_s = max(max_s, min_s)
        
        return max_s
    
    d_adj = np.sqrt(3)
    
    def axial_to_cartesian(q, r, spacing):
        x = spacing * (q + r / 2.0)
        y = spacing * (np.sqrt(3) / 2.0) * r
        return x, y
    
    # Build initial hexagonal packing
    all_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    all_axial.extend(ring1)
    
    # Ring 2: 12 positions, pick 5
    ring2_all = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            dist = (abs(q) + abs(r) + abs(q + r)) // 2
            if dist == 2:
                ring2_all.append((q, r))
    
    # Pick 5 for symmetry: use positions at roughly 72-degree intervals
    # (2,0), (0,2), (-2,1), (-2,-1), (0,-2)
    ring2_pick = [(2, 0), (0, 2), (-2, 1), (-2, -1), (0, -2)]
    all_axial.extend(ring2_pick)
    
    inner_pos = np.array([axial_to_cartesian(q, r, d_adj) for q, r in all_axial])
    inner_ang = np.zeros(12)
    
    s0 = compute_min_outer_side(inner_pos, inner_ang)
    
    from scipy.optimize import minimize
    
    def obj(p):
        pos = p[:24].reshape(12, 2)
        ang = p[24:]
        return compute_min_outer_side(pos, ang)
    
    x0 = np.concatenate([inner_pos.ravel(), inner_ang])
    best_p = x0.copy()
    best_s = s0
    
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline - 60:
            break
        if i == 0:
            p0 = x0.copy()
        else:
            p0 = x0 + np.random.randn(36) * 0.2
        try:
            res = minimize(obj, p0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_s:
                best_s = res.fun
                best_p = res.x.copy()
        except:
            pass
    
    bp = best_p[:24].reshape(12, 2)
    ba = best_p[24:]
    bs = compute_min_outer_side(bp, ba)
    
    return np.column_stack([bp, ba]), np.array([0.0, 0.0, 0.0]), bs
