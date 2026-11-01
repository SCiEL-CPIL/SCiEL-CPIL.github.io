import numpy as np
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    verts = np.array([[cx + side * np.cos(a + k * np.pi / 3),
                       cy + side * np.sin(a + k * np.pi / 3)] for k in range(6)])
    return verts

def compute_outer_side_differentiable(inner_verts_list, cx, cy):
    """Compute outer hexagon side length using differentiable operations.
    Outer hexagon has pointy-top orientation with normals at 30, 90, 150, 210, 270, 330 deg.
    """
    angles = np.radians(30 + np.arange(6) * 60)
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    max_proj_per_normal = np.zeros(6)
    for verts in inner_verts_list:
        rel_verts = verts - np.array([cx, cy])
        projs = rel_verts @ normals.T
        max_proj_per_normal = np.maximum(max_proj_per_normal, np.max(projs, axis=0))
    
    max_proj = np.max(max_proj_per_normal)
    return max_proj * 2.0 / np.sqrt(3)

def penetration_depth_smooth(v1, v2, k=5.0):
    """Compute smooth penetration depth penalty"""
    normals = []
    for verts in [v1, v2]:
        for i in range(len(verts)):
            edge = verts[(i + 1) % len(verts)] - verts[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normals.append(normal / norm)
    
    min_overlap = np.inf
    for normal in normals:
        p1 = v1 @ normal
        p2 = v2 @ normal
        overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
        if overlap < min_overlap:
            min_overlap = overlap
    
    # Smooth penalty: (smooth_max(0, overlap))^2
    sm = min_overlap + np.logaddexp(k*(0.0-min_overlap), 0) / k
    return sm ** 2

def objective(params):
    inner_params = params[:3 * N]
    cx, cy = params[3 * N:3 * N + 2]
    
    inner_verts_list = []
    for i in range(N):
        x, y, a = inner_params[3*i:3*i+3]
        verts = hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        
    s = compute_outer_side_differentiable(inner_verts_list, cx, cy)
    
    total_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            pd = penetration_depth_smooth(inner_verts_list[i], inner_verts_list[j])
            total_penalty += pd
            
    return s + 10.0 * total_penalty

def initial_guess(seed=0):
    rng = np.random.RandomState(seed)
    spacing = np.sqrt(3) * 1.05
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = k * np.pi / 3
        positions.append((spacing * np.cos(theta), spacing * np.sin(theta)))
    for k in range(5):
        theta = k * 2 * np.pi / 5 + np.pi / 6
        positions.append((2 * spacing * np.cos(theta), 2 * spacing * np.sin(theta)))
        
    params = []
    for x, y in positions:
        params.extend([x + rng.normal(0, 0.1), y + rng.normal(0, 0.1), rng.uniform(-15, 15)])
    params.extend([0.0, 0.0])
    return np.array(params)

def construct():
    best_params = None
    best_score = np.inf
    
    for seed in range(8):
        x0 = initial_guess(seed)
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    
    if best_params is None:
        best_params = initial_guess(0)
    
    params = best_params
    inner = []
    for i in range(N):
        x, y, a = params[3*i:3*i+3]
        inner.append([float(x), float(y), float(a)])
    cx, cy = params[3*N:3*N+2]
    
    inner_verts_list = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    s = compute_outer_side_differentiable(inner_verts_list, cx, cy)
    return inner, (float(cx), float(cy)), float(s), 0.0
