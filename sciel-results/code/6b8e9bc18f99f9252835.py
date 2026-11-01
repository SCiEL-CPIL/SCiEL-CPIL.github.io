import numpy as np
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    verts = np.array([[cx + side * np.cos(a + k * np.pi / 3),
                       cy + side * np.sin(a + k * np.pi / 3)] for k in range(6)])
    return verts

def get_normals(verts):
    normals = []
    for i in range(len(verts)):
        edge = verts[(i + 1) % len(verts)] - verts[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals.append(normal / norm)
    return normals

def penetration_depth(v1, v2):
    normals = get_normals(v1) + get_normals(v2)
    min_overlap = np.inf
    for normal in normals:
        p1 = v1 @ normal
        p2 = v2 @ normal
        overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
        if overlap < min_overlap:
            min_overlap = overlap
    return min_overlap

def support_function(verts, normal):
    return (verts @ normal).max()

def objective(params):
    # params: [x1,y1,a1, ..., x12,y12,a12, s]
    inner_params = params[:3 * N]
    s = params[3 * N]
    
    if s < 1.0:
        return 1e6

    # Outer hexagon is fixed at (0,0) with rotation 0
    # Normals of outer hexagon (pointing out)
    outer_normals = np.array([
        [np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(6)
    ])
    
    total_penalty = 0.0
    inner_verts_list = []
    
    for i in range(N):
        x, y, a = inner_params[3*i:3*i+3]
        verts = hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        
        # Constraint: Inner hex must be inside outer hex
        # For each outer normal n, max projection of inner <= s
        for n in outer_normals:
            supp = support_function(verts, n)
            if supp > s:
                total_penalty += (supp - s)
                
    # Pairwise overlap
    for i in range(N):
        for j in range(i+1, N):
            pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
            if pd > 0:
                total_penalty += pd
                
    return s + 100.0 * total_penalty

def initial_guess():
    # Better initial guess: Hexagonal lattice
    # 1 center, 6 around, 5 more?
    # Let's use a grid of hexagons.
    # Spacing for touching hexagons is sqrt(3).
    spacing = np.sqrt(3)
    positions = []
    # 3x3 grid minus corners? 3x3=9. Need 12.
    # Let's try a hexagonal cluster:
    # Ring 0: 1
    # Ring 1: 6
    # Ring 2: 12 (we need 5 more to make 12 total? 1+6+5=12)
    positions.append((0.0, 0.0))
    for k in range(6):
        theta = k * np.pi / 3
        positions.append((spacing * np.cos(theta), spacing * np.sin(theta)))
    # Add 5 from ring 2
    for k in range(5):
        theta = k * 2 * np.pi / 5 + np.pi / 6
        positions.append((2 * spacing * np.cos(theta), 2 * spacing * np.sin(theta)))
        
    params = []
    for x, y in positions:
        params.extend([x, y, 0.0])
    # Initial s estimate
    params.append(5.0)
    return np.array(params)

def construct():
    x0 = initial_guess()
    # Use SLSQP or L-BFGS-B
    # L-BFGS-B is faster but might get stuck in local minima due to non-smoothness
    # Let's try a few restarts or just a good L-BFGS-B
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
    params = result.x
    inner = []
    for i in range(N):
        x, y, a = params[3*i:3*i+3]
        inner.append([float(x), float(y), float(a)])
    s = params[3*N]
    # Return in required format: inner, center, side, angle
    # Center is (0,0), angle is 0
    return inner, (0.0, 0.0), float(s), 0.0
