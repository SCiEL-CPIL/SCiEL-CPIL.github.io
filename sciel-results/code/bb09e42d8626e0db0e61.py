import numpy as np
from scipy.optimize import differential_evolution

N = 11
SQRT3 = np.sqrt(3)

def hex_verts(cx, cy, side=1.0):
    """Vertices of unit hexagon at (cx,cy), flat-top orientation."""
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def sat_disjoint(v1, v2):
    for i in range(6):
        e = v2[(i+1)%6] - v2[i]
        n = np.array([-e[1], e[0]])
        p1 = np.dot(v1, n)
        p2 = np.dot(v2, n)
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
    for i in range(6):
        e = v1[(i+1)%6] - v1[i]
        n = np.array([-e[1], e[0]])
        p1 = np.dot(v1, n)
        p2 = np.dot(v2, n)
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
    return False

def obj(params):
    pos = params[:2*N].reshape(N, 2)
    S = params[2*N]
    verts = [hex_verts(x, y) for x, y in pos]
    
    pen = 0.0
    # Overlap
    for i in range(N):
        for j in range(i+1, N):
            if not sat_disjoint(verts[i], verts[j]):
                # Estimate penetration
                pen += 0.5
    # Boundary
    apothem = S * SQRT3 / 2
    normals = np.array([[np.cos(np.radians(30+k*60)), np.sin(np.radians(30+k*60))] for k in range(6)])
    for v in verts:
        for n in normals:
            d = np.dot(v, n)
            if d.max() > apothem:
                pen += (d.max() - apothem)
    
    return S + 100 * pen

def construct():
    s = SQRT3
    # Initial: hexagonal lattice
    init = []
    # 11 hexagons in a hexagonal arrangement
    # Use a subset of triangular lattice
    lattice = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * s + j * s / 2
            y = j * s * SQRT3 / 2
            lattice.append((x, y))
    # Take 11 closest to origin
    lattice.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = lattice[:N]
    
    x0 = []
    for x, y in init_pos:
        x0.extend([x, y])
    x0.append(8.0)
    
    bounds = [(-5, 5), (-5, 5)] * N + [(3, 12)]
    
    res = differential_evolution(obj, bounds, maxiter=100, popsize=10, seed=42, tol=0.1)
    
    pos = res.x[:2*N].reshape(N, 2)
    S = res.x[2*N]
    
    inner = [[float(x), float(y), 0.0] for x, y in pos]
    return inner, (0.0, 0.0), float(S), 0.0
