import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    r = side
    angles = angle_rad + np.pi/3 * np.arange(6)
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def _construct_loss(params):
    inner_params = params[:36].reshape(12, 3)
    ocx, ocy, oside, oangle = params[36], params[37], params[38], params[39]
    
    if oside <= 0.1:
        return 1e6
    
    loss = oside
    
    # Containment: all vertices of inner hexagons inside outer
    oangle_rad = np.radians(oangle)
    normals = oangle_rad + np.pi/3 * np.arange(6)
    inradius = oside * np.sqrt(3) / 2
    
    for i in range(12):
        ix, iy, iangle = inner_params[i]
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        for vx, vy in verts:
            dx = vx - ocx
            dy = vy - ocy
            projections = dx * np.cos(normals) + dy * np.sin(normals)
            max_proj = np.max(projections)
            if max_proj > inradius:
                loss += 10.0 * (max_proj - inradius) ** 2
    
    # Disjointness: center distance >= 2.0
    for i in range(12):
        for j in range(i+1, 12):
            dx = inner_params[i,0] - inner_params[j,0]
            dy = inner_params[i,1] - inner_params[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 2.0:
                loss += 5.0 * (2.0 - dist) ** 2
    
    return loss

def construct():
    np.random.seed(42)
    
    positions = [
        (0, 0), (2, 0), (-2, 0),
        (1, 1.732), (-1, 1.732),
        (1, -1.732), (-1, -1.732),
        (4, 0), (-4, 0),
        (2, 3.464), (-2, 3.464),
        (2, -3.464), (-2, -3.464),
    ]
    inner = [[float(x), float(y), 0.0] for x, y in positions]
    
    x0 = np.array([v for p in inner for v in p] + [0.0, 0.0, 5.0, 0.0])
    
    best_params = x0.copy()
    best_loss = _construct_loss(x0)
    
    for attempt in range(3):
        result = minimize(_construct_loss, x0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7})
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x.copy()
        
        # Perturb for next attempt
        x0 = best_params + np.random.normal(0, 0.1, size=40)
        x0[38] = max(x0[38], 3.0)  # keep side length reasonable
    
    params = best_params
    inner_params = params[:36].reshape(12, 3)
    ocx, ocy, oside, oangle = params[36], params[37], params[38], params[39]
    
    inner_result = [[float(inner_params[i,0]), float(inner_params[i,1]), float(inner_params[i,2])] for i in range(12)]
    
    return inner_result, (float(ocx), float(ocy)), float(oside), float(oangle)
