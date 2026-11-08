import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _get_hex_vertices(center, angle_deg):
    """Get vertices of a unit hexagon."""
    cx, cy = center
    angles = np.deg2rad(np.arange(0, 360, 60) + angle_deg)
    xs = cx + np.cos(angles)
    ys = cy + np.sin(angles)
    return np.column_stack((xs, ys))

def _min_outer_side(vertices, theta):
    """
    Calculate minimal side length of a regular hexagon (rotated by theta)
    enclosing the given vertices, centered at origin.
    """
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    x = vertices[:, 0] * cos_t - vertices[:, 1] * sin_t
    y = vertices[:, 0] * sin_t + vertices[:, 1] * cos_t
    
    vals = np.array([
        2.0 * x / np.sqrt(3),
        (x + np.sqrt(3) * y) / np.sqrt(3),
        (-x + np.sqrt(3) * y) / np.sqrt(3),
        -2.0 * x / np.sqrt(3),
        (-x - np.sqrt(3) * y) / np.sqrt(3),
        (x - np.sqrt(3) * y) / np.sqrt(3)
    ])
    
    return np.max(vals)

def _objective(params, vertices):
    xc, yc, theta = params
    v = vertices.copy()
    v[:, 0] -= xc
    v[:, 1] -= yc
    return _min_outer_side(v, theta)

def _optimize_cluster(cluster_centers, angles):
    all_vertices = []
    for c, a in zip(cluster_centers, angles):
        verts = _get_hex_vertices(c, a)
        all_vertices.append(verts)
    
    vertices = np.vstack(all_vertices)
    
    xc0 = np.mean(vertices[:, 0])
    yc0 = np.mean(vertices[:, 1])
    theta0 = 0.0
    
    bounds = [(xc0 - 10, xc0 + 10), (yc0 - 10, yc0 + 10), (0, np.pi/3)]
    
    try:
        res = minimize(_objective, [xc0, yc0, theta0], args=(vertices,), method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-5})
        if res.success:
            xc, yc, theta = res.x
            side = res.fun
            return side, (xc, yc), theta
    except Exception:
        pass
    
    try:
        res = minimize(_objective, [xc0, yc0, theta0], args=(vertices,), method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        xc, yc, theta = res.x
        side = res.fun
        return side, (xc, yc), theta
    except Exception:
        return float('inf'), (0,0), 0.0

def construct():
    start_time = time.time()
    
    e1 = np.array([np.sqrt(3), 0.0])
    e2 = np.array([np.sqrt(3)/2, 1.5])
    
    def lattice_to_cartesian(q, r):
        return q * e1 + r * e2
    
    candidates = [
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,-2), (-1,-2), (-2,0)],
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)],
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (-2,0), (-2,1)],
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,2), (-2,1), (-2,0)]
    ]
    
    best_side = float('inf')
    best_config = None
    
    for cluster in candidates:
        if time.time() - start_time > 145:
            break
            
        centers = [lattice_to_cartesian(q, r) for q, r in cluster]
        angles = [0.0] * 11
        
        try:
            side, center, theta = _optimize_cluster(centers, angles)
            if side < best_side:
                best_side = side
                best_config = (centers, angles, center, theta)
        except Exception:
            pass
            
    if best_config is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    centers, angles, center, theta = best_config
    
    inner = []
    for c, a in zip(centers, angles):
        inner.append([c[0], c[1], a])
        
    return inner, center, best_side, np.degrees(theta)
# EVOLVE-BLOCK-END
