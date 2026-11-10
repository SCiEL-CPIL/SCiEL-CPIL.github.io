# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_TIME_LIMIT = 610.0

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])

def _edge_normals(vertices):
    n = len(vertices)
    centroid = vertices.mean(axis=0)
    norms = np.empty((n, 2))
    for i in range(n):
        edge = vertices[(i + 1) % n] - vertices[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-15:
            normal = np.array([1.0, 0.0])
        else:
            normal /= norm
        mid = 0.5 * (vertices[i] + vertices[(i + 1) % n])
        if np.dot(normal, mid - centroid) < 0.0:
            normal = -normal
        norms[i] = normal
    return norms

def _min_gap(v1, v2, n1, n2):
    best = np.inf
    for n in np.vstack((n1, n2)):
        a = v1 @ n
        b = v2 @ n
        if a.max() < b.min():
            g = b.min() - a.max()
        elif b.max() < a.min():
            g = a.min() - b.max()
        else:
            g = 0.0
        if g < best:
            best = g
    return best

def _outer_side_for(inner, outer_center, outer_angle):
    a = np.deg2rad(outer_angle)
    th = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(th), np.sin(th)])
    verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) - outer_center for h in inner])
    max_proj = (verts @ normals.T).max(axis=0)
    return float((2.0 / np.sqrt(3.0)) * max_proj.max())

def _validity(inner, outer_center, outer_angle, side):
    if not np.all(np.isfinite(inner)):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    verts = [_hex_vertices(h[0], h[1], 1.0, h[2]) for h in inner]
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            if _min_gap(verts[i], verts[j], _edge_normals(verts[i]), _edge_normals(verts[j])) <= 1e-6:
                return False
    ov = _hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    for v in verts:
        for k in range(6):
            p1, p2 = ov[k], ov[(k + 1) % 6]
            if np.cross(p2 - p1, v - p1) < -1e-6:
                return False
    return True

def _smooth_max(x, beta=80.0):
    mx = np.max(x)
    if mx > 1000:
        return mx
    return mx + np.log(np.sum(np.exp(beta * (x - mx)))) / beta

def _objective(x, outer_center, outer_angle, beta=80.0):
    inner = x.reshape(11, 3)
    a = np.deg2rad(outer_angle)
    th = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(th), np.sin(th)])
    
    all_verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) - outer_center for h in inner])
    max_proj = (all_verts @ normals.T).max(axis=0)
    side = (2.0 / np.sqrt(3.0)) * _smooth_max(max_proj, beta)
    
    pen = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            dx = inner[i, 0] - inner[j, 0]
            dy = inner[i, 1] - inner[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = np.sqrt(3.0)
            if dist < min_dist:
                pen += (min_dist - dist) ** 2
    return side + 50.0 * pen

def _initial_packing():
    d = np.sqrt(3.0)
    inner = np.zeros((11, 3))
    inner[0] = [0.0, 0.0, 0.0]
    for k in range(6):
        inner[k + 1] = [d * np.cos(np.deg2rad(60.0 * k)), d * np.sin(np.deg2rad(60.0 * k)), 0.0]
    for k in range(4):
        ang = np.deg2rad(30.0 + 90.0 * k)
        inner[7 + k] = [2.0 * d * np.cos(ang), 2.0 * d * np.sin(ang), 0.0]
    return inner

def _polish(inner0, oc, ang, max_iter=300, ftol=1e-12, gtol=1e-10):
    try:
        x0 = inner0.ravel()
        res = minimize(_objective, x0, args=(oc, ang),
                       method="L-BFGS-B", options={"maxiter": max_iter, "ftol": ftol, "gtol": gtol})
        inner2 = res.x.reshape(11, 3)
        side2 = _outer_side_for(inner2, oc, ang)
        if _validity(inner2, oc, ang, side2):
            return inner2, side2
    except Exception:
        pass
    return None, None

def hexagon_packing_11():
    oc0 = np.array([0.0, 0.0])
    best_inner = None
    best_side = np.inf
    best_ang = 0.0
    
    inner0 = _initial_packing()
    s0 = _outer_side_for(inner0, oc0, 0.0)
    if _validity(inner0, oc0, 0.0, s0):
        best_inner = inner0.copy()
        best_side = s0
        best_ang = 0.0
    
    # Try multiple outer angles
    for trial_ang in [0.0, 30.0, 15.0, -15.0, 45.0, -45.0, 60.0, -60.0, 20.0, -20.0]:
        if time.time() - _START_TIME > _TIME_LIMIT - 30.0:
            break
        if best_inner is None:
            inner0 = _initial_packing()
        else:
            inner0 = best_inner.copy()
        inner0[:, 2] = trial_ang
        inner_pol, side_pol = _polish(inner0, oc0, trial_ang, max_iter=400)
        if inner_pol is not None and side_pol < best_side:
            best_inner = inner_pol
            best_side = side_pol
            best_ang = trial_ang
    
    # Differential evolution for global search if time permits
    if time.time() - _START_TIME < _TIME_LIMIT - 60.0:
        def _de_objective(params):
            inner = np.zeros((11, 3))
            for i in range(11):
                inner[i] = [params[i*3], params[i*3+1], params[i*3+2]]
            return _objective(params, oc0, 0.0)
        
        # Bounds: positions in [-5, 5], angles in [-90, 90]
        bounds = []
        for i in range(11):
            bounds.append((-5.0, 5.0))
            bounds.append((-5.0, 5.0))
            bounds.append((-90.0, 90.0))
        
        try:
            de_res = differential_evolution(_de_objective, bounds, maxiter=50, tol=1e-8, seed=42,
                                            polish=False, init='latinhypercube')
            inner_de = de_res.x.reshape(11, 3)
            # Optimize outer angle for this configuration
            for trial_ang in [0.0, 30.0, 15.0, -15.0, 45.0]:
                inner_pol, side_pol = _polish(inner_de, oc0, trial_ang, max_iter=200)
                if inner_pol is not None and side_pol < best_side:
                    best_inner = inner_pol
                    best_side = side_pol
                    best_ang = trial_ang
                    break
        except Exception:
            pass
    
    return best_inner, np.array([oc0[0], oc0[1], best_ang]), float(best_side)

# EVOLVE-BLOCK-END
