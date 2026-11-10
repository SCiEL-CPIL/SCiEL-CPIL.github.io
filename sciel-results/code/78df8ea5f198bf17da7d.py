# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 580.0  # safety margin before the 630s wall-clock kill

# ---------- geometry helpers ----------

def _hex_vertices(cx, cy, s, a_deg):
    """(6,2) vertex array of a regular hexagon (center, side, angle_deg)."""
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])

def _edge_normals(vertices):
    """(6,2) outward-pointing unit edge normals of a convex polygon."""
    n = len(vertices)
    centroid = vertices.mean(axis=0)
    norms = np.empty((n, 2))
    for i in range(n):
        edge = vertices[(i + 1) % n] - vertices[i]
        normal = np.array([-edge[1], edge[0]])
        normal /= np.linalg.norm(normal)
        mid = 0.5 * (vertices[i] + vertices[(i + 1) % n])
        if np.dot(normal, mid - centroid) < 0.0:
            normal = -normal
        norms[i] = normal
    return norms

def _min_gap(v1, v2, n1, n2):
    """Minimum separation of two convex hexagons via SAT (12 edge normals).
    Positive => separated, 0 => touching/overlapping."""
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

def _separation(h1, h2):
    v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
    return _min_gap(v1, v2, _edge_normals(v1), _edge_normals(v2))

def _outer_side_for(inner, outer_center, outer_angle):
    """Smallest side of a regular hexagon (center, angle) that encloses all
    inner vertices.  Uses the support function: s >= 2/sqrt(3) * max_k max_p (p·n_k)."""
    a = np.deg2rad(outer_angle)
    th = a + np.deg2rad(30.0 + 60.0 * np.arange(6))  # outward normal directions
    normals = np.column_stack([np.cos(th), np.sin(th)])
    verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) - outer_center for h in inner])
    max_proj = (verts @ normals.T).max(axis=0)
    return float((2.0 / np.sqrt(3.0)) * max_proj.max())

def _validity(inner, outer_center, outer_angle, side):
    """Return True if the packing is valid (separated + contained + finite)."""
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

# ---------- search machinery ----------

def _penalty_aligned(inner):
    """Penalty for overlapping aligned hexagons.
    Assumes all angles are 0.
    Two aligned hexagons overlap if the vector between centers is inside the Minkowski sum (hexagon of inradius sqrt(3)).
    This is equivalent to |u . n_k| <= sqrt(3) for all 6 normals.
    Penalty is sum of squared violations.
    """
    n = len(inner)
    pen = 0.0
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    # Inradius of Minkowski sum is sqrt(3)
    limit = np.sqrt(3.0)
    
    for i in range(n):
        for j in range(i + 1, n):
            u = inner[i, :2] - inner[j, :2]
            projs = np.abs(u @ normals.T)
            # If any projection > limit, they are separated.
            # If all projections <= limit, they overlap.
            # We want to push them apart.
            # Penalty for each direction where they are "too close"
            # Actually, if they overlap, we need to increase the minimum projection.
            # Let's use the minimum projection as the "gap" indicator.
            # If min_proj < limit, they are overlapping.
            # Depth = limit - min_proj.
            min_proj = np.min(projs)
            if min_proj < limit:
                pen += (limit - min_proj) ** 2
    return pen

def _objective(x, outer_center, outer_angle):
    inner = x.reshape(11, 3)
    # We fix angles to 0 for now, so x[2] is ignored in penalty but kept for output
    return _outer_side_for(inner, outer_center, outer_angle) + 100.0 * _penalty_aligned(inner)

def _initial_packing():
    """7 hexagons in a hexagonal lattice (center + ring).
    We will let the optimizer place the other 4.
    To help, we place them in the 4 outer gaps roughly.
    """
    d = np.sqrt(3.0)
    inner = np.zeros((11, 3))
    inner[0] = [0.0, 0.0, 0.0]
    for k in range(6):
        inner[k + 1] = [d * np.cos(np.deg2rad(60.0 * k)), d * np.sin(np.deg2rad(60.0 * k)), 0.0]
    
    # Place 4 more in approximate outer gap positions
    # Gaps are between ring hexagons.
    # Let's place them at distance 2*d from origin, at angles 30, 90, 150, 210?
    # Or just spread them out to avoid initial overlap.
    # A 3x3 grid is a good alternative.
    # Let's try a grid.
    # Grid spacing sqrt(3).
    # Points:
    # (-d, 0), (0,0), (d,0)
    # (-d/2, 1.5), (d/2, 1.5)  <- y is sqrt(3)/2 * sqrt(3) = 1.5? No.
    # y spacing is sqrt(3) * sin(60) = 3/2? No.
    # Triangular lattice:
    # (0,0), (d,0)
    # (d/2, 3/2)
    # Let's generate 9 points in a 3x3 diamond.
    # (0,0)
    # (d,0), (-d,0)
    # (d/2, 1.5), (-d/2, 1.5), (3d/2, 1.5), (-3d/2, 1.5)
    # (d, 3), (-d, 3)
    # This is getting complicated.
    # Let's just use the flower + 4 random-ish points far away.
    for k in range(4):
        ang = np.deg2rad(45.0 + 90.0 * k)
        inner[7 + k] = [3.0 * d * np.cos(ang), 3.0 * d * np.sin(ang), 0.0]
    return inner

def hexagon_packing_11():
    # 1. Generate initial packing
    inner0 = _initial_packing()
    oc0 = np.array([0.0, 0.0])
    ang0 = 0.0
    
    # Check if initial is valid
    s0 = _outer_side_for(inner0, oc0, ang0)
    if _validity(inner0, oc0, ang0, s0):
        best_inner = inner0.copy()
        best_side = s0
    else:
        # Fallback to a very spread out packing
        inner0 = np.array([
            [0, 0, 0], [-5.0, 0, 0], [5.0, 0, 0], [-2.5, 4.3, 0], [2.5, 4.3, 0],
            [-2.5, -4.3, 0], [2.5, -4.3, 0], [-7.5, 4.3, 0], [7.5, 4.3, 0],
            [-7.5, -4.3, 0], [7.5, -4.3, 0],
        ], dtype=float)
        s0 = _outer_side_for(inner0, oc0, ang0)
        best_inner = inner0.copy()
        best_side = s0

    # 2. Optimize
    # Use Nelder-Mead for robustness, or L-BFGS-B.
    # L-BFGS-B is faster if we have gradients, but we don't.
    # Nelder-Mead is derivative-free.
    
    if time.time() - _START_TIME < _TIME_LIMIT - 5.0:
        try:
            x0 = best_inner.ravel()
            # Optimize centers (x,y) only, keep angle 0
            # We will optimize all 33 variables, but the penalty ignores angle.
            # Better to fix angle to 0 in the objective?
            # The objective uses inner which has angles.
            # If we let angles vary, the penalty is wrong.
            # So we must fix angles to 0 in the search space.
            # We will only search x,y.
            xy0 = best_inner[:, :2].ravel()
            
            def obj_xy(xy):
                inner_xy = np.column_stack([xy.reshape(11, 2), np.zeros(11)])
                return _objective(inner_xy, oc0, ang0)
            
            res = minimize(obj_xy, xy0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            xy_opt = res.x.reshape(11, 2)
            inner_opt = np.column_stack([xy_opt, np.zeros(11)])
            side_opt = _outer_side_for(inner_opt, oc0, ang0)
            
            if _validity(inner_opt, oc0, ang0, side_opt) and side_opt < best_side:
                best_inner = inner_opt
                best_side = side_opt
        except Exception:
            pass

    # 3. If time permits, try a second round with a different initial guess
    if time.time() - _START_TIME < _TIME_LIMIT - 10.0:
        try:
            # Perturb the best solution
            inner1 = best_inner + np.random.normal(0, 0.5, best_inner.shape)
            # Re-center
            inner1[:, :2] -= inner1[:, :2].mean(axis=0)
            inner1[:, 2] = 0.0 # Fix angles
            
            xy1 = inner1[:, :2].ravel()
            def obj_xy2(xy):
                inner_xy = np.column_stack([xy.reshape(11, 2), np.zeros(11)])
                return _objective(inner_xy, oc0, ang0)
            
            res2 = minimize(obj_xy2, xy1, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            xy_opt2 = res2.x.reshape(11, 2)
            inner_opt2 = np.column_stack([xy_opt2, np.zeros(11)])
            side_opt2 = _outer_side_for(inner_opt2, oc0, ang0)
            
            if _validity(inner_opt2, oc0, ang0, side_opt2) and side_opt2 < best_side:
                best_inner = inner_opt2
                best_side = side_opt2
        except Exception:
            pass

    return best_inner, np.array([oc0[0], oc0[1], ang0]), float(best_side)

# EVOLVE-BLOCK-END
