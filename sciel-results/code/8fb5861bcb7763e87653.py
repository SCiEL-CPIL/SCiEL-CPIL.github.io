# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

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

def _outer_side_for(inner, outer_center, outer_angle):
    """Smallest side of a regular hexagon (center, angle) that encloses all
    inner vertices. Uses the support function."""
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

# ---------- Rigid-body lattice representation ----------
# 11 hexagons on a triangular lattice: center + 6 in first ring + 4 in second ring
# Lattice basis vectors: b1, b2 at 60 degrees. Shearing transforms the lattice.
# Parameters: shear_x, shear_y (lattice deformation), outer_angle, and small per-hex offsets

_LATTICE_POSITIONS = np.array([
    [0, 0],       # center
    [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],  # first ring
    [2, 0], [1, 1], [-1, 2], [2, -1],  # four of second ring (corners)
])

def _lattice_to_positions(shear_x, shear_y, outer_angle):
    """Convert lattice parameters to (11, 3) hexagon positions.
    shear_x, shear_y deform the triangular lattice slightly.
    All hexagons share the same angle (rigid body on lattice)."""
    d = np.sqrt(3.0)  # minimum center-to-center distance for touching unit hexagons
    # Basis vectors with shear
    b1 = np.array([d * (1.0 + shear_x), d * shear_y])
    b2 = np.array([d * (0.5 - shear_x * 0.5), d * (np.sqrt(3.0)/2.0 + shear_y * 0.5)])
    
    positions = np.zeros((11, 3))
    for i in range(11):
        lx, ly = _LATTICE_POSITIONS[i]
        pos = lx * b1 + ly * b2
        positions[i] = [pos[0], pos[1], outer_angle]
    return positions

def _smooth_max_penalty(inner, outer_center, outer_angle, beta=50.0):
    """Smooth max surrogate for the enclosing side length + separation penalty."""
    a = np.deg2rad(outer_angle)
    th = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(th), np.sin(th)])
    
    # Compute all vertex projections
    all_verts = np.vstack([_hex_vertices(h[0], h[1], 1.0, h[2]) - outer_center for h in inner])
    max_proj = (all_verts @ normals.T).max(axis=0)
    side = (2.0 / np.sqrt(3.0)) * max_proj.max()
    
    # Smooth max over the 6 normal projections for better gradient flow
    logsumexp = np.log(np.sum(np.exp(beta * max_proj))) / beta
    side_smooth = (2.0 / np.sqrt(3.0)) * logsumexp
    
    # Separation penalty: use center distances as proxy
    pen = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            dx = inner[i, 0] - inner[j, 0]
            dy = inner[i, 1] - inner[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = np.sqrt(3.0)
            if dist < min_dist:
                pen += (min_dist - dist) ** 2
    
    return side_smooth + 10.0 * pen

def _objective_lattice(params, outer_center):
    """Objective for lattice-based optimization."""
    shear_x, shear_y, outer_angle = params
    inner = _lattice_to_positions(shear_x, shear_y, outer_angle)
    return _smooth_max_penalty(inner, outer_center, outer_angle)

def _initial_packing():
    """7 hexagons in a hexagonal lattice (center + ring) + 4 in corner gaps."""
    d = np.sqrt(3.0)
    inner = np.zeros((11, 3))
    inner[0] = [0.0, 0.0, 0.0]
    for k in range(6):
        inner[k + 1] = [d * np.cos(np.deg2rad(60.0 * k)), d * np.sin(np.deg2rad(60.0 * k)), 0.0]
    for k in range(4):
        ang = np.deg2rad(30.0 + 90.0 * k)
        inner[7 + k] = [2.0 * d * np.cos(ang), 2.0 * d * np.sin(ang), 0.0]
    return inner

def _try_optimize(inner0, oc, ang, max_iter=50, ftol=1e-8, gtol=1e-6):
    """Attempt a single L-BFGS-B optimization and return improved result if valid."""
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

def _objective(x, outer_center, outer_angle):
    inner = x.reshape(11, 3)
    return _outer_side_for(inner, outer_center, outer_angle) + 10.0 * _penalty(inner)

def _penalty(inner):
    n = len(inner)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = inner[i, 0] - inner[j, 0]
            dy = inner[i, 1] - inner[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = np.sqrt(3.0)
            if dist < min_dist:
                pen += (min_dist - dist) ** 2
    return pen

def _lattice_optimize(oc, initial_angle=0.0, max_iter=300):
    """Optimize lattice parameters using Nelder-Mead with smooth max."""
    x0 = np.array([0.0, 0.0, initial_angle])
    res = minimize(_objective_lattice, x0, args=(oc,),
                   method="Nelder-Mead",
                   options={"maxiter": max_iter, "xatol": 1e-8, "fatol": 1e-10})
    return res.x

def _polish(inner0, oc, ang, max_iter=200):
    """High-precision polish using L-BFGS-B on individual positions."""
    try:
        x0 = inner0.ravel()
        res = minimize(_objective, x0, args=(oc, ang),
                       method="L-BFGS-B", options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-10})
        inner2 = res.x.reshape(11, 3)
        side2 = _outer_side_for(inner2, oc, ang)
        if _validity(inner2, oc, ang, side2):
            return inner2, side2
    except Exception:
        pass
    return None, None

def hexagon_packing_11():
    oc0 = np.array([0.0, 0.0])
    
    # Start with lattice-based optimization
    best_inner = None
    best_side = np.inf
    best_ang = 0.0
    
    # Try multiple initial angles for the outer hexagon
    for trial_ang in [0.0, 30.0, 15.0, -15.0, 45.0, -45.0]:
        if time.time() - _START_TIME > _TIME_LIMIT - 60.0:
            break
        try:
            params = _lattice_optimize(oc0, trial_ang, max_iter=200)
            inner = _lattice_to_positions(params[0], params[1], params[2])
            side = _outer_side_for(inner, oc0, params[2])
            if _validity(inner, oc0, params[2], side):
                if side < best_side:
                    best_inner = inner.copy()
                    best_side = side
                    best_ang = params[2]
        except Exception:
            continue
    
    # Fallback if lattice optimization failed
    if best_inner is None:
        inner0 = _initial_packing()
        s0 = _outer_side_for(inner0, oc0, 0.0)
        if _validity(inner0, oc0, 0.0, s0):
            best_inner = inner0.copy()
            best_side = s0
            best_ang = 0.0
        else:
            inner0 = np.array([
                [0, 0, 0], [-3.0, 0, 0], [3.0, 0, 0], [-1.5, 2.6, 0], [1.5, 2.6, 0],
                [-1.5, -2.6, 0], [1.5, -2.6, 0], [-4.5, 2.6, 0], [4.5, 2.6, 0],
                [-4.5, -2.6, 0], [4.5, -2.6, 0],
            ], dtype=float)
            s0 = _outer_side_for(inner0, oc0, 0.0)
            best_inner = inner0.copy()
            best_side = s0
            best_ang = 0.0
    
    # Polish with individual position optimization
    if time.time() - _START_TIME < _TIME_LIMIT - 30.0:
        inner_pol, side_pol = _polish(best_inner, oc0, best_ang, max_iter=300)
        if inner_pol is not None and side_pol < best_side:
            best_inner = inner_pol
            best_side = side_pol
    
    # Try a few more outer angles with the polished packing
    if time.time() - _START_TIME < _TIME_LIMIT - 20.0:
        for trial_ang in [30.0, 15.0, -15.0, 45.0, -45.0, 60.0]:
            if time.time() - _START_TIME > _TIME_LIMIT - 15.0:
                break
            s_trial = _outer_side_for(best_inner, oc0, trial_ang)
            if s_trial < best_side:
                if _validity(best_inner, oc0, trial_ang, s_trial):
                    best_side = s_trial
                    best_ang = trial_ang
    
    return best_inner, np.array([oc0[0], oc0[1], best_ang]), float(best_side)

# EVOLVE-BLOCK-END
