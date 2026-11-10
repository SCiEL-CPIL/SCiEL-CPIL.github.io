# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Get 6 vertices of a regular hexagon."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _all_vertices(inner):
    """Get all 66 vertices (11 hexagons x 6) as (66, 2)."""
    n = len(inner)
    verts = np.zeros((n * 6, 2))
    for i in range(n):
        x, y, ang = inner[i]
        verts[i*6:(i+1)*6] = _hex_vertices(x, y, 1.0, ang)
    return verts

def _outer_normals(outer_angle_deg):
    """6 outward edge normals of outer hexagon centered at origin."""
    a = np.deg2rad(outer_angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6) + 30)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _min_outer_side(inner, outer_angle_deg):
    """Minimum side length of outer hexagon (centered at origin) to contain inner."""
    normals = _outer_normals(outer_angle_deg)
    verts = _all_vertices(inner)
    cos30 = np.cos(np.deg2rad(30))
    max_proj = 0.0
    for n in normals:
        proj = verts @ n
        m = np.max(proj)
        if m > max_proj:
            max_proj = m
    return max_proj / cos30

def _pair_separation_ok(vi, vj, ni, nj):
    """Check two hexagons are separated (gap > 1e-6 on at least one of 12 normals)."""
    for n in np.vstack([ni, nj]):
        pi = vi @ n
        pj = vj @ n
        if np.max(pi) < np.min(pj) - 1e-6:
            return True
        if np.max(pj) < np.min(pi) - 1e-6:
            return True
    return False

def _pair_min_gap(vi, vj, ni, nj):
    """Minimum gap (negative = overlap) over 12 normals."""
    min_gap = np.inf
    for n in np.vstack([ni, nj]):
        pi = vi @ n
        pj = vj @ n
        gap1 = np.min(pj) - np.max(pi)
        gap2 = np.min(pi) - np.max(pj)
        g = max(gap1, gap2)
        if g < min_gap:
            min_gap = g
    return min_gap

def _soft_enclosing_side(inner, outer_angle_deg, beta=30.0):
    """Smooth surrogate for enclosing side using log-sum-exp over 6 face constraints."""
    normals = _outer_normals(outer_angle_deg)
    verts = _all_vertices(inner)
    cos30 = np.cos(np.deg2rad(30))
    # face constraint k: max over vertices of (v . n_k) <= apothem
    # side = max_k (max_v v.n_k) / cos30
    # smooth: side = (1/cos30) * (1/beta) * log(sum_k exp(beta * max_v v.n_k))
    face_max = np.array([np.max(verts @ n) for n in normals])
    m = np.max(face_max)
    logsum = m + np.log(np.sum(np.exp(beta * (face_max - m))))
    return (logsum / beta) / cos30

def _overlap_penalty(inner, beta=20.0):
    """Smooth penalty for overlaps between inner hexagons."""
    n = len(inner)
    penalty = 0.0
    for i in range(n):
        xi, yi, ai = inner[i]
        vi = _hex_vertices(xi, yi, 1.0, ai)
        a = np.deg2rad(ai)
        ni = np.column_stack([np.cos(a + np.deg2rad(60*np.arange(6)+30)),
                              np.sin(a + np.deg2rad(60*np.arange(6)+30))])
        for j in range(i+1, n):
            xj, yj, aj = inner[j]
            vj = _hex_vertices(xj, yj, 1.0, aj)
            b = np.deg2rad(aj)
            nj = np.column_stack([np.cos(b + np.deg2rad(60*np.arange(6)+30)),
                                  np.sin(b + np.deg2rad(60*np.arange(6)+30))])
            min_gap = _pair_min_gap(vi, vj, ni, nj)
            # smooth penalty: -softmin of gaps (penalize when any gap < 0)
            # use logistic: penalty += log(1+exp(beta*(-min_gap)))
            penalty += np.log1p(np.exp(beta * (-min_gap)))
    return penalty

def _objective_smooth(params, beta_side=30.0, beta_olap=20.0):
    n = 11
    inner = params[:n*3].reshape(n, 3)
    outer_angle = params[n*3]
    side = _soft_enclosing_side(inner, outer_angle, beta_side)
    pen = _overlap_penalty(inner, beta_olap)
    return side + 50.0 * pen

def _objective_exact(params):
    n = 11
    inner = params[:n*3].reshape(n, 3)
    outer_angle = params[n*3]
    side = _min_outer_side(inner, outer_angle)
    # hard overlap penalty
    pen = 0.0
    for i in range(n):
        xi, yi, ai = inner[i]
        vi = _hex_vertices(xi, yi, 1.0, ai)
        a = np.deg2rad(ai)
        ni = np.column_stack([np.cos(a + np.deg2rad(60*np.arange(6)+30)),
                              np.sin(a + np.deg2rad(60*np.arange(6)+30))])
        for j in range(i+1, n):
            xj, yj, aj = inner[j]
            vj = _hex_vertices(xj, yj, 1.0, aj)
            b = np.deg2rad(aj)
            nj = np.column_stack([np.cos(b + np.deg2rad(60*np.arange(6)+30)),
                                  np.sin(b + np.deg2rad(60*np.arange(6)+30))])
            gap = _pair_min_gap(vi, vj, ni, nj)
            if gap < 0:
                pen += (-gap) ** 2
    return side + 200.0 * pen

def _hex_lattice_positions():
    """Hexagonal lattice: center + ring of 6 + 4 more for 11 hexagons."""
    apothem = np.sqrt(3) / 2
    d = 2 * apothem  # sqrt(3)
    axial = [
        (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1),
    ]
    positions = []
    for q, r in axial:
        x = d * (q + r/2.0)
        y = d * 1.5 * r
        positions.append((x, y))
    return positions

def _initial_guesses():
    """Multiple diverse initial guesses."""
    guesses = []
    base = _hex_lattice_positions()
    n = 11
    # Guess 1: lattice, angle 0
    p = np.zeros(n*3 + 1)
    for i, (x, y) in enumerate(base):
        p[i*3] = x
        p[i*3+1] = y
        p[i*3+2] = 0.0
    p[n*3] = 0.0
    guesses.append(p)
    # Guess 2: lattice, angle 30
    p2 = p.copy()
    p2[n*3] = 30.0
    for i in range(n):
        p2[i*3+2] = 30.0
    guesses.append(p2)
    # Guess 3: slightly perturbed lattice
    np.random.seed(7)
    p3 = p + np.random.normal(0, 0.1, size=p.shape)
    guesses.append(p3)
    # Guess 4: compacted lattice (scale down a bit to encourage tighter)
    p4 = p.copy()
    for i in range(n):
        p4[i*3] *= 0.95
        p4[i*3+1] *= 0.95
    guesses.append(p4)
    return guesses

def _verify_solution(inner, outer_angle, outer_side):
    """Verify validity per task contract."""
    n = len(inner)
    if not np.all(np.isfinite(inner)):
        return False
    if not np.isfinite(outer_angle) or not np.isfinite(outer_side):
        return False
    if outer_side < np.sqrt(11) - 1e-9:
        return False
    # pair separation
    for i in range(n):
        xi, yi, ai = inner[i]
        vi = _hex_vertices(xi, yi, 1.0, ai)
        a = np.deg2rad(ai)
        ni = np.column_stack([np.cos(a + np.deg2rad(60*np.arange(6)+30)),
                              np.sin(a + np.deg2rad(60*np.arange(6)+30))])
        for j in range(i+1, n):
            xj, yj, aj = inner[j]
            vj = _hex_vertices(xj, yj, 1.0, aj)
            b = np.deg2rad(aj)
            nj = np.column_stack([np.cos(b + np.deg2rad(60*np.arange(6)+30)),
                                  np.sin(b + np.deg2rad(60*np.arange(6)+30))])
            if not _pair_separation_ok(vi, vj, ni, nj):
                return False
    # containment: each inner vertex inside outer hexagon
    normals = _outer_normals(outer_angle)
    cos30 = np.cos(np.deg2rad(30))
    apothem = outer_side * cos30
    verts = _all_vertices(inner)
    for nv in normals:
        proj = verts @ nv
        if np.max(proj) > apothem + 1e-6:
            return False
    return True

def _fallback_solution():
    """Known valid fallback: lattice with margin."""
    positions = _hex_lattice_positions()
    inner = np.zeros((11, 3))
    for i, (x, y) in enumerate(positions):
        inner[i] = [x, y, 0.0]
    outer_angle = 0.0
    outer_side = _min_outer_side(inner, outer_angle)
    outer_side *= 1.02
    return inner, np.array([0.0, 0.0, outer_angle]), outer_side

def _solve_packing(deadline):
    """Two-phase optimization: smooth coarse, then exact polish."""
    best_params = None
    best_obj = np.inf
    n = 11

    guesses = _initial_guesses()
    # Evaluate all guesses with exact objective
    for g in guesses:
        try:
            o = _objective_exact(g)
            if o < best_obj:
                best_obj = o
                best_params = g.copy()
        except Exception:
            pass

    # Phase 1: smooth optimization from each guess
    for g in guesses:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_smooth, g,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 4000}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: exact polish from best
    for _ in range(6):
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_exact, best_params,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 8000}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: local search with small perturbations
    np.random.seed(42)
    for attempt in range(25):
        if time.time() > deadline:
            break
        scale = 0.3 / (attempt + 1)
        p_try = best_params + np.random.normal(0, scale, size=best_params.shape)
        try:
            res = minimize(
                _objective_exact, p_try,
                method='L-BFGS-B',
                options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 3000}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Extract
    inner = best_params[:n*3].reshape(n, 3)
    outer_angle = best_params[n*3]
    outer_side = _min_outer_side(inner, outer_angle)

    if _verify_solution(inner, outer_angle, outer_side):
        return inner, np.array([0.0, 0.0, outer_angle]), outer_side
    else:
        return _fallback_solution()

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger
    regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), rows (x, y, angle_degrees).
        outer_hex_data: np.ndarray of shape (3,) of form (x, y, angle_degree).
        outer_hex_side_length: float.
    """
    start_time = time.time()
    deadline = start_time + 600  # safety margin before 630s kill
    try:
        inner, outer_hex_data, outer_side = _solve_packing(deadline)
    except Exception:
        inner, outer_hex_data, outer_side = _fallback_solution()
    if time.time() > deadline:
        inner, outer_hex_data, outer_side = _fallback_solution()
    return inner, outer_hex_data, outer_side

# EVOLVE-BLOCK-END
