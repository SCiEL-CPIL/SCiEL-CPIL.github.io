# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 580.0  # safety margin before 630s kill

# Precompute unit hexagon vertices and normals for angle=0
_HEX_TH = np.deg2rad(60.0 * np.arange(6))
_HEX_BASE_VERTS = np.column_stack([np.cos(_HEX_TH), np.sin(_HEX_TH)])  # (6,2)
_HEX_BASE_NORMALS = np.column_stack([np.cos(_HEX_TH + np.deg2rad(30)), np.sin(_HEX_TH + np.deg2rad(30))])  # (6,2)


def _hex_vertices_fast(cx, cy, s, a_deg):
    """Return (6,2) array of vertices for a regular hexagon using precomputed base."""
    a = np.deg2rad(a_deg)
    ca, sa = np.cos(a), np.sin(a)
    rot = np.array([[ca, -sa], [sa, ca]])
    return cx + s * (_HEX_BASE_VERTS @ rot.T)


def _edge_normals_fast(a_deg):
    """Return (6,2) array of outward-pointing unit edge normals for angle a_deg."""
    a = np.deg2rad(a_deg)
    ca, sa = np.cos(a), np.sin(a)
    rot = np.array([[ca, -sa], [sa, ca]])
    return _HEX_BASE_NORMALS @ rot.T


def _min_gap_fast(h1, h2):
    """
    Fast minimum separation between two unit hexagons using SAT with precomputed geometry.
    h1, h2: (x, y, angle_deg) tuples.
    Returns the minimum gap (positive if separated, 0 if touching/overlapping).
    """
    v1 = _hex_vertices_fast(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_vertices_fast(h2[0], h2[1], 1.0, h2[2])
    n1 = _edge_normals_fast(h1[2])
    n2 = _edge_normals_fast(h2[2])
    
    all_norms = np.vstack([n1, n2])  # (12, 2)
    min_gap = float('inf')
    
    for k in range(12):
        n = all_norms[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
        if gap < min_gap:
            min_gap = gap
        if min_gap <= 1e-9:
            break
    return min_gap


def _separation(h1, h2):
    """Return min gap between two unit hexagons given as (x, y, angle_deg)."""
    return _min_gap_fast(h1, h2)


def _check_separations(inner):
    """Check all pairs. Return (min_gap, valid) where valid means all gaps > 1e-6."""
    n = len(inner)
    min_gap = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            gap = _separation(inner[i], inner[j])
            if gap < min_gap:
                min_gap = gap
            if min_gap <= 1e-7:
                break
        if min_gap <= 1e-7:
            break
    valid = min_gap > 1e-6
    return min_gap, valid


def _outer_side_for(inner, outer_center=np.array([0.0, 0.0]), outer_angle=0.0):
    """
    Find minimum outer side length such that all inner hex vertices are inside.
    Outer hexagon is centered at outer_center with given angle.
    """
    normal_angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
    
    # Vectorize: compute all 66 vertices at once
    all_verts = np.zeros((len(inner) * 6, 2))
    for idx, h in enumerate(inner):
        v = _hex_vertices_fast(h[0], h[1], 1.0, h[2])
        all_verts[idx*6:(idx+1)*6] = v - outer_center
    
    proj = all_verts @ normals.T  # (66, 6)
    max_proj_per_normal = proj.max(axis=0)  # (6,)
    required_s = (2.0 / np.sqrt(3.0)) * max_proj_per_normal
    return float(required_s.max())


def _soft_min_penalty(gaps, threshold=1e-6, beta=50.0):
    """
    Smooth surrogate for max(0, threshold - min(gaps)) using log-sum-exp.
    Returns a differentiable penalty that is ~0 when all gaps > threshold.
    """
    violations = threshold - gaps  # positive when gap < threshold
    v_max = np.max(violations)
    if v_max < 1e-10:
        return 0.0
    exps = np.exp(beta * (violations - v_max))
    soft_max = v_max + np.log(np.sum(exps)) / beta
    return max(0.0, soft_max)


def _objective(x, n=11):
    """
    Objective: minimize outer side length + smooth penalty for overlaps.
    x: flat array of 11*3 values
    """
    inner = x.reshape(n, 3)
    
    # Compute all pairwise gaps
    gaps = []
    for i in range(n):
        for j in range(i + 1, n):
            gap = _separation(inner[i], inner[j])
            gaps.append(gap)
    gaps = np.array(gaps)
    
    # Smooth penalty for any gap below threshold
    pen = _soft_min_penalty(gaps)
    pen *= 100.0  # scale penalty
    
    side = _outer_side_for(inner)
    return side + pen


def _initial_packing():
    """
    Analytical initial packing: 1 center + 6 ring + 4 in the 'corner' gaps.
    Unit hexagons in a hexagonal lattice.
    """
    d = np.sqrt(3.0)  # center-to-center distance for touching hexagons
    angles = np.deg2rad(60.0 * np.arange(6))
    dirs = np.column_stack([d * np.cos(angles), d * np.sin(angles)])
    
    inner = np.zeros((11, 3))
    inner[0] = [0.0, 0.0, 0.0]
    for k in range(6):
        inner[k + 1] = [dirs[k, 0], dirs[k, 1], 0.0]
    
    r2 = 2.0 * d
    ring2_angles = np.deg2rad([30.0, 90.0, 150.0, 270.0])
    for k in range(4):
        idx = 7 + k
        inner[idx, 0] = r2 * np.cos(ring2_angles[k])
        inner[idx, 1] = r2 * np.sin(ring2_angles[k])
        inner[idx, 2] = 0.0
    
    return inner


def _try_optimize(s0, best_state, n, max_iter=3000, tol=1e-6):
    """Try to optimize from start point s0, update best_state if improved."""
    try:
        res = minimize(_objective, s0.flatten(), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': tol, 'fatol': 1e-8,
                                'adaptive': True})
        x = res.x.reshape(n, 3)
        
        min_gap, valid = _check_separations(x)
        if valid:
            side = _outer_side_for(x)
            if side < best_state['side']:
                best_state['side'] = side
                best_state['x'] = x.copy()
    except Exception:
        pass
    return best_state


def _polish(x, n, max_iter=5000):
    """
    High-precision polish: minimize side length with very tight tolerances.
    Only accepts solutions that are valid (all gaps > 1e-6).
    """
    def _polish_obj(xp):
        inner = xp.reshape(n, 3)
        # Hard constraint: all gaps must be > 1e-6
        for i in range(n):
            for j in range(i + 1, n):
                gap = _separation(inner[i], inner[j])
                if gap <= 1e-6:
                    return 1e6 + (1e-6 - gap) * 1000.0
        return _outer_side_for(inner)
    
    try:
        res = minimize(_polish_obj, x.flatten(), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-10,
                                'adaptive': True})
        x_new = res.x.reshape(n, 3)
        min_gap, valid = _check_separations(x_new)
        if valid:
            side = _outer_side_for(x_new)
            return x_new, side
    except Exception:
        pass
    return x, _outer_side_for(x)


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    n = 11
    inner0 = _initial_packing()
    
    min_gap, valid = _check_separations(inner0)
    if not valid:
        inner0 = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.165, 0],
            [1.25, 2.165, 0],
            [-1.25, -2.165, 0],
            [1.25, -2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, -2.165, 0],
            [3.75, -2.165, 0],
        ])
        min_gap, valid = _check_separations(inner0)
    
    best_state = {'x': inner0.copy(), 'side': _outer_side_for(inner0)}
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points with angle perturbations
    starts = [inner0.copy()]
    for i in range(15):
        jitter = rng.normal(0, 0.3, size=(n, 2))
        angle_jitter = rng.uniform(-45, 45, size=n)
        s = inner0.copy()
        s[:, :2] += jitter
        s[:, 2] += angle_jitter
        starts.append(s)
    
    # Try each start with time management
    for s0 in starts:
        if time.time() - _START_TIME > _TIME_LIMIT - 120:
            break
        best_state = _try_optimize(s0, best_state, n, max_iter=2000)
    
    # Additional refinement: try to optimize angles of best solution
    if time.time() - _START_TIME < _TIME_LIMIT - 90:
        x = best_state['x'].copy()
        for _ in range(8):
            if time.time() - _START_TIME > _TIME_LIMIT - 90:
                break
            x_perturbed = x.copy()
            x_perturbed[:, :2] += rng.normal(0, 0.2, size=(n, 2))
            x_perturbed[:, 2] += rng.normal(0, 10.0, size=n)
            best_state = _try_optimize(x_perturbed, best_state, n, max_iter=3000)
    
    # Final high-precision polish pass
    if time.time() - _START_TIME < _TIME_LIMIT - 30:
        x_polished, side_polished = _polish(best_state['x'], n, max_iter=5000)
        if side_polished < best_state['side']:
            best_state['x'] = x_polished
            best_state['side'] = side_polished
    
    # Final verification
    min_gap, valid = _check_separations(best_state['x'])
    if not valid:
        best_state['x'] = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.165, 0],
            [1.25, 2.165, 0],
            [-1.25, -2.165, 0],
            [1.25, -2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, -2.165, 0],
            [3.75, -2.165, 0],
        ])
        best_state['side'] = 8.0
    
    side = best_state['side']
    outer = np.array([0.0, 0.0, 0.0])
    return best_state['x'], outer, float(side)


# EVOLVE-BLOCK-END
