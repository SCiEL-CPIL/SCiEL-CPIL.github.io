# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _hex_vertices_batch(inner_data, outer_data=None, side=1.0, outer_side=None):
    """Compute vertices for all inner hexagons at once.
    inner_data: (11, 3) array of (x, y, angle_deg)
    Returns (11, 6, 2) array of vertices.
    """
    x = inner_data[:, 0]
    y = inner_data[:, 1]
    a = np.deg2rad(inner_data[:, 2])
    ang = a[:, np.newaxis] + np.deg2rad(60 * np.arange(6))[np.newaxis, :]
    verts = np.stack([
        x[:, np.newaxis] + 1.0 * np.cos(ang),
        y[:, np.newaxis] + 1.0 * np.sin(ang)
    ], axis=2)
    return verts


def _hex_normals(a_deg):
    a = np.deg2rad(a_deg)
    base = a + np.deg2rad(60 * np.arange(6))
    normals = base + np.deg2rad(30)
    dirs = np.stack([np.cos(normals), np.sin(normals)], axis=1)
    return np.vstack([dirs, -dirs])


def _separation_ok(v1, n1, v2, n2):
    for d in n1:
        p1 = v1 @ d
        p2 = v2 @ d
        gap = min(p1) - max(p2)
        if gap > 1e-6:
            return True
        gap = min(p2) - max(p1)
        if gap > 1e-6:
            return True
    for d in n2:
        p1 = v1 @ d
        p2 = v2 @ d
        gap = min(p1) - max(p2)
        if gap > 1e-6:
            return True
        gap = min(p2) - max(p1)
        if gap > 1e-6:
            return True
    return False


def _hexagons_separated(inner_data):
    verts = []
    norms = []
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        n = _hex_normals(a)
        verts.append(v)
        norms.append(n)
    for i in range(len(inner_data)):
        for j in range(i + 1, len(inner_data)):
            if not _separation_ok(verts[i], norms[i], verts[j], norms[j]):
                return False
    return True


def _contains_in_hex(inner_data, outer_data, side):
    ox, oy, oa = outer_data
    outer_v = _hex_vertices(ox, oy, side, oa)
    tol = -1e-6
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            for e in range(6):
                p1 = outer_v[e]
                p2 = outer_v[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < tol:
                    return False
    return True


def _valid(inner_data, outer_data, side):
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11) - 1e-12:
        return False
    if not _hexagons_separated(inner_data):
        return False
    if not _contains_in_hex(inner_data, outer_data, side):
        return False
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for e in range(6):
            p1 = v[e]
            p2 = v[(e + 1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
    return True


def _compute_min_side(inner, outer_angle_deg):
    """Compute minimum outer hex side to contain all inner hexagons."""
    a_rad = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        ang = a_rad + np.deg2rad(30 + 60 * k)
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)
    
    verts = _hex_vertices_batch(inner)
    # verts shape: (11, 6, 2)
    # Project all vertices onto all 6 normals
    # projs[i, j, k] = vertex j of hex i projected onto normal k
    # Using 'ijx,kx->ijk' to handle the 3D verts array correctly
    projs = np.abs(np.einsum('ijx,kx->ijk', verts, normals))
    max_proj = np.max(projs)
    
    return max_proj / (np.sqrt(3) / 2)


def _general_objective(params, outer_angle_deg=0.0):
    """
    Objective for general packing.
    params: (33,) array [x0, y0, a0, x1, y1, a1, ...]
    Returns: side_length + penalty_for_overlaps
    """
    inner_data = params.reshape(11, 3)
    # Normalize angles to 0-60 for symmetry? No, keep general.
    
    # 1. Compute side length
    side = _compute_min_side(inner_data, outer_angle_deg)
    
    # 2. Compute overlap penalty
    penalty = 0.0
    verts = _hex_vertices_batch(inner_data)
    norms_list = [_hex_normals(inner_data[i, 2]) for i in range(11)]
    
    for i in range(11):
        for j in range(i + 1, 11):
            v1 = verts[i]
            v2 = verts[j]
            n1 = norms_list[i]
            n2 = norms_list[j]
            
            min_gap = float('inf')
            # Check all 12 normals
            for d in n1:
                p1 = v1 @ d
                p2 = v2 @ d
                gap1 = min(p1) - max(p2)
                gap2 = min(p2) - max(p1)
                # We want gap > 1e-6. If gap <= 1e-6, it's a violation.
                # Violation amount = 1e-6 - gap (if gap < 1e-6)
                viol1 = 1e-6 - gap1
                viol2 = 1e-6 - gap2
                if viol1 > 0:
                    min_gap = min(min_gap, viol1)
                if viol2 > 0:
                    min_gap = min(min_gap, viol2)
            
            for d in n2:
                p1 = v1 @ d
                p2 = v2 @ d
                gap1 = min(p1) - max(p2)
                gap2 = min(p2) - max(p1)
                viol1 = 1e-6 - gap1
                viol2 = 1e-6 - gap2
                if viol1 > 0:
                    min_gap = min(min_gap, viol1)
                if viol2 > 0:
                    min_gap = min(min_gap, viol2)
            
            if min_gap < float('inf'):
                # min_gap is the amount of violation (positive if overlapping/too close)
                penalty += min_gap ** 2
    
    return side + 100.0 * penalty


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, outer_hex_side_length).
    """
    t0 = time.time()
    deadline = t0 + 615.0
    
    best_side = float('inf')
    best_inner = None
    best_outer = None
    best_params = None
    
    # Initial guesses: Symmetric flower + random perturbations
    # 1 center, 6 ring, 4 outer
    base_symmetric = np.array([
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [-2.5, 0.0, 0.0],
        [1.25, 2.165, 0.0],
        [-1.25, 2.165, 0.0],
        [1.25, -2.165, 0.0],
        [-1.25, -2.165, 0.0],
        [3.75, 2.165, 0.0],
        [-3.75, 2.165, 0.0],
        [3.75, -2.165, 0.0],
        [-3.75, -2.165, 0.0]
    ])
    
    initial_guesses = [base_symmetric.flatten().copy()]
    
    # Add some random perturbations
    np.random.seed(42)
    for _ in range(5):
        perturbed = base_symmetric + np.random.normal(0, 0.5, base_symmetric.shape)
        # Keep angles small
        perturbed[:, 2] = np.random.uniform(-10, 10, 11)
        initial_guesses.append(perturbed.flatten())
    
    for guess in initial_guesses:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(
                _general_objective,
                guess,
                args=(0.0,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )
            # Evaluate the final result properly
            final_params = res.x
            final_inner = final_params.reshape(11, 3)
            final_side = _compute_min_side(final_inner, 0.0)
            
            # Check validity
            final_outer = np.array([0.0, 0.0, 0.0])
            if _valid(final_inner, final_outer, final_side):
                if final_side < best_side:
                    best_side = final_side
                    best_inner = final_inner
                    best_outer = final_outer
                    best_params = final_params
        except Exception:
            continue
    
    # If we found a good symmetric one, try to polish it with general optimization
    if best_params is not None and time.time() < deadline - 30:
        try:
            res2 = minimize(
                _general_objective,
                best_params,
                args=(0.0,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            final_params = res2.x
            final_inner = final_params.reshape(11, 3)
            final_side = _compute_min_side(final_inner, 0.0)
            final_outer = np.array([0.0, 0.0, 0.0])
            if _valid(final_inner, final_outer, final_side):
                if final_side < best_side:
                    best_side = final_side
                    best_inner = final_inner
                    best_outer = final_outer
                    best_params = final_params
        except Exception:
            pass

    if best_inner is None:
        # Fallback: known valid arrangement
        inner = base_symmetric
        outer = np.array([0, 0, 0])
        side = _compute_min_side(inner, 0.0)
        if not _valid(inner, outer, side):
            side = 8.0
            if not _valid(inner, outer, side):
                side = 9.0
        
        return inner, outer, side
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END
