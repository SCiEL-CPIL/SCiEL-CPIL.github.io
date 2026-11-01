"""
Hexagon packing: 11 unit hexagons in minimal enclosing regular hexagon.
Strategy: compact initial cluster + full continuous optimization over all
positions, outer rotation, and translation with smooth objective.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def _hex_verts(cx, cy, side, angle_rad):
    """6 vertices of a regular hexagon (pointy-top at angle 0)."""
    angs = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angs), cy + side * np.sin(angs)])

def _exact_side(positions, angles, phi_rad, center):
    """Exact min enclosing hex side for given outer rotation phi and center."""
    all_v = np.vstack([_hex_verts(p[0], p[1], 1.0, a) for p, a in zip(positions, angles)])
    all_v -= np.array(center)
    # Rotate into outer hex frame
    c, s = math.cos(phi_rad), math.sin(phi_rad)
    R = np.array([[c, -s], [s, c]])
    all_v = all_v @ R.T
    # Face normals for flat-top hex: (2k+1)*pi/6
    thetas = np.array([(2 * k + 1) * np.pi / 6 for k in range(6)])
    max_proj = 0.0
    for t in thetas:
        proj = all_v[:, 0] * math.cos(t) + all_v[:, 1] * math.sin(t)
        mp = np.max(proj)
        if mp > max_proj:
            max_proj = mp
    return 2.0 * max_proj / SQRT3

def _smooth_side(positions, angles, x, beta=30.0):
    """
    Smooth objective. x = [phi, tx, ty, x0,y0,...,x10,y10]
    Outer hex centered at origin. Positions are base + (tx,ty).
    """
    phi = x[0]
    tx, ty = x[1], x[2]
    pos_flat = x[3:]  # 22 values: x0,y0,...,x10,y10
    positions = [(pos_flat[2*i] + tx, pos_flat[2*i+1] + ty) for i in range(N)]
    
    all_v = np.vstack([_hex_verts(p[0], p[1], 1.0, a) for p, a in zip(positions, angles)])
    # Rotate into outer frame
    c, s = math.cos(phi), math.sin(phi)
    R = np.array([[c, -s], [s, c]])
    all_v = all_v @ R.T
    
    thetas = np.array([(2 * k + 1) * np.pi / 6 for k in range(6)])
    projs = np.zeros((6, len(all_v)))
    for k, t in enumerate(thetas):
        projs[k] = all_v[:, 0] * math.cos(t) + all_v[:, 1] * math.sin(t)
    
    # Smooth max over vertices per direction
    m = np.max(projs, axis=1)
    lse_per_dir = m + (1.0 / beta) * np.log(np.sum(np.exp(projs - m[:, None]), axis=1))
    
    # Smooth max over directions
    m2 = np.max(lse_per_dir)
    lse_final = m2 + (1.0 / beta) * math.log(np.sum(np.exp((lse_per_dir - m2) * beta)))
    return 2.0 * lse_final / SQRT3

def _sat_gap(v1, v2, eps=1e-9):
    """Min gap between two convex polygons. Negative = overlap."""
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    min_gap = float('inf')
    for e in list(edges1) + list(edges2):
        nv = np.array([-e[1], e[0]])
        n = np.linalg.norm(nv)
        if n < 1e-12:
            continue
        nv /= n
        p1 = v1 @ nv
        p2 = v2 @ nv
        gap = min(np.max(p1) - np.min(p2), np.max(p2) - np.min(p1))
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _check_disjoint(positions, angles, tol=1e-7):
    n = len(positions)
    verts = [_hex_verts(p[0], p[1], 1.0, a) for p, a in zip(positions, angles)]
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_gap(verts[i], verts[j]) < -tol:
                return False
    return True

def _overlap_penalty(positions, angles, mu=50.0):
    """Penalty for overlapping pairs."""
    verts = [_hex_verts(p[0], p[1], 1.0, a) for p, a in zip(positions, angles)]
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            gap = _sat_gap(verts[i], verts[j])
            if gap < 0:
                pen += mu * gap * gap
    return pen

def _initial_cluster():
    """Compact hexagonal cluster: center + 6 ring + 4 more."""
    # Axial coords for tight packing
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-2, 1), (-1, -2)
    ]
    positions = []
    for q, r in axial:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        positions.append((x, y))
    angles = [0.0] * N
    return positions, angles

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    positions, angles = _initial_cluster()
    
    # Centroid
    cx0 = np.mean([p[0] for p in positions])
    cy0 = np.mean([p[1] for p in positions])
    
    # Initial x: [phi, tx, ty, x0,y0,...,x10,y10]
    # tx, ty shift so centroid is at origin
    tx0 = -cx0
    ty0 = -cy0
    pos_flat = np.array([v for p in positions for v in p])
    x0 = np.concatenate([[0.0, tx0, ty0], pos_flat])
    
    best_side = float('inf')
    best_x = x0.copy()
    
    # Coarse search over phi first
    for phi_deg in np.arange(0, 60, 5.0):
        if time.time() > deadline - 80.0:
            break
        phi = math.radians(phi_deg)
        x_trial = x0.copy()
        x_trial[0] = phi
        val = _smooth_side(positions, angles, x_trial, beta=30.0)
        if val < best_side:
            best_side = val
            best_x = x_trial.copy()
    
    # Nelder-Mead refinement
    def obj(x):
        if time.time() > deadline - 5.0:
            return 1e10
        phi = x[0]
        tx, ty = x[1], x[2]
        pf = x[3:]
        pos = [(pf[2*i] + tx, pf[2*i+1] + ty) for i in range(N)]
        val = _smooth_side(pos, angles, np.array([0.0, 0.0, 0.0]) + np.array([phi, 0, 0])[:1] * 0 + x[:3], beta=30.0)
        # Actually recompute properly
        val = _smooth_side(pos, angles, x, beta=30.0)
        pen = _overlap_penalty(pos, angles, mu=100.0)
        return val + pen
    
    try:
        res = minimize(obj, best_x, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True, 'maxfev': 5000})
        if res.fun < best_side + 1.0:
            # Check validity
            x_sol = res.x
            phi = x_sol[0]
            tx, ty = x_sol[1], x_sol[2]
            pf = x_sol[3:]
            pos = [(pf[2*i] + tx, pf[2*i+1] + ty) for i in range(N)]
            if _check_disjoint(pos, angles):
                exact = _exact_side(pos, angles, phi, (0.0, 0.0))
                if exact < best_side:
                    best_side = exact
                    best_x = x_sol.copy()
    except Exception:
        pass
    
    # Fine polish: optimize phi and translation only
    if best_x is not None:
        phi0 = best_x[0]
        tx0 = best_x[1]
        ty0 = best_x[2]
        pf = best_x[3:]
        pos_base = [(pf[2*i], pf[2*i+1]) for i in range(N)]
        
        for phi_deg in np.arange(phi0 * 180 / math.pi - 2, phi0 * 180 / math.pi + 2.01, 0.02):
            if time.time() > deadline - 20.0:
                break
            phi = math.radians(phi_deg)
            for dx in np.arange(-0.1, 0.101, 0.02):
                for dy in np.arange(-0.1, 0.101, 0.02):
                    if time.time() > deadline - 10.0:
                        break
                    pos = [(p[0] + tx0 + dx, p[1] + ty0 + dy) for p in pos_base]
                    side = _exact_side(pos, angles, phi, (0.0, 0.0))
                    if side < best_side:
                        best_side = side
                        best_x = np.array([phi, tx0 + dx, ty0 + dy])
                        best_x = np.concatenate([best_x, pf])
    
    # Final validation
    x_final = best_x
    phi = x_final[0]
    tx, ty = x_final[1], x_final[2]
    pf = x_final[3:]
    positions_final = [(pf[2*i] + tx, pf[2*i+1] + ty) for i in range(N)]
    
    if not _check_disjoint(positions_final, angles):
        # Fallback to initial cluster
        positions_final, angles = _initial_cluster()
        cx = np.mean([p[0] for p in positions_final])
        cy = np.mean([p[1] for p in positions_final])
        positions_final = [(p[0] - cx, p[1] - cy) for p in positions_final]
        phi = 0.0
        best_side = _exact_side(positions_final, angles, phi, (0.0, 0.0))
    
    best_side *= (1.0 + 1e-9)
    inner = [[positions_final[i][0], positions_final[i][1], angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, math.degrees(phi)
