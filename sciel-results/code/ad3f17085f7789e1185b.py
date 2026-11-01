import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def _time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN

def _hex_vertices(cx, cy, s, ang_rad):
    """Return 6 vertices of a regular hexagon (side s) centered at (cx,cy) rotated by ang_rad."""
    verts = np.empty((6, 2))
    for k in range(6):
        a = ang_rad + k * math.pi / 3.0
        verts[k, 0] = cx + s * math.cos(a)
        verts[k, 1] = cy + s * math.sin(a)
    return verts

def _container_normals(beta):
    """6 outward normals of the outer hexagon given rotation beta."""
    normals = np.empty((6, 2))
    for k in range(6):
        a = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals[k, 0] = math.cos(a)
        normals[k, 1] = math.sin(a)
    return normals

def _support_function(pts, normals):
    """Max of |dot(p-c, n)| over all normals and points (c=origin assumed)."""
    # pts: (n,2), normals: (6,2)
    # projections: for each normal, max over points of (p·n) and -min over points of (p·n)
    projs = pts @ normals.T  # (n, 6)
    max_pos = np.max(projs, axis=0)
    max_neg = -np.min(projs, axis=0)
    return float(np.max(np.maximum(max_pos, max_neg)))

def _required_side(pts, beta):
    """Exact minimal outer side length for point set pts (centered at origin) at rotation beta."""
    normals = _container_normals(beta)
    D = _support_function(pts, normals)
    return (D + APOTHEM) / APOTHEM

def _required_side_smooth(pts, beta, temperature=0.05):
    """Smooth surrogate for required side length via log-sum-exp."""
    normals = _container_normals(beta)
    projs = pts @ normals.T  # (n, 6)
    all_projs = np.concatenate([projs, -projs], axis=1)  # (n, 12)
    all_projs = all_projs.flatten()
    max_val = np.max(all_projs)
    if max_val > 100:
        return float('inf')
    shifted = (all_projs - max_val) / temperature
    log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
    D = temperature * log_sum
    return (D + APOTHEM) / APOTHEM

def _sat_overlap(v1, v2, eps=1e-12):
    """Check if two hexagons (vertex arrays) overlap using SAT. Returns overlap distance (0 if disjoint)."""
    # v1, v2: (6,2)
    overlap = 0.0
    # Normals for both hexagons (3 unique axes each)
    for v in (v1, v2):
        for i in range(3):
            edge = v[(i + 1) % 6] - v[i]
            # normal to edge
            nx = -edge[1]
            ny = edge[0]
            nlen = math.hypot(nx, ny)
            if nlen < 1e-15:
                continue
            nx /= nlen
            ny /= nlen
            # project both
            p1 = v1 @ np.array([nx, ny])
            p2 = v2 @ np.array([nx, ny])
            min1, max1 = np.min(p1), np.max(p1)
            min2, max2 = np.min(p2), np.max(p2)
            gap = min(max1 - max2, max1 - min2)
            if gap < -eps:
                overlap = max(overlap, -gap)
    return overlap

def _check_disjoint(centers, angles, s=1.0, eps=1e-10):
    """Check all pairs of hexagons are disjoint. Returns max overlap (0 if all disjoint)."""
    n = len(centers)
    max_ov = 0.0
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(centers[i][0], centers[i][1], s, angles[i]))
    for i in range(n):
        for j in range(i + 1, n):
            ov = _sat_overlap(verts_list[i], verts_list[j], eps)
            if ov > max_ov:
                max_ov = ov
    return max_ov

def _overlap_penalty(centers, angles, s=1.0, temperature=0.05):
    """Smooth penalty for overlaps."""
    n = len(centers)
    total = 0.0
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(centers[i][0], centers[i][1], s, angles[i]))
    for i in range(n):
        for j in range(i + 1, n):
            ov = _sat_overlap(verts_list[i], verts_list[j], 0.0)
            if ov > 0:
                # smooth max(ov, 0)
                total += ov
    return total

def _axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def _centroid(pts):
    return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

def _build_candidates():
    """Build diverse candidate configurations."""
    candidates = []

    # Standard hexagonal packing patterns
    patterns = [
        # Compact hexagonal cluster
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        # Cross pattern
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        # 3x3 block
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        # 4x3 block
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        # Elongated
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        # Symmetric 7+4
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        # 5x2 strip
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        # Asymmetric
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
        # 3x4
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        # Diamond
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)],
        # Compact 2
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(0,-1)],
        # Wide
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4),(2,3)],
        # Star
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        # Tight cluster
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(1,-1)],
        # 3x3 + extras
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        # 4x3 variant
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3)],
        # Compact 3
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        # 3x4 variant
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        # 4x3 variant 2
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
        # Compact 4
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        # 3x3 + corners
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        # Elongated 2
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        # 2x5
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        # Compact 5
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(1,2),(-1,3)],
        # 3x3 + top
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        # Wide 2
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(0,3),(1,3)],
    ]

    for pat in patterns:
        if len(pat) == N:
            candidates.append(_axial_to_cartesian(pat))

    # Add some perturbed variants
    base = _axial_to_cartesian([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    for dx in [-0.1, 0.1]:
        for dy in [-0.1, 0.1]:
            perturbed = base.copy()
            perturbed[7, 0] += dx
            perturbed[7, 1] += dy
            candidates.append(perturbed)

    return candidates

def _optimize_rigid(pts, max_iter_total=2000):
    """Optimize center and beta for a fixed rigid point set."""
    best_S = float('inf')
    best_beta = 0.0
    best_center = None

    # Initial grid
    for i in range(37):
        if not _time_ok():
            break
        beta = i * (math.pi / 6) / 36
        S = _required_side(pts, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = (0.0, 0.0)

    # Bounded scalar refinement
    try:
        if _time_ok():
            res = minimize_scalar(lambda b: _required_side(pts, b),
                                  bounds=(0, math.pi / 6), method='bounded',
                                  options={'maxiter': 500, 'xatol': 1e-14})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                best_center = (0.0, 0.0)
    except Exception:
        pass

    # Smooth Nelder-Mead polish
    try:
        if _time_ok():
            temperatures = [0.3, 0.1, 0.05, 0.02, 0.01]
            start_betas = [0.0, math.pi/12, math.pi/6, best_beta]
            for temp in temperatures:
                if not _time_ok():
                    break
                for sb in start_betas:
                    if not _time_ok():
                        break
                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        shifted_pts = pts - np.array([cx, cy])
                        return _required_side_smooth(shifted_pts, beta, temperature=t)
                    init = [best_center[0], best_center[1], sb]
                    res = minimize(smooth_obj, init, method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14,
                                         'adaptive': True})
                    if res.fun < 1e10:
                        cx_r, cy_r, beta_r = res.x
                        beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        shifted_pts = pts - np.array([cx_r, cy_r])
                        S_exact = _required_side(shifted_pts, beta_r)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_beta = beta_r
                            best_center = (cx_r, cy_r)
    except Exception:
        pass

    # Final exact Nelder-Mead polish
    try:
        if _time_ok():
            def objective(params):
                cx, cy, beta = params
                beta = max(0.0, min(beta, math.pi / 6))
                shifted_pts = pts - np.array([cx, cy])
                return _required_side(shifted_pts, beta)

            start_points = [
                [best_center[0], best_center[1], best_beta],
                [best_center[0], best_center[1], 0.0],
                [best_center[0], best_center[1], math.pi/12],
                [best_center[0] + 0.05, best_center[1] + 0.05, best_beta],
                [best_center[0] - 0.05, best_center[1] - 0.05, best_beta],
            ]
            for init in start_points:
                if not _time_ok():
                    break
                res = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-16,
                                     'adaptive': True})
                if res.fun < best_S:
                    best_S = res.fun
                    best_center = (res.x[0], res.x[1])
                    best_beta = max(0.0, min(res.x[2], math.pi / 6))
    except Exception:
        pass

    return best_S, best_beta, best_center

def _soft_rigid_polish(pts, beta, center, max_iter=300):
    """Allow small independent perturbations of each point to find tighter packing.
    This is a 'soft rigid' refinement: each point can move slightly, but we penalize
    large deviations to maintain the overall structure."""
    n = len(pts)
    # Parameters: 2n translations + n small angle perturbations (total 3n)
    # We'll use a penalty approach: minimize required_side + lambda * sum of deviations
    
    best_S = float('inf')
    best_pts = pts.copy()
    best_beta = beta
    best_center = center

    # Start from current best
    cx, cy = center
    shifted = pts - np.array([cx, cy])

    def full_objective(params, lam=10.0):
        # params: [dx0, dy0, ..., dx_{n-1}, dy_{n-1}, da0, ..., da_{n-1}, beta]
        # Total: 2n + n + 1
        idx = 0
        new_pts = np.zeros((n, 2))
        new_angles = np.zeros(n)
        for i in range(n):
            new_pts[i, 0] = shifted[i, 0] + params[idx]
            new_pts[i, 1] = shifted[i, 1] + params[idx + 1]
            idx += 2
        for i in range(n):
            new_angles[i] = params[idx]
            idx += 1
        new_beta = params[idx]
        new_beta = max(0.0, min(new_beta, math.pi / 6))

        # Penalty for deviations
        dev_penalty = 0.0
        for i in range(n):
            dev_penalty += (new_pts[i, 0] - shifted[i, 0])**2 + (new_pts[i, 1] - shifted[i, 1])**2
            dev_penalty += new_angles[i]**2

        S = _required_side(new_pts, new_beta)
        return S + lam * dev_penalty

    # Initial params
    init_params = np.zeros(3 * n + 1)
    init_params[-1] = beta

    try:
        res = minimize(full_objective, init_params, method='Nelder-Mead',
                     options={'maxiter': max_iter, 'xatol': 1e-14, 'fatol': 1e-14,
                             'adaptive': True})
        if res.fun < 1e10:
            # Extract new points
            idx = 0
            new_pts = np.zeros((n, 2))
            for i in range(n):
                new_pts[i, 0] = shifted[i, 0] + res.x[idx]
                new_pts[i, 1] = shifted[i, 1] + res.x[idx + 1]
                idx += 2
            idx = 2 * n
            for i in range(n):
                idx += 1  # skip angle perturbations
            new_beta = max(0.0, min(res.x[idx], math.pi / 6))

            S_exact = _required_side(new_pts, new_beta)
            if S_exact < best_S:
                best_S = S_exact
                best_pts = new_pts
                best_beta = new_beta
                best_center = (cx, cy)
    except Exception:
        pass

    return best_S, best_pts, best_beta, best_center

def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = _build_candidates()

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Phase 1: Fast screen all candidates
    screened = []
    for pts in candidates:
        if not _time_ok():
            break
        S, beta, center = _optimize_rigid(pts, max_iter_total=500)
        screened.append((S, pts, beta, center))

    # Sort by S and keep top for expensive optimization
    screened.sort(key=lambda x: x[0])
    top_k = screened[:6]

    for S, pts, beta, center in top_k:
        if not _time_ok():
            break
        # Rigid optimization
        S_opt, beta_opt, center_opt = _optimize_rigid(pts)
        if S_opt < best_global_S:
            best_global_S = S_opt
            best_global_pts = pts
            best_global_beta = beta_opt
            best_global_center = center_opt

        # Soft rigid polish
        if _time_ok():
            S_soft, pts_soft, beta_soft, center_soft = _soft_rigid_polish(pts, beta_opt, center_opt)
            if S_soft < best_global_S:
                best_global_S = S_soft
                best_global_pts = pts_soft
                best_global_beta = beta_soft
                best_global_center = center_soft

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = candidates[0]
        best_global_center = _centroid(best_global_pts)
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Final verification and scaling
    pts_centered = best_global_pts - np.array([cx, cy])
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = pts_centered[:, 0] * nx + pts_centered[:, 1] * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
