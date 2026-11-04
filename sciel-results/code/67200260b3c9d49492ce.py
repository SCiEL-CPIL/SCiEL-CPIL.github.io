import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
DEADLINE = 140.0
EPS = 1e-9

# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 330]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_convex_polygon(pt, poly):
    """Check if point is inside (or on boundary of) a convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    if a < 1e-12:
        t = np.clip(f / e, 0.0, 1.0)
        return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        s = np.clip(c / a, 0.0, 1.0)
        return np.linalg.norm((p1 + s * d1) - p3)

    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = np.clip(s, 0.0, 1.0)
        t = (b * s + f) / e
        if t < 0:
            t = 0.0
            s = np.clip(c / a, 0.0, 1.0)
        elif t > 1:
            t = 1.0
            s = np.clip((b + f) / a, 0.0, 1.0)
        return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))
    else:
        t = np.clip(f / e, 0.0, 1.0)
        return np.linalg.norm(p1 - (p3 + t * d2))

def _poly_separation(p1, p2):
    """Minimum distance between two convex polygons. 0 if they overlap."""
    for v in p1:
        if _point_in_convex_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_convex_polygon(v, p1):
            return 0.0
    min_dist = float('inf')
    n1, n2 = len(p1), len(p2)
    for i in range(n1):
        a1, a2 = p1[i], p1[(i + 1) % n1]
        for j in range(n2):
            b1, b2 = p2[j], p2[(j + 1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

# ---------------------------------------------------------------------------
# Outer hexagon side computation
# ---------------------------------------------------------------------------
def _compute_min_side_fast(inner_params, outer_angle_deg):
    """
    Fast computation of minimum outer hexagon side length via support function.
    The outer hexagon is centered at origin, rotated outer_angle_deg.
    inner_params: (N, 3) array of [x, y, angle_deg]
    """
    all_vertices = np.zeros((N * 6, 2))
    for i in range(N):
        x, y, a = inner_params[i]
        all_vertices[i * 6:(i + 1) * 6] = _hex_vertices(x, y, a)

    # Outward normals of a regular hexagon rotated by outer_angle_deg
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    max_proj = 0.0
    for n_vec in normals:
        projs = np.dot(all_vertices, n_vec)
        m = np.max(projs)
        if m > max_proj:
            max_proj = m
    return 2.0 * max_proj / np.sqrt(3.0)

def _outer_hex_vertices(side, angle_deg):
    """Vertices of outer regular hexagon centered at origin."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs

def _is_contained(inner_params, outer_angle_deg, side):
    """Exact containment check: every vertex of every inner hexagon inside outer hexagon."""
    outer_v = _outer_hex_vertices(side, outer_angle_deg)
    for i in range(N):
        v = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for pt in v:
            if not _point_in_convex_polygon(pt, outer_v):
                return False
    return True

# ---------------------------------------------------------------------------
# Lattice-based configuration enumeration
# ---------------------------------------------------------------------------
def _generate_lattice_configurations():
    """
    Generate candidate configurations using hexagonal lattice positions.
    Each configuration is a list of 11 (x, y) positions with all hexagons at 30 degrees.
    Returns list of (N, 3) arrays.
    """
    configs = []
    sqrt3 = np.sqrt(3)
    
    # Configuration 1: Current "flower" pattern
    centers = [(0.0, 0.0)]
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (sqrt3 + EPS) * np.cos(angle)
        y = (sqrt3 + EPS) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2 * EPS) * np.cos(angle)
        y = (3.0 + 2 * EPS) * np.sin(angle)
        centers.append((x, y))
    configs.append(np.array([[x, y, 30.0] for x, y in centers]))
    
    # Configuration 2: Compact 3-3-5 row layout (hexagonal lattice)
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 5 hexagons
    centers = []
    # Row 0: y = 0
    for i in range(-1, 2):
        centers.append((i * 2 * sqrt3, 0.0))
    # Row 1: y = sqrt3, offset by sqrt3
    for i in range(-1, 2):
        centers.append((i * 2 * sqrt3 + sqrt3, sqrt3))
    # Row 2: y = 2*sqrt3
    for i in range(-2, 3):
        centers.append((i * 2 * sqrt3, 2 * sqrt3))
    # Center the configuration
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    centers = [(x - cx, y - cy) for x, y in centers]
    configs.append(np.array([[x, y, 30.0] for x, y in centers]))
    
    # Configuration 3: 2-3-3-3 layout
    centers = []
    # Row 0: 2 hexagons
    for i in range(-1, 1):
        centers.append((i * 2 * sqrt3, 0.0))
    # Row 1: 3 hexagons
    for i in range(-1, 2):
        centers.append((i * 2 * sqrt3 + sqrt3, sqrt3))
    # Row 2: 3 hexagons
    for i in range(-1, 2):
        centers.append((i * 2 * sqrt3, 2 * sqrt3))
    # Row 3: 3 hexagons
    for i in range(-1, 2):
        centers.append((i * 2 * sqrt3 + sqrt3, 3 * sqrt3))
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    centers = [(x - cx, y - cy) for x, y in centers]
    configs.append(np.array([[x, y, 30.0] for x, y in centers]))
    
    # Configuration 4: 1-3-4-3 layout
    centers = []
    # Row 0: 1 hexagon
    centers.append((0.0, 0.0))
    # Row 1: 3 hexagons
    for i in range(-1, 2):
        centers.append((i * 2 * sqrt3 + sqrt3, sqrt3))
    # Row 2: 4 hexagons
    for i in range(-2, 2):
        centers.append((i * 2 * sqrt3, 2 * sqrt3))
    # Row 3: 3 hexagons
    for i in range(-1, 2):
        centers.append((i * 2 * sqrt3 + sqrt3, 3 * sqrt3))
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    centers = [(x - cx, y - cy) for x, y in centers]
    configs.append(np.array([[x, y, 30.0] for x, y in centers]))
    
    # Configuration 5: 3-4-4 layout
    centers = []
    # Row 0: 3 hexagons
    for i in range(-1, 2):
        centers.append((i * 2 * sqrt3, 0.0))
    # Row 1: 4 hexagons
    for i in range(-2, 2):
        centers.append((i * 2 * sqrt3 + sqrt3, sqrt3))
    # Row 2: 4 hexagons
    for i in range(-2, 2):
        centers.append((i * 2 * sqrt3, 2 * sqrt3))
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    centers = [(x - cx, y - cy) for x, y in centers]
    configs.append(np.array([[x, y, 30.0] for x, y in centers]))
    
    # Configuration 6: 2-4-3-2 layout
    centers = []
    # Row 0: 2 hexagons
    for i in range(-1, 1):
        centers.append((i * 2 * sqrt3, 0.0))
    # Row 1: 4 hexagons
    for i in range(-2, 2):
        centers.append((i * 2 * sqrt3 + sqrt3, sqrt3))
    # Row 2: 3 hexagons
    for i in range(-1, 2):
        centers.append((i * 2 * sqrt3, 2 * sqrt3))
    # Row 3: 2 hexagons
    for i in range(-1, 1):
        centers.append((i * 2 * sqrt3 + sqrt3, 3 * sqrt3))
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    centers = [(x - cx, y - cy) for x, y in centers]
    configs.append(np.array([[x, y, 30.0] for x, y in centers]))
    
    # Configuration 7: Diamond/cross pattern
    centers = [(0.0, 0.0)]
    # 4 cardinal neighbors
    centers.append((2 * sqrt3, 0.0))
    centers.append((-2 * sqrt3, 0.0))
    centers.append((0.0, 2 * sqrt3))
    centers.append((0.0, -2 * sqrt3))
    # 4 diagonal neighbors
    centers.append((sqrt3, sqrt3))
    centers.append((-sqrt3, sqrt3))
    centers.append((sqrt3, -sqrt3))
    centers.append((-sqrt3, -sqrt3))
    # 2 more
    centers.append((3 * sqrt3, 0.0))
    centers.append((-3 * sqrt3, 0.0))
    configs.append(np.array([[x, y, 30.0] for x, y in centers]))
    
    return configs

# ---------------------------------------------------------------------------
# Overlap penalty (smooth-ish)
# ---------------------------------------------------------------------------
def _overlap_penalty(inner_params, weight=100.0):
    pen = 0.0
    for i in range(N):
        v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for j in range(i + 1, N):
            v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
            sep = _poly_separation(v1, v2)
            if sep < 0.01:
                pen += weight * (0.01 - sep) ** 2
    return pen

def _overlap_penalty_smooth(inner_params, weight=100.0, gap=0.01, beta=50.0):
    """Soft penalty using log-sum-exp to smooth the max of per-pair violations."""
    vals = []
    for i in range(N):
        v1 = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
        for j in range(i + 1, N):
            v2 = _hex_vertices(inner_params[j, 0], inner_params[j, 1], inner_params[j, 2])
            sep = _poly_separation(v1, v2)
            viol = max(0.0, gap - sep)
            vals.append(viol)
    if not vals:
        return 0.0
    arr = np.array(vals)
    mx = np.max(arr)
    if mx <= 0:
        return 0.0
    # log-sum-exp for a smooth max
    logsum = mx + np.log(np.sum(np.exp((arr - mx) * beta)) / beta)
    return weight * logsum

# ---------------------------------------------------------------------------
# Seed solution
# ---------------------------------------------------------------------------
def _seed_solution():
    """Deterministic flower packing: 1 center + 6 ring1 + 4 ring2."""
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + EPS) * np.cos(angle)
        y = (np.sqrt(3) + EPS) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2 * EPS) * np.cos(angle)
        y = (3.0 + 2 * EPS) * np.sin(angle)
        centers.append((x, y))

    inner = np.array([[x, y, 30.0] for x, y in centers])
    side = _compute_min_side_fast(inner, 30.0)
    return inner, (0.0, 0.0), side, 30.0

# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------
def _make_objective(inner_params, outer_angle, weight, smooth):
    """Returns a function f(outer_side) = outer_side + penalty (epigraph form)."""
    def f(t):
        if t < 0.5:
            return 1e6
        if smooth:
            pen = _overlap_penalty_smooth(inner_params, weight=weight)
        else:
            pen = _overlap_penalty(inner_params, weight=weight)
        return t + pen
    return f

def _polish_position(inner_params, outer_angle, weight):
    """
    High-precision local polish: optimize only the 20 free position coordinates,
    fixing the center at origin and all angles at 30 deg. Uses the epigraph
    formulation with an auxiliary variable t (outer side) to get a smooth
    objective for L-BFGS-B.
    """
    def f(params):
        # params: [t, x1, y1, ..., x10, y10]  (t is the outer side)
        t = params[0]
        ip = np.empty((N, 3))
        ip[0] = [0.0, 0.0, 30.0]
        for i in range(1, N):
            idx = (i - 1) * 2
            ip[i, 0] = params[1 + idx]
            ip[i, 1] = params[1 + idx + 1]
            ip[i, 2] = 30.0
        side = _compute_min_side_fast(ip, outer_angle)
        # Epigraph: minimize t s.t. t >= side, plus overlap penalty
        pen = _overlap_penalty_smooth(ip, weight=weight, gap=0.005, beta=100.0)
        # Soft epigraph: penalize t < side smoothly
        ep = max(0.0, side - t)
        return t + pen + 1000.0 * ep ** 2

    return f

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()

    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = seed_inner.copy()
    best_side = seed_side

    outer_angle = 30.0

    # --- Verify seed validity ---
    pen = 0.0
    for i in range(N):
        v1 = _hex_vertices(seed_inner[i, 0], seed_inner[i, 1], seed_inner[i, 2])
        for j in range(i + 1, N):
            v2 = _hex_vertices(seed_inner[j, 0], seed_inner[j, 1], seed_inner[j, 2])
            sep = _poly_separation(v1, v2)
            if sep < 0.005:
                pen += (0.005 - sep) ** 2

    seed_valid = pen < 1e-8

    # --- Phase 1: Enumerate lattice configurations and optimize outer angle ---
    if time.time() - start_time < DEADLINE - 20.0:
        try:
            configs = _generate_lattice_configurations()
            
            for cfg_idx, config in enumerate(configs):
                if time.time() - start_time > DEADLINE - 15.0:
                    break
                
                # Check if this configuration has overlaps
                overlap = False
                for i in range(N):
                    v1 = _hex_vertices(config[i, 0], config[i, 1], config[i, 2])
                    for j in range(i + 1, N):
                        v2 = _hex_vertices(config[j, 0], config[j, 1], config[j, 2])
                        sep = _poly_separation(v1, v2)
                        if sep < 0.001:
                            overlap = True
                            break
                    if overlap:
                        break
                
                if overlap:
                    continue
                
                # Optimize outer angle for this configuration
                def obj_angle(angle):
                    return _compute_min_side_fast(config, angle)
                
                # Try multiple starting angles
                best_cfg_side = float('inf')
                best_cfg_angle = 0.0
                
                for start_angle in [0, 15, 30, 45, 60]:
                    try:
                        res = minimize_scalar(obj_angle, bounds=(0, 60), 
                                             method='bounded',
                                             options={'xatol': 1e-10})
                        if res.fun < best_cfg_side:
                            best_cfg_side = res.fun
                            best_cfg_angle = res.x
                    except Exception:
                        pass
                
                # Also check the current best angle
                side_at_current = _compute_min_side_fast(config, outer_angle)
                if side_at_current < best_cfg_side:
                    best_cfg_side = side_at_current
                    best_cfg_angle = outer_angle
                
                if best_cfg_side < best_side - 1e-8:
                    best_side = best_cfg_side
                    best_inner = config.copy()
                    outer_angle = best_cfg_angle
        
        except Exception:
            pass

    # --- Phase 2: Nelder-Mead global search over positions (angles fixed at 30) ---
    if time.time() - start_time < DEADLINE - 15.0:
        try:
            def obj_nm(params):
                if time.time() - start_time > DEADLINE:
                    return 1e6
                ip = np.empty((N, 3))
                ip[0] = [0.0, 0.0, 30.0]
                for i in range(1, N):
                    idx = (i - 1) * 2
                    ip[i, 0] = params[idx]
                    ip[i, 1] = params[idx + 1]
                    ip[i, 2] = 30.0
                side = _compute_min_side_fast(ip, outer_angle)
                p = _overlap_penalty(ip, weight=200.0)
                return side + p

            init = np.zeros(20)
            for i in range(1, N):
                idx = (i - 1) * 2
                init[idx] = best_inner[i, 0]
                init[idx + 1] = best_inner[i, 1]

            result = minimize(obj_nm, init, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8,
                                       'adaptive': True, 'disp': False})

            if time.time() - start_time < DEADLINE - 10.0 and result.fun < best_side:
                ip = np.empty((N, 3))
                ip[0] = [0.0, 0.0, 30.0]
                for i in range(1, N):
                    idx = (i - 1) * 2
                    ip[i, 0] = result.x[idx]
                    ip[i, 1] = result.x[idx + 1]
                    ip[i, 2] = 30.0
                # Verify no overlap
                vp = 0.0
                for i in range(N):
                    v1 = _hex_vertices(ip[i, 0], ip[i, 1], ip[i, 2])
                    for j in range(i + 1, N):
                        v2 = _hex_vertices(ip[j, 0], ip[j, 1], ip[j, 2])
                        sep = _poly_separation(v1, v2)
                        if sep < 0.005:
                            vp += (0.005 - sep) ** 2
                if vp < 1e-8:
                    new_side = _compute_min_side_fast(ip, outer_angle)
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = ip
        except Exception:
            pass

    # --- Phase 3: High-precision L-BFGS-B polish with smooth epigraph ---
    if time.time() - start_time < DEADLINE - 5.0:
        try:
            f_polish = _polish_position(best_inner, outer_angle, weight=500.0)
            # Initial params: [t, x1, y1, ..., x10, y10]
            t0 = best_side
            p0 = np.zeros(21)
            p0[0] = t0
            for i in range(1, N):
                idx = (i - 1) * 2
                p0[1 + idx] = best_inner[i, 0]
                p0[1 + idx + 1] = best_inner[i, 1]

            res2 = minimize(f_polish, p0, method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10,
                                     'maxfun': 5000, 'disp': False})

            if res2.fun < best_side + 1e-6:
                ip = np.empty((N, 3))
                ip[0] = [0.0, 0.0, 30.0]
                for i in range(1, N):
                    idx = (i - 1) * 2
                    ip[i, 0] = res2.x[1 + idx]
                    ip[i, 1] = res2.x[1 + idx + 1]
                    ip[i, 2] = 30.0
                # Verify
                vp = 0.0
                for i in range(N):
                    v1 = _hex_vertices(ip[i, 0], ip[i, 1], ip[i, 2])
                    for j in range(i + 1, N):
                        v2 = _hex_vertices(ip[j, 0], ip[j, 1], ip[j, 2])
                        sep = _poly_separation(v1, v2)
                        if sep < 0.003:
                            vp += (0.003 - sep) ** 2
                if vp < 1e-10:
                    new_side = _compute_min_side_fast(ip, outer_angle)
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = ip
        except Exception:
            pass

    # --- Phase 4: Final exact side computation and containment verification ---
    # Recompute side precisely
    final_side = _compute_min_side_fast(best_inner, outer_angle)
    # Add tiny margin for floating-point safety
    final_side += 1e-9

    # Verify exact containment with the computed side
    if not _is_contained(best_inner, outer_angle, final_side):
        # Fallback: use the seed which we know is valid
        final_side = seed_side + 1e-9
        best_inner = seed_inner

    return [list(row) for row in best_inner], seed_center, final_side, outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
