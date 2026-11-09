import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
SQRT3 = math.sqrt(3.0)

# Precompute vertex offsets for 6 unique rotation angles (60° symmetry)
_ROT_ANGLES = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
_ROT_COS = np.array([math.cos(math.radians(a)) for a in _ROT_ANGLES])
_ROT_SIN = np.array([math.sin(math.radians(a)) for a in _ROT_ANGLES])
_BASE_ANGLES = np.array([math.radians(60*k) for k in range(6)])
_BASE_COS = np.cos(_BASE_ANGLES)
_BASE_SIN = np.sin(_BASE_ANGLES)
# VERTEX_OFFSETS[rot_idx, 6, 2]
_VERTEX_OFFSETS = np.zeros((6, 6, 2))
for ri in range(6):
    for vi in range(6):
        ca, sa = _ROT_COS[ri], _ROT_SIN[ri]
        ba, bs = _BASE_COS[vi], _BASE_SIN[vi]
        _VERTEX_OFFSETS[ri, vi, 0] = ca * ba - sa * bs
        _VERTEX_OFFSETS[ri, vi, 1] = ca * bs + sa * ba

# Precompute 6 normals for L computation (outward directions of regular hexagon)
_NORMAL_ANGLES = np.array([math.radians(30 + 60*k) for k in range(6)])
_NORMAL_COS = np.cos(_NORMAL_ANGLES)
_NORMAL_SIN = np.sin(_NORMAL_ANGLES)
_NORMALS = np.column_stack([_NORMAL_COS, _NORMAL_SIN])  # (6, 2)


def _compute_all_vertices(params):
    """Compute vertices for all N hexagons. Returns (N, 6, 2) array."""
    centers = params[:N*2].reshape(N, 2)
    rot_idx = np.round(params[N*2:N*3]).astype(int) % 6
    # (N, 6, 2)
    verts = centers[:, np.newaxis, :] + _VERTEX_OFFSETS[rot_idx]
    return verts


def _compute_L(verts):
    """Compute minimal regular hexagon side length containing all vertices."""
    # verts: (N, 6, 2)
    all_v = verts.reshape(-1, 2)
    # Project onto 6 normals: (6, total_vertices)
    projs = all_v @ _NORMALS.T  # (total_vertices, 6)
    max_proj = projs.max(axis=0)  # (6,)
    max_proj = max_proj.max()
    return max_proj / (SQRT3 / 2.0)


def _compute_min_separation(verts_i, verts_j):
    """Compute minimum distance between two hexagons. Negative if overlapping.
    verts_i, verts_j: (6, 2) arrays."""
    min_dist = float('inf')
    
    # Check vertex containment
    for vi in verts_i:
        if _point_in_hex(vi, verts_j):
            d = _point_poly_dist(vi, verts_j)
            min_dist = min(min_dist, d)
    for vj in verts_j:
        if _point_in_hex(vj, verts_i):
            d = _point_poly_dist(vj, verts_i)
            min_dist = min(min_dist, d)
    
    # Edge-to-edge: vertex to opposite edge
    for vi in verts_i:
        for k in range(6):
            p3 = verts_j[k]
            p4 = verts_j[(k+1) % 6]
            d = _point_seg_dist(vi, p3, p4)
            min_dist = min(min_dist, d)
    for vj in verts_j:
        for k in range(6):
            p1 = verts_i[k]
            p2 = verts_i[(k+1) % 6]
            d = _point_seg_dist(vj, p1, p2)
            min_dist = min(min_dist, d)
    
    return min_dist


def _point_in_hex(pt, hex_verts):
    """Check if point is inside a convex hexagon using cross products."""
    for k in range(6):
        p1 = hex_verts[k]
        p2 = hex_verts[(k+1) % 6]
        # Cross product of (p2-p1) x (pt-p1)
        cross = (p2[0]-p1[0])*(pt[1]-p1[1]) - (p2[1]-p1[1])*(pt[0]-p1[0])
        if cross < -1e-12:
            return False
    return True


def _point_seg_dist(pt, p1, p2):
    """Distance from point to line segment p1-p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    len_sq = dx*dx + dy*dy
    if len_sq < 1e-20:
        return math.hypot(pt[0]-p1[0], pt[1]-p1[1])
    t = ((pt[0]-p1[0])*dx + (pt[1]-p1[1])*dy) / len_sq
    t = max(0.0, min(1.0, t))
    px = p1[0] + t*dx
    py = p1[1] + t*dy
    return math.hypot(pt[0]-px, pt[1]-py)


def _point_poly_dist(pt, poly):
    """Distance from point to polygon boundary."""
    min_d = float('inf')
    for k in range(6):
        d = _point_seg_dist(pt, poly[k], poly[(k+1)%6])
        min_d = min(min_d, d)
    return min_d


def _objective(params, sharpness=80.0):
    """Smooth objective: L + separation penalties."""
    verts = _compute_all_vertices(params)
    L = _compute_L(verts)
    
    # Separation penalty: for each pair, penalize if min_dist < 0
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            md = _compute_min_separation(verts[i], verts[j])
            if md < 0:
                penalty += sharpness * md * md
            elif md < 0.005:
                # Small gap penalty to encourage clean separation
                penalty += 0.5 * (0.005 - md) ** 2
    
    return L + penalty


def _objective_strict(params):
    """Strict objective for final polish."""
    return _objective(params, sharpness=500.0)


def _polygons_disjoint(v1, v2):
    """Exact disjointness check."""
    for v in v1:
        if _point_in_hex(v, v2):
            return False
    for v in v2:
        if _point_in_hex(v, v1):
            return False
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            if _segments_intersect(p1, p2, p3, p4):
                return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) - (B[1]-A[1])*(C[0]-A[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def _make_initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Natural hex grid: center + 6 ring + 5 outer
    # In hex grid, adjacent hexagons are SQRT3 apart
    # Center at origin
    grid = []
    grid.append([0.0, 0.0, 0.0])  # center
    for i in range(6):
        ang = math.pi/3 * i
        r = SQRT3
        grid.append([r*math.cos(ang), r*math.sin(ang), 0.0])
    # 5 more on outer ring (skip 1 to get 12 total)
    for i in range(5):
        ang = math.pi/3 * (i + 1)
        r = 2*SQRT3
        grid.append([r*math.cos(ang), r*math.sin(ang), 0.0])
    inits.append(np.array(grid).flatten())
    
    # 2. Full hex grid with all 12 closest positions
    # Hex grid positions: (q, r) -> x = 1.5*q, y = SQRT3*(r + q/2)
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q) <= 3 and abs(r) <= 3 and abs(q+r) <= 3:
                x = 1.5 * q
                y = SQRT3 * (r + q/2.0)
                candidates.append((x, y))
    # Sort by distance from origin, take 12
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    top12 = candidates[:12]
    inits.append(np.array([[x, y, 0.0] for x, y in top12]).flatten())
    
    # 3. Ring of 12
    ring = []
    for i in range(12):
        ang = 2*math.pi*i/12
        r = 2.0
        ring.append([r*math.cos(ang), r*math.sin(ang), float(i%6)])
    inits.append(np.array(ring).flatten())
    
    # 4. Two rows staggered
    two_row = []
    for row in range(2):
        y = (-1 if row == 0 else 1) * SQRT3 * 0.55
        for col in range(6):
            x = -2.25 + col * 1.5
            if row == 1:
                x += 0.75
            two_row.append([x, y, 0.0])
    inits.append(np.array(two_row).flatten())
    
    # 5. 3-4-3-2 rows
    rows = []
    row_data = [(1.5*SQRT3, 3), (0.5*SQRT3, 4), (-0.5*SQRT3, 3), (-1.5*SQRT3, 2)]
    for y, count in row_data:
        x_start = -(count-1)*0.75
        for ci in range(count):
            x = x_start + ci*1.5
            rows.append([x, y, 0.0])
    inits.append(np.array(rows).flatten())
    
    # 6. Random
    np.random.seed(7)
    rand = np.random.uniform(-2.5, 2.5, size=(N, 3))
    rand[:, 2] = np.random.randint(0, 6, size=N).astype(float)
    inits.append(rand.flatten())
    
    # 7. Compact with varied rotations
    compact = []
    for i in range(12):
        ang = 2*math.pi*i/12
        r = 1.7
        compact.append([r*math.cos(ang), r*math.sin(ang), float(i % 6)])
    inits.append(np.array(compact).flatten())
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
    
    best_params = None
    best_score = float('inf')
    
    def update_best(params, score):
        nonlocal best_params, best_score
        if score < best_score:
            best_score = score
            best_params = params.copy()
    
    initializations = _make_initializations()
    
    # Phase 1: Nelder-Mead from multiple starts
    for init in initializations:
        if time_left() < 60.0:
            break
        try:
            res = minimize(_objective, init, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: DE for global exploration (only if time permits)
    if time_left() > 55.0 and best_params is not None:
        try:
            bounds = []
            for i in range(N):
                bounds.append((-3.5, 3.5))
                bounds.append((-3.5, 3.5))
                bounds.append((0, 6))
            
            de_res = differential_evolution(_objective, bounds, seed=42,
                                            maxiter=60, tol=1e-6,
                                            popsize=12, polish=True,
                                            update_interval=15, workers=1)
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Local refinement with strict objective
    if best_params is not None and time_left() > 30.0:
        try:
            res = minimize(_objective_strict, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: L-BFGS-B polish
    if best_params is not None and time_left() > 15.0:
        try:
            res = minimize(_objective_strict, best_params, method='L-BFGS-B',
                          options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12,
                                   'maxfun': 20000})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final ultra-fine Nelder-Mead
    if best_params is not None and time_left() > 5.0:
        try:
            res = minimize(_objective_strict, best_params, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-12,
                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Coordinate descent fine-tuning
    if best_params is not None and time_left() > 3.0:
        try:
            p = best_params.copy()
            for _ in range(200):
                improved = False
                for dim in range(len(p)):
                    for delta in [0.01, -0.01, 0.001, -0.001, 0.0001, -0.0001]:
                        p2 = p.copy()
                        p2[dim] += delta
                        s2 = _objective_strict(p2)
                        if s2 < best_score - 1e-12:
                            p = p2
                            best_score = s2
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            best_params = p
        except Exception:
            pass
    
    if best_params is None:
        best_params = initializations[0]
    
    # Final evaluation
    verts = _compute_all_vertices(best_params)
    L_final = _compute_L(verts)
    
    # Build positions
    positions = []
    for i in range(N):
        x = best_params[2*i]
        y = best_params[2*i+1]
        rot = int(round(best_params[2*N+i])) % 6
        angle_deg = float(rot * 60)
        positions.append([x, y, angle_deg])
    
    # Verify disjointness
    is_valid = True
    verts_list = [verts[i] for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if not _polygons_disjoint(verts_list[i], verts_list[j]):
                is_valid = False
                break
        if not is_valid:
            break
    
    if not is_valid:
        # Fallback: use the hex grid initialization
        inner = []
        for i in range(12):
            x = initializations[0][2*i]
            y = initializations[0][2*i+1]
            inner.append([x, y, 0.0])
        all_v = _compute_all_vertices(initializations[0])
        L_final = _compute_L(all_v)
        return inner, (0.0, 0.0), L_final, 0.0
    
    return positions, (0.0, 0.0), L_final, 0.0
