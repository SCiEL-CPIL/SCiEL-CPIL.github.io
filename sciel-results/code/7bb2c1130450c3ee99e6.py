import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 20.0

# Unit hexagon (side 1) vertices for angle 0 (flat-top):
# angles 0,60,120,180,240,300 deg
_HEX_VERTS = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))]
    for a in range(0, 360, 60)
])  # 6x2

def _time_ok():
    return (time.time() - START_TIME) < (TIME_LIMIT - SAFETY_MARGIN)

def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def _hex_vertices(cx, cy, theta):
    R = _rot(theta)
    return (R @ _HEX_VERTS.T).T + np.array([cx, cy])

def _min_sep(V1, V2):
    """Minimum Euclidean distance between two convex polygons (each 6 vertices).
    Returns (min_dist, pair_idx1, pair_idx2)."""
    best = float('inf')
    bi1 = bi2 = 0
    # edge-edge, vertex-edge, vertex-vertex
    n1, n2 = len(V1), len(V2)
    for i in range(n1):
        p1 = V1[i]; p2 = V1[(i + 1) % n1]
        e1 = p2 - p1
        for j in range(n2):
            q1 = V2[j]; q2 = V2[(j + 1) % n2]
            e2 = q2 - q1
            # vertex-vertex
            d = np.linalg.norm(p1 - q1)
            if d < best:
                best, bi1, bi2 = d, (i, 0), (j, 0)
            # vertex p1 to edge e2
            w = q1 - p1
            a = np.dot(e2, e2)
            b = 2.0 * np.dot(e2, w)
            c = np.dot(w, w)
            denom = a * a - b * b
            if abs(denom) > 1e-12:
                t1 = (b * c - a * w.dot(e2)) / denom
                t2 = (b * w.dot(e2) - c * b) / denom
                # Actually use standard closest-point-on-segment for p1 to e2
            # simpler: clamp projection
            t = np.dot(p1 - q1, e2) / (np.dot(e2, e2) + 1e-15)
            t = max(0.0, min(1.0, t))
            cp = q1 + t * e2
            d = np.linalg.norm(p1 - cp)
            if d < best:
                best, bi1, bi2 = d, (i, 0), (j, 1)
            # vertex q1 to edge e1
            t = np.dot(q1 - p1, e1) / (np.dot(e1, e1) + 1e-15)
            t = max(0.0, min(1.0, t))
            cp = p1 + t * e1
            d = np.linalg.norm(q1 - cp)
            if d < best:
                best, bi1, bi2 = d, (i, 1), (j, 0)
    return best, bi1, bi2

def _separation(V1, V2):
    """Separation distance: positive if disjoint, negative if overlapping.
    We use min distance between boundaries as a proxy (works for disjoint)."""
    d, _, _ = _min_sep(V1, V2)
    return d

def _objective(params, eps=0.05, lam=1e-4):
    """Soft objective: penalize overlaps (negative separations) via softmin,
    plus tiny spread penalty to avoid degenerate coincident solutions."""
    n = N
    cx = params[0:n]
    cy = params[n:2*n]
    theta = params[2*n:3*n]
    
    # Build vertices for each hexagon
    verts = []
    for i in range(n):
        verts.append(_hex_vertices(cx[i], cy[i], theta[i]))
    
    # Compute pairwise separations
    seps = []
    for i in range(n):
        for j in range(i + 1, n):
            d = _separation(verts[i], verts[j])
            seps.append(d)
    seps = np.array(seps)
    
    # Soft-min of separations (smooth min): want all seps >= 0
    # Penalize the "most violating" separation
    # Use: -softmin(seps) where softmin = -eps*log(sum(exp(-s/eps)))
    # If all seps > 0, softmin > 0, -softmin < 0 -> we want to maximize softmin
    # Objective = -softmin(seps) + lam * spread
    # To encourage separation, we minimize -softmin (i.e., maximize softmin)
    m = np.min(seps)
    shifted = (seps - m) / eps
    shifted = np.clip(shifted, -50, 50)
    softmin = m - eps * math.log(np.sum(np.exp(shifted)))
    
    # Spread penalty: keep hexagons from collapsing to same point
    spread = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            spread += (cx[i] - cx[j])**2 + (cy[i] - cy[j])**2
    spread = math.sqrt(spread)
    
    return -softmin + lam * spread

def _enclosing_hexagon(params):
    """Given positions/angles, find minimal enclosing regular hexagon (rotatable).
    Returns (side, center, rotation_deg)."""
    n = N
    cx = params[0:n]
    cy = params[n:2*n]
    theta = params[2*n:3*n]
    
    # Collect all vertices of all inner hexagons
    all_pts = []
    for i in range(n):
        all_pts.append(_hex_vertices(cx[i], cy[i], theta[i]))
    all_pts = np.vstack(all_pts)  # (6n, 2)
    
    # Search over outer hexagon rotation beta in [0, pi/6)
    # For a given beta, the outer hexagon has 6 half-plane constraints.
    # We need smallest D such that all points satisfy |dot(p-c, n_k)| <= D for all k.
    # Optimal c for fixed beta: minimize D -> LP or use Chebyshev-like.
    # For hexagon symmetry, optimal center is the center of the bounding box in each of 3 axes.
    # Actually for a regular hexagon with normals n_k, the minimal enclosing is:
    # D = max over k of max over p of dot(p-c, n_k) where c chosen to center.
    # For fixed beta, the optimal c is the intersection of midlines; we can solve:
    # minimize D s.t. dot(p_i - c, n_k) <= D for all i,k and >= -D.
    # This is LP in (c_x, c_y, D). Use scipy linprog.
    
    best_S = float('inf')
    best_beta = 0.0
    best_c = None
    
    # Coarse grid
    for i in range(37):
        beta = i * (math.pi / 6) / 36
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # LP: min D s.t. for each point p, each normal n:
        #   dot(p, n) - dot(c, n) <= D
        #   dot(c, n) - dot(p, n) <= D
        # Variables: c_x, c_y, D
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for nx, ny in normals:
            for px, py in all_pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        try:
            from scipy.optimize import linprog
            res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                          bounds=[(None, None), (None, None), (0, None)],
                          method='highs')
            if res.status == 0:
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM
                if S < best_S:
                    best_S = S
                    best_beta = beta
                    best_c = (res.x[0], res.x[1])
        except Exception:
            pass
    
    # Refine beta
    try:
        from scipy.optimize import minimize_scalar
        def obj_beta(beta):
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []
            for nx, ny in normals:
                for px, py in all_pts:
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(px * nx + py * ny))
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(px * nx + py * ny)
            res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                          bounds=[(None, None), (None, None), (0, None)],
                          method='highs')
            if res.status == 0:
                return (res.x[2] + APOTHEM) / APOTHEM
            return 1e10
        res = minimize_scalar(obj_beta, bounds=(0, math.pi / 6), method='bounded',
                              options={'maxiter': 200, 'xatol': 1e-10})
        if res.fun < best_S:
            best_S = res.fun
            best_beta = res.x
            # recompute center
            normals = []
            for k in range(6):
                ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []
            for nx, ny in normals:
                for px, py in all_pts:
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(px * nx + py * ny))
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(px * nx + py * ny)
            res2 = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                           bounds=[(None, None), (None, None), (0, None)],
                           method='highs')
            if res2.status == 0:
                best_c = (res2.x[0], res2.x[1])
    except Exception:
        pass
    
    if best_c is None:
        best_c = (float(np.mean(all_pts[:, 0])), float(np.mean(all_pts[:, 1])))
    
    angle_deg = math.degrees(best_beta)
    return best_S, best_c, angle_deg

def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Initialize positions: loose hexagonal lattice to ensure disjoint start
    # Hex lattice spacing: for unit hexagons, center-to-center >= 2*APOTHEM*sqrt(3) = 3
    # Actually for flat-top hexagons, horizontal spacing = 2*side = 2, vertical = 1.5*sqrt(3)
    # Use spacing 3.0 to be safe.
    def make_initial(seed_offset=0.0):
        positions = []
        # 4x3 grid arrangement
        cols = 4
        rows = 3
        spacing = 3.0 + seed_offset
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N:
                    break
                x = c * spacing + (r % 2) * spacing * 0.5
                y = r * spacing * math.sqrt(3) / 2.0
                positions.append([x, y, 0.0])
                idx += 1
        # Center around origin
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        mx = (max(xs) + min(xs)) / 2.0
        my = (max(ys) + min(ys)) / 2.0
        for p in positions:
            p[0] -= mx
            p[1] -= my
        return positions
    
    best_result = None
    best_S = float('inf')
    
    # Try multiple initializations
    for attempt in range(5):
        if not _time_ok():
            break
        
        init_pos = make_initial(seed_offset=attempt * 0.1)
        params0 = []
        for x, y, th in init_pos:
            params0.extend([x, y, th])
        
        try:
            res = minimize(_objective, params0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if not _time_ok():
                break
            params = res.x
            S, center, angle_deg = _enclosing_hexagon(params)
            if S < best_S:
                best_S = S
                best_result = (params, center, angle_deg)
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: single hexagon
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    params, center, angle_deg = best_result
    cx = params[0:N]
    cy = params[N:2*N]
    theta = params[2*N:3*N]
    
    inner = []
    for i in range(N):
        inner.append([float(cx[i]), float(cy[i]), float(math.degrees(theta[i]))])
    
    side = float(best_S)
    rotation = float(angle_deg)
    center_out = [float(center[0]), float(center[1])]
    
    return inner, center_out, side, rotation
