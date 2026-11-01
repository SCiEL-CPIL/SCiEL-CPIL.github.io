import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
R = 1.0
START_TIME = time.time()
TIME_LIMIT = 148.0


def _time_ok(margin=5.0):
    return time.time() - START_TIME < TIME_LIMIT - margin


def _axial_to_cartesian(axial_list, scale=1.0):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0) * scale
        y = 1.5 * r * scale
        pts.append((x, y))
    return np.array(pts, dtype=float)


def _hexagon_vertices(center, angle, side=1.0):
    """Return 6 vertices of a regular hexagon centered at center, rotated by angle, side length side."""
    # A regular hexagon with side s has circumradius s
    # Vertices are at angles angle + k*pi/3 from center
    vs = []
    for k in range(6):
        a = angle + k * math.pi / 3.0
        vs.append((center[0] + side * math.cos(a), center[1] + side * math.sin(a)))
    return np.array(vs)


def _support_function(pts, angle):
    """Max of p·n over points p, where n = (cos(angle), sin(angle))."""
    n = np.array([math.cos(angle), math.sin(angle)])
    return float(np.max(pts @ n))


def _outer_side_lp(pts, beta):
    """
    Given fixed inner hexagon centers pts and outer hexagon orientation beta,
    find outer center and minimum side length S such that all inner hexagons fit.
    
    The outer hexagon (side S, orientation beta) contains inner hexagon i (center p_i, angle th_i)
    iff for all 6 face normals of outer: p_i·n_k + h_i(n_k) <= S * apothem - c·n_k
    where h_i(n_k) is the support function of hexagon i in direction n_k,
    and c is outer center.
    
    Actually: outer hexagon with center c, side S, orientation beta has 6 faces.
    Face k has outward normal n_k = (cos(beta+pi/6+k*pi/3), sin(...)) and distance S*APOTHEM from center.
    Inner hexagon i with center p_i and orientation th_i is inside iff:
      (p_i - c)·n_k + h_i(n_k) <= S * APOTHEM  for all k
    where h_i(n_k) = max over vertices v of hexagon i of v·n_k = R * max_k' cos(th_i + k'*pi/3 - angle(n_k))
    Actually h_i(n) for a regular hexagon of circumradius R centered at origin, rotated by th:
      h(n) = R * max_{k=0..5} cos(th + k*pi/3 - phi) where phi = atan2(n_y, n_x)
    But since the hexagon has 6-fold symmetry, h(n) = R * max_k cos(k*pi/3 - (phi - th))
    And max over k of cos(k*pi/3 - alpha) = cos(alpha mod pi/3 adjusted) ... 
    Actually for a regular hexagon, h(n) = R * cos(d) where d is the minimum angle between n and any vertex direction.
    The vertex directions are th, th+60, th+120, th+180, th+240, th+300.
    So d = min_k |angle_diff(phi, th + k*60)|, and h = R*cos(d).
    
    Let me just compute it directly.
    """
    # 6 outward normals of outer hexagon
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    # For each inner hexagon, compute h_i(n_k) for each outer normal
    # h_i(n) = max over 6 vertices of hex i of v·n
    # Vertex j of hex i: p_i + R*(cos(th_i + j*pi/3), sin(th_i + j*pi/3))
    # v·n = p_i·n + R*cos(th_i + j*pi/3 - phi_n)
    # max over j: p_i·n + R * max_j cos(th_i + j*pi/3 - phi_n)
    
    # LP variables: c_x, c_y, S
    # Constraints: for each i, k: (p_i - c)·n_k + h_i(n_k) <= S * APOTHEM
    # => -c·n_k - S*APOTHEM <= -(p_i·n_k + h_i(n_k))
    # Also: S >= 0
    
    c = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        n = np.array([nx, ny])
        phi_n = math.atan2(ny, nx)
        for i in range(len(pts)):
            pi = pts[i]
            pi_n = float(pi @ n)
            # h_i(n) = R * max_j cos(th_i + j*pi/3 - phi_n)
            # But we don't know th_i here... 
            # In this formulation, we need th_i to be fixed.
            # Let's handle this differently.
    
    # This approach requires knowing th_i. Let me restructure.
    # For a given configuration (positions p_i, angles th_i), compute the required S.
    # The LP is: minimize S subject to (p_i - c)·n_k + h_i(n_k) <= S*APOTHEM for all i,k
    # This is linear in (c_x, c_y, S) given fixed p_i, th_i.
    
    # I'll make a separate function that takes positions and angles.
    return None  # placeholder


def _outer_side_given_config(pts, th, beta):
    """
    Given inner hexagon centers pts (n,2), angles th (n,), and outer orientation beta,
    solve LP to find outer center and min side S.
    """
    n = len(pts)
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append(np.array([math.cos(ang), math.sin(ang)]))
    
    c = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    
    for nk in normals:
        phi_n = math.atan2(nk[1], nk[0])
        for i in range(n):
            pi = pts[i]
            pi_n = float(pi @ nk)
            # h_i(nk) = R * max_j cos(th_i + j*pi/3 - phi_n)
            angles = th[i] + np.arange(6) * (math.pi / 3.0) - phi_n
            h_i = R * float(np.max(np.cos(angles)))
            # Constraint: (p_i - c)·n_k + h_i <= S * APOTHEM
            # => -c_x*nk_x - c_y*nk_y - S*APOTHEM <= -(pi_n + h_i)
            A_ub.append([-nk[0], -nk[1], -APOTHEM])
            b_ub.append(-(pi_n + h_i))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0.0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            S = res.x[2]
            center = (res.x[0], res.x[1])
            return S, center
    except Exception:
        pass
    return float('inf'), None


def _best_orientation_given_config(pts, th):
    """Find best outer orientation beta for a given inner configuration."""
    best_S = float('inf')
    best_beta = 0.0
    best_c = None
    
    # Coarse sweep over beta in [0, pi/3) due to 6-fold symmetry of outer hex
    for i in range(61):
        beta = i * (math.pi / 3.0) / 60
        S, c = _outer_side_given_config(pts, th, beta)
        if S < best_S:
            best_S, best_beta, best_c = S, beta, c
    
    # Refine with local optimization
    try:
        def obj(b):
            S, _ = _outer_side_given_config(pts, th, b)
            return S if np.isfinite(S) else 1e10
        
        res = minimize_scalar(obj, bounds=(max(0, best_beta - 0.05), min(math.pi/3, best_beta + 0.05)),
                              method='bounded', options={'maxiter': 100, 'xatol': 1e-10})
        if res.fun < best_S:
            best_S = res.fun
            best_beta = res.x
            _, best_c = _outer_side_given_config(pts, th, best_beta)
    except Exception:
        pass
    
    return best_S, best_beta, best_c


def _check_nonoverlap(pts, th, tol=-1e-8):
    """Check if all pairs of inner hexagons are non-overlapping. Return True if valid."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(pts[i], th[i], pts[j], th[j], tol):
                return False
    return True


def _hexagons_disjoint(p1, th1, p2, th2, tol=-1e-8):
    """Check if two hexagons are disjoint (non-overlapping). Use separating axis theorem."""
    # SAT: if there exists a separating axis, they are disjoint.
    # For two convex polygons, check all edge normals of both.
    v1 = _hexagon_vertices(p1, th1)
    v2 = _hexagon_vertices(p2, th2)
    
    # Get all edge normals
    axes = []
    for v in [v1, v2]:
        for k in range(6):
            edge = v[(k+1) % 6] - v[k]
            # Normal to edge
            nx = -edge[1]
            ny = edge[0]
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append((nx/norm, ny/norm))
    
    for nx, ny in axes:
        # Project both polygons
        proj1 = np.array([v[0]*nx + v[1]*ny for v in v1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in v2])
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # If separated: max1 + tol < min2 or max2 + tol < min1
        if max1 + tol < min2 or max2 + tol < min1:
            return True
    return False


def _gap_sat(p1, th1, p2, th2):
    """Compute minimum gap between two hexagons using SAT. Positive = separated, negative = overlap."""
    v1 = _hexagon_vertices(p1, th1)
    v2 = _hexagon_vertices(p2, th2)
    
    axes = []
    for v in [v1, v2]:
        for k in range(6):
            edge = v[(k+1) % 6] - v[k]
            nx = -edge[1]
            ny = edge[0]
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append((nx/norm, ny/norm))
    
    min_gap = float('inf')
    for nx, ny in axes:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in v1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in v2])
        gap = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
        # Actually the gap in this direction is:
        # If max1 < min2: gap = min2 - max1 (positive, separated)
        # If max2 < min1: gap = min1 - max2 (positive, separated)  
        # If overlapping in this direction: gap = min(max1,max2) - max(min1,min2) (negative)
        if np.max(proj1) <= np.min(proj2):
            gap = np.min(proj2) - np.max(proj1)
        elif np.max(proj2) <= np.min(proj1):
            gap = np.min(proj1) - np.max(proj2)
        else:
            gap = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
        min_gap = min(min_gap, gap)
    
    return float(min_gap)


def _optimize_positions(pts0, th0, max_iter=2000):
    """
    Optimize positions and angles to maximize minimum gap between all pairs.
    Uses subgradient method.
    """
    n = len(pts0)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pts = pts0.copy()
    th = th0.copy()
    
    best_gap = _min_gap(pts, th)
    
    lr = 0.1
    for it in range(max_iter):
        if not _time_ok(3.0):
            break
        
        # Compute gradient of min gap (subgradient)
        grad_pts = np.zeros_like(pts)
        grad_th = np.zeros_like(th)
        
        # Find the pair with minimum gap
        min_g = float('inf')
        min_pair = None
        min_axis = None
        
        for i, j in pairs:
            g = _gap_sat(pts[i], th[i], pts[j], th[j])
            if g < min_g:
                min_g = g
                min_pair = (i, j)
        
        if min_g > 1e-6:
            # Well separated, try to improve further
            pass
        
        # For the critical pair, compute subgradient
        if min_pair is not None:
            i, j = min_pair
            g, dp_i, dp_j, dth_i, dth_j = _gap_grad_sat(pts[i], th[i], pts[j], th[j])
            grad_pts[i] += dp_i
            grad_pts[j] += dp_j
            grad_th[i] += dth_i
            grad_th[j] += dth_j
        
        # Update
        pts += lr * grad_pts
        th += lr * grad_th
        
        # Normalize angles
        th = th % (2 * math.pi)
        
        # Adaptive learning rate
        new_gap = _min_gap(pts, th)
        if new_gap > best_gap:
            best_gap = new_gap
            lr *= 0.95
        else:
            lr *= 0.9
        
        if lr < 1e-6:
            break
    
    return pts, th, best_gap


def _min_gap(pts, th):
    """Compute minimum gap over all pairs."""
    n = len(pts)
    min_g = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            g = _gap_sat(pts[i], th[i], pts[j], th[j])
            if g < min_g:
                min_g = g
    return min_g


def _gap_grad_sat(p1, th1, p2, th2):
    """
    Compute subgradient of gap between two hexagons.
    Returns (gap, dp1, dp2, dth1, dth2).
    """
    v1 = _hexagon_vertices(p1, th1)
    v2 = _hexagon_vertices(p2, th2)
    
    axes = []
    for v in [v1, v2]:
        for k in range(6):
            edge = v[(k+1) % 6] - v[k]
            nx = -edge[1]
            ny = edge[0]
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append((nx/norm, ny/norm))
    
    min_gap = float('inf')
    min_axis = None
    
    for nx, ny in axes:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in v1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in v2])
        
        if np.max(proj1) <= np.min(proj2):
            gap = np.min(proj2) - np.max(proj1)
        elif np.max(proj2) <= np.min(proj1):
            gap = np.min(proj1) - np.max(proj2)
        else:
            gap = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
        
        if gap < min_gap:
            min_gap = gap
            min_axis = (nx, ny)
    
    if min_axis is None:
        return min_gap, np.zeros(2), np.zeros(2), 0.0, 0.0
    
    nx, ny = min_axis
    
    # Determine which polygon is on which side
    c1 = float(np.mean([v[0]*nx + v[1]*ny for v in v1]))
    c2 = float(np.mean([v[0]*nx + v[1]*ny for v in v2]))
    
    if c1 < c2:
        # Hex1 is to the left, hex2 to the right
        dp1 = np.array([nx, ny])
        dp2 = np.array([-nx, -ny])
    else:
        dp1 = np.array([-nx, -ny])
        dp2 = np.array([nx, ny])
    
    # Angular gradients: d/dth of projection
    # For hex1, vertex k has angle th1 + k*pi/3
    # d/dth1 of (p1 + R*cos(th1+k*pi/3, sin(th1+k*pi/3)))·n
    # = R*(-sin(th1+k*pi/3)*nx + cos(th1+k*pi/3)*ny)
    # The active vertex is the one with max projection
    proj1 = np.array([v[0]*nx + v[1]*ny for v in v1])
    k1 = int(np.argmax(proj1))
    a1 = th1 + k1 * math.pi / 3.0
    dth1 = R * (-math.sin(a1) * nx + math.cos(a1) * ny)
    
    proj2 = np.array([v[0]*nx + v[1]*ny for v in v2])
    k2 = int(np.argmax(proj2))
    a2 = th2 + k2 * math.pi / 3.0
    dth2 = R * (-math.sin(a2) * nx + math.cos(a2) * ny)
    
    # Sign depends on which side
    if c1 < c2:
        dth1 = dth1  # increasing th1 moves hex1 in +n direction if dth1>0
        dth2 = -dth2
    else:
        dth1 = -dth1
        dth2 = dth2
    
    return min_gap, dp1, dp2, float(dth1), float(dth2)


def construct():
    if not _time_ok(10.0):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Candidate lattice configurations
    candidates = [
        # Flower (1+6+4)
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (1, 1), (-1, 2), (-2, 1), (2, -1)],
        # Rows 4-3-4
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2), (3, 2)],
        # 3-5-3
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (4, 1), (0, 2), (1, 2), (2, 2)],
        # 5-3-3
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2)],
        # 4-4-3
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (1, 2), (2, 2), (3, 2)],
        # 3-4-4
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
        # 2-3-3-3
        [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2),
         (2, 2), (0, 3), (1, 3), (2, 3)],
        # 3-3-3-2
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2),
         (1, 2), (2, 2), (0, 3), (1, 3)],
        # 4-3-4 variant
        [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2), (3, 2)],
        # 5-4-2
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1),
         (2, 1), (3, 1), (1, 2), (2, 2)],
        # 2-4-5
        [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2),
         (1, 2), (2, 2), (3, 2), (4, 2)],
        # 3-4-4 alt
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)],
    ]

    best_S = float('inf')
    best_result = None

    for idx, axial in enumerate(candidates):
        if not _time_ok(5.0):
            break
        
        pts0 = _axial_to_cartesian(axial, scale=1.0 + 1e-6)
        centroid = pts0.mean(axis=0)
        pts0 = pts0 - centroid
        th0 = np.zeros(len(pts0))
        
        # Try multiple initial angle configurations
        angle_inits = [
            np.zeros(len(pts0)),
            np.full(len(pts0), math.pi/6),
            np.full(len(pts0), math.pi/12),
            np.linspace(0, math.pi/6, len(pts0)),
        ]
        
        for ai, th_init in enumerate(angle_inits):
            if not _time_ok(3.0):
                break
            
            # Optimize positions and angles
            try:
                pts_opt, th_opt, gap = _optimize_positions(pts0, th_init, max_iter=800)
            except Exception:
                pts_opt, th_opt = pts0, th_init
            
            # Check validity
            if not _check_nonoverlap(pts_opt, th_opt, tol=-1e-7):
                # Try with original lattice positions
                if _check_nonoverlap(pts0, th_init, tol=-1e-7):
                    pts_use = pts0
                    th_use = th_init
                else:
                    continue
            else:
                pts_use = pts_opt
                th_use = th_opt
            
            # Find best outer orientation
            S_out, beta_out, c_out = _best_orientation_given_config(pts_use, th_use)
            
            if S_out < best_S and c_out is not None and np.isfinite(S_out):
                best_S = S_out
                best_result = (pts_use, th_use, c_out, beta_out)
        
        # Also try the lattice configuration directly with shared orientation
        S_lattice, beta_l, c_l = _best_orientation_given_config(pts0, np.zeros(len(pts0)))
        if S_lattice < best_S and c_l is not None:
            best_S = S_lattice
            best_result = (pts0, np.zeros(len(pts0)), c_l, beta_l)

    if best_result is None or not _time_ok(2.0):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    p_use, th_use, c_out, best_beta = best_result
    inner = []
    for k in range(len(p_use)):
        inner.append([float(p_use[k, 0]), float(p_use[k, 1]),
                      float(math.degrees(th_use[k]))])
    center = [float(c_out[0]), float(c_out[1])]
    side = float(best_S)
    rotation = float(math.degrees(best_beta))
    return inner, center, side, rotation
