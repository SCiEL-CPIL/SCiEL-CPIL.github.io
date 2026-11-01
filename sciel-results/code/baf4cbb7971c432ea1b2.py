"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    # Normals in outer frame (30 + i*60 degrees)
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        # Transform to outer local frame
        local = (verts - np.array([ox, oy])) @ R.T
        # Project onto all 6 normals
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth (log-sum-exp) surrogate for the max projection.
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        all_projs.append(projs)
    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0
    m = P.max()
    if m < -1e6:
        return 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (lists of (x,y))."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
    for ax in axes:
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty(inner, min_dist=SQRT3 - 0.004):
    """Quadratic penalty when center distances fall below min_dist."""
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def construct():
    t0 = time.time()
    deadline = t0 + 142.0  # keep 8 s safety margin

    # --- Initial hexagonal lattice (11 centers) ---
    axial = [(0, 0),
             (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, -1), (1, 1), (-2, 1), (-1, -1)]
    s0 = 1.0 + 1e-7
    init_centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + q / 2.0)) for q, r in axial]
    init_inner = [[x, y, 0.0] for x, y in init_centers]
    cx0 = sum(c[0] for c in init_centers) / len(init_centers)
    cy0 = sum(c[1] for c in init_centers) / len(init_centers)

    best_side = compute_required_side_fast(init_inner, (cx0, cy0), 0.0)
    best_inner = init_inner
    best_outer_center = (cx0, cy0)
    best_outer_angle = 0.0

    # ============ Phase 1: Optimize outer hexagon (center + rotation) ============
    def obj_outer_smooth(p):
        return compute_required_side_smooth(init_inner, (p[0], p[1]), p[2], beta=50.0)

    def obj_outer_exact(p):
        return compute_required_side_fast(init_inner, (p[0], p[1]), p[2])

    best_val_p1 = float('inf')
    best_p1 = None
    init_angles = list(range(0, 360, 15))
    for ia in init_angles:
        if time.time() > deadline - 60:
            break
        p0 = np.array([cx0, cy0, float(ia)])
        try:
            r1 = minimize(obj_outer_smooth, p0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 600})
            r2 = minimize(obj_outer_exact, r1.x, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 1000})
            if r2.fun < best_val_p1:
                best_val_p1 = r2.fun
                best_p1 = r2.x.copy()
        except Exception:
            pass
    if best_p1 is not None:
        best_outer_center = (float(best_p1[0]), float(best_p1[1]))
        best_outer_angle = float(best_p1[2])
        best_side = best_val_p1
    else:
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(init_inner, best_outer_center, best_outer_angle)

    # ============ Phase 2: Multi-start Simulated Annealing ============
    
    def run_sa(init_params, max_iter=8000):
        """Run SA from a given initial parameter vector."""
        current = init_params.copy()
        current_val = obj_joint_smooth(current)
        best_params = current.copy()
        best_val_j = current_val
        
        T = 1.0
        T_min = 0.003
        cool = 0.9995
        np.random.seed(int(hash(tuple(init_params)) % 10000))
        
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            nb = current.copy()
            r = np.random.random()
            if r < 0.70:
                # perturb one inner hexagon
                idx = np.random.randint(0, N)
                dim = np.random.randint(0, 3)
                if dim == 2:
                    nb[idx * 3 + 2] += np.random.normal(0, 4.0)
                else:
                    nb[idx * 3 + dim] += np.random.normal(0, 0.20)
            else:
                # perturb outer
                dim = np.random.randint(N * 3, N * 3 + 3)
                if dim == N * 3 + 2:
                    nb[dim] += np.random.normal(0, 3.0)
                else:
                    nb[dim] += np.random.normal(0, 0.12)

            nv = obj_joint_smooth(nb)
            if nv < current_val or np.random.random() < math.exp(-(nv - current_val) / max(T, 1e-12)):
                current = nb
                current_val = nv
                if current_val < best_val_j:
                    best_params = current.copy()
                    best_val_j = current_val
            T *= cool
            if T < T_min:
                break
        return best_params, best_val_j

    def obj_joint_smooth(params, beta=60.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 400.0 * pen

    # Build initial params for SA
    p0_lattice = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(init_inner):
        p0_lattice[i * 3] = x
        p0_lattice[i * 3 + 1] = y
        p0_lattice[i * 3 + 2] = a
    p0_lattice[N * 3] = best_outer_center[0]
    p0_lattice[N * 3 + 1] = best_outer_center[1]
    p0_lattice[N * 3 + 2] = best_outer_angle

    # Start 1: Lattice
    best_params_sa, best_val_sa = run_sa(p0_lattice, max_iter=6000)
    
    # Start 2: Lattice + Noise
    if time.time() < deadline - 40:
        p0_noisy = p0_lattice.copy()
        np.random.seed(123)
        for i in range(N):
            p0_noisy[i*3] += np.random.normal(0, 0.1)
            p0_noisy[i*3+1] += np.random.normal(0, 0.1)
            p0_noisy[i*3+2] += np.random.normal(0, 5.0)
        best_params_sa2, best_val_sa2 = run_sa(p0_noisy, max_iter=6000)
        if best_val_sa2 < best_val_sa:
            best_params_sa = best_params_sa2
            best_val_sa = best_val_sa2

    # Extract joint result
    ji = []
    for i in range(N):
        ji.append([float(best_params_sa[i * 3]), float(best_params_sa[i * 3 + 1]), float(best_params_sa[i * 3 + 2])])
    jocx, jocy, joa = float(best_params_sa[N * 3]), float(best_params_sa[N * 3 + 1]), float(best_params_sa[N * 3 + 2])

    # Validate disjointness
    if check_disjoint(ji, eps=1e-5):
        cand_side = compute_required_side_fast(ji, (jocx, jocy), joa)
        if cand_side < best_side:
            best_inner = ji
            best_outer_center = (jocx, jocy)
            best_outer_angle = joa
            best_side = cand_side
    else:
        # Fall back to lattice (should always be valid)
        best_inner = init_inner
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)

    # ============ Phase 3: Differential Evolution for global search ============
    bounds = []
    for _ in range(N):
        bounds.append((-6.0, 6.0))
        bounds.append((-6.0, 6.0))
        bounds.append((-30.0, 30.0))
    bounds.append((-2.0, 2.0))  # outer cx
    bounds.append((-2.0, 2.0))  # outer cy
    bounds.append((-30.0, 30.0))  # outer angle

    def de_obj(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        if not check_disjoint(il, eps=1e-4):
            return best_side + 5.0  # heavy penalty if overlapping
        return compute_required_side_fast(il, (ocx, ocy), oa)

    if time.time() < deadline - 25:
        try:
            de_res = differential_evolution(de_obj, bounds, seed=7, maxiter=80,
                                            popsize=15, tol=1e-8, mutation=(0.3, 1.2),
                                            recombination=0.9, polish=False,
                                            init='sobol', update_interval=10)
            if de_res.fun < best_side:
                dp = de_res.x
                di = [[float(dp[i * 3]), float(dp[i * 3 + 1]), float(dp[i * 3 + 2])] for i in range(N)]
                docx, docy, doa = float(dp[N * 3]), float(dp[N * 3 + 1]), float(dp[N * 3 + 2])
                if check_disjoint(di, eps=1e-5):
                    best_inner = di
                    best_outer_center = (docx, docy)
                    best_outer_angle = doa
                    best_side = de_res.fun
        except Exception:
            pass

    # ============ Phase 4: Final high-precision joint polish ============
    # Polish all parameters (inner + outer)
    
    def obj_polish_joint(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        if not check_disjoint(il, eps=1e-4):
            return best_side + 10.0
        return compute_required_side_fast(il, (ocx, ocy), oa)

    # Prepare initial params for polish
    p0_polish = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(best_inner):
        p0_polish[i * 3] = x
        p0_polish[i * 3 + 1] = y
        p0_polish[i * 3 + 2] = a
    p0_polish[N * 3] = best_outer_center[0]
    p0_polish[N * 3 + 1] = best_outer_center[1]
    p0_polish[N * 3 + 2] = best_outer_angle

    if time.time() < deadline - 5:
        try:
            r = minimize(obj_polish_joint, p0_polish, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000})
            if r.fun < best_side:
                dp = r.x
                di = [[float(dp[i * 3]), float(dp[i * 3 + 1]), float(dp[i * 3 + 2])] for i in range(N)]
                docx, docy, doa = float(dp[N * 3]), float(dp[N * 3 + 1]), float(dp[N * 3 + 2])
                if check_disjoint(di, eps=1e-5):
                    best_inner = di
                    best_outer_center = (docx, docy)
                    best_outer_angle = doa
                    best_side = r.fun
        except Exception:
            pass

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
