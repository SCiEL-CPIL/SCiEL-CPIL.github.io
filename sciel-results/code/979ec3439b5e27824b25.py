import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 6.0
EPS = 1e-9
HPI = math.pi / 6.0          # hexagon periodicity for beta
HPI6 = math.pi / 3.0         # 60 deg for individual hexagon rotation

# Fixed orientation of the 6 inner hexagons (pointy-top, matches the seed's lattice)
# Unit hexagon vertices relative to center at angle 0 (flat-top style used by axial_to_cartesian)
def _hex_verts(angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # base pointy-top unit hexagon vertices
    base = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2],
    ])
    R = np.array([[ca, -sa], [sa, ca]])
    return base @ R.T

def _outer_normals(beta):
    out = []
    for k in range(6):
        ang = beta + math.pi/6.0 + k * math.pi/3.0
        out.append((math.cos(ang), math.sin(ang)))
    return np.array(out)

def _enclosing_S(pts, verts_list, beta, cx, cy):
    """Exact enclosing regular-hexagon side length for given centers/verts/beta/center."""
    normals = _outer_normals(beta)
    maxD = 0.0
    for verts in verts_list:
        # vertices in world coords
        p = pts + verts
        proj = (p[:,0]-cx)*normals[:,0] + (p[:,1]-cy)*normals[:,1]
        d = max(float(np.max(proj)), float(-np.min(proj)))
        if d > maxD:
            maxD = d
    return (maxD + APOTHEM) / APOTHEM

def _smooth_S(pts, verts_list, beta, cx, cy, t):
    """Smooth (log-sum-exp) surrogate of enclosing side length."""
    normals = _outer_normals(beta)
    nrm = np.linalg.norm(normals, axis=1)
    normals = normals / nrm[:, None]
    allproj = []
    for verts in verts_list:
        p = pts + verts
        proj = (p[:,0]-cx)*normals[:,0] + (p[:,1]-cy)*normals[:,1]   # (6, nverts)
        allproj.append(proj.ravel())
    allproj = np.concatenate(allproj)
    neg = -allproj
    allproj = np.concatenate([allproj, neg])
    m = float(np.max(allproj))
    if m > 1e6:
        return 1e6
    sh = (allproj - m) / t
    sh = np.clip(sh, -50, 50)
    D = t * (math.log(float(np.sum(np.exp(sh)))) + m / t)
    return (D + APOTHEM) / APOTHEM

def _time_ok():
    return (time.time() - START_TIME) < (TIME_LIMIT - SAFETY_MARGIN)

def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Fixed orientation for all inner hexagons (matches seed lattice; rotation freedom
    # is still explored by the global beta). This keeps the problem well-conditioned
    # and guarantees the known-good lattice packing is reachable as an optimum.
    verts_list = [_hex_verts(0.0)] * N

    # Build initial configurations: a few good axial packings + random starts.
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    axial_sets = [
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
    ]

    starts = []
    for ax in axial_sets:
        p = axial_to_cartesian(ax)
        # parameterize: x0..x10, y0..y10, beta  (centers + global rotation)
        par = np.concatenate([p[:,0], p[:,1], [0.0]])
        starts.append(par)

    # random starts (perturbed lattice + random)
    rng = np.random.default_rng(12345)
    for _ in range(8):
        ax = axial_sets[rng.integers(0, len(axial_sets))]
        p = axial_to_cartesian(ax)
        p = p + rng.normal(0, 0.3, size=p.shape)
        par = np.concatenate([p[:,0], p[:,1], [rng.uniform(0, HPI)]])
        starts.append(par)
    for _ in range(8):
        p = rng.uniform(-2.5, 2.5, size=(N,2))
        par = np.concatenate([p[:,0], p[:,1], [rng.uniform(0, HPI)]])
        starts.append(par)

    def unpack(par):
        cx_arr = par[0:N]
        cy_arr = par[N:2*N]
        beta = par[2*N]
        beta = ((beta % HPI) + HPI) % HPI
        return np.column_stack([cx_arr, cy_arr]), beta

    def objective(par, t):
        pts, beta = unpack(par)
        return _smooth_S(pts, verts_list, beta, 0.0, 0.0, t)

    def objective_centered(par, t):
        # include center as free variables for extra robustness
        # par = [x0..x10, y0..y10, beta, cx, cy]
        cx_arr = par[0:N]; cy_arr = par[N:2*N]; beta = par[2*N]
        cx = par[2*N+1]; cy = par[2*N+2]
        beta = ((beta % HPI) + HPI) % HPI
        pts = np.column_stack([cx_arr, cy_arr])
        return _smooth_S(pts, verts_list, beta, cx, cy, t)

    best_S = float('inf')
    best_par = None

    # Multi-start with annealing temperatures (global phase) then exact polish.
    temps = [0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.008, 0.004, 0.002]
    for t in temps:
        if not _time_ok():
            break
        # number of starts we can afford shrinks as we go; keep all early, fewer late
        nstarts = len(starts)
        for i in range(nstarts):
            if not _time_ok():
                break
            par0 = starts[i]
            if best_par is not None and i > 0:
                # occasionally restart near the current best to refine
                if i % 3 == 0:
                    par0 = best_par + rng.normal(0, 0.05, size=best_par.shape)
            try:
                res = minimize(objective, par0, args=(t,), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12,
                                        'adaptive': True})
            except Exception:
                continue
            if res.fun < 1e5:
                # evaluate exact objective at this point
                pts, beta = unpack(res.x)
                S_exact = _enclosing_S(pts, verts_list, beta, 0.0, 0.0)
                if S_exact < best_S:
                    best_S = S_exact
                    best_par = res.x.copy()

    # Exact polish with free center (cx, cy) and beta, multi-start near best.
    if best_par is not None and _time_ok():
        def exact_centered(par):
            cx_arr = par[0:N]; cy_arr = par[N:2*N]; beta = par[2*N]
            cx = par[2*N+1]; cy = par[2*N+2]
            beta = ((beta % HPI) + HPI) % HPI
            pts = np.column_stack([cx_arr, cy_arr])
            return _enclosing_S(pts, verts_list, beta, cx, cy)

        # build centered parameterization from best_par
        pts_b, beta_b = unpack(best_par)
        par_c = np.concatenate([pts_b[:,0], pts_b[:,1], [beta_b], [0.0, 0.0]])
        # first find good center via LP-like: just start at centroid
        par_c[2*N+1] = float(np.mean(pts_b[:,0]))
        par_c[2*N+2] = float(np.mean(pts_b[:,1]))

        polish_starts = [par_c.copy()]
        for dx in (-0.05,0.05):
            for dy in (-0.05,0.05):
                p = par_c.copy(); p[2*N+1]+=dx; p[2*N+2]+=dy; polish_starts.append(p)
        for db in (-0.02,0.02,-0.01,0.01):
            p = par_c.copy(); p[2*N]+=db; polish_starts.append(p)

        for p0 in polish_starts:
            if not _time_ok():
                break
            try:
                res = minimize(exact_centered, p0, method='Nelder-Mead',
                               options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-16,
                                        'adaptive': True})
            except Exception:
                continue
            if res.fun < best_S:
                best_S = res.fun
                best_par = res.x.copy()

    # Assemble output
    if best_par is None:
        # fallback: first axial set, centered
        p = axial_to_cartesian(axial_sets[0])
        cx = float(np.mean(p[:,0])); cy = float(np.mean(p[:,1]))
        beta = 0.0
        S = _enclosing_S(p, verts_list, beta, cx, cy)
        inner = [[float(x), float(y), 0.0] for (x,y) in p]
        return inner, [cx, cy], float(S), 0.0

    # best_par is either 2N+1 (no center) or 2N+3 (with center)
    if best_par.shape[0] >= 2*N+3:
        cx_arr = best_par[0:N]; cy_arr = best_par[N:2*N]
        beta = ((best_par[2*N] % HPI) + HPI) % HPI
        cx = float(best_par[2*N+1]); cy = float(best_par[2*N+2])
    else:
        cx_arr = best_par[0:N]; cy_arr = best_par[N:2*N]
        beta = ((best_par[2*N] % HPI) + HPI) % HPI
        cx = float(np.mean(cx_arr)); cy = float(np.mean(cy_arr))

    pts = np.column_stack([cx_arr, cy_arr])
    S = _enclosing_S(pts, verts_list, beta, cx, cy)

    inner = []
    for i in range(N):
        inner.append([float(cx_arr[i]), float(cy_arr[i]), 0.0])

    return inner, [float(cx), float(cy)], float(S), float(math.degrees(beta))
