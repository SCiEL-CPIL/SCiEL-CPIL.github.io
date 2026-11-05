import math
import time
import numpy as np

# EVOLVE-BLOCK-START

_N = 11
_DEADLINE = 110.0  # seconds; leave margin under the 150s kill

# --- Geometry helpers -------------------------------------------------------

# Vertex directions of a unit regular hexagon at angle 0 (pointy-top: vertices at 0,60,...)
_HEX_DIRS = np.array([np.cos(np.deg2rad(a)) for a in range(0, 360, 60)])
_HEX_DIRY = np.array([np.sin(np.deg2rad(a)) for a in range(0, 360, 60)])

# The 6 outward normal directions of a flat-side hexagon at rotation psi.
# A regular hexagon's sides have outward normals at psi + 30 + 60*k.
def outer_normals(psi):
    return np.array([psi + np.deg2rad(30 + 60 * k) for k in range(6)])


def support_unit_hex(n_angle, phi):
    """Support of a unit hexagon (side 1) at rotation phi in direction n_angle.
    = max over 6 vertex directions of cos(n - (phi + 60k))."""
    diff = n_angle - (phi + np.deg2rad(np.arange(0, 360, 60)))
    return float(np.max(np.cos(diff)))


def dmin_table():
    """Precompute d_min(theta) for theta in [0,60) degrees via half-plane method."""
    thetas = np.linspace(0, np.deg2rad(60), 1201)
    vals = np.empty_like(thetas)
    # fixed hexagon A at angle 0; hexagon B at angle theta
    for i, th in enumerate(thetas):
        # B's vertex directions
        bdir = th + np.deg2rad(np.arange(0, 360, 60))
        # A's outward normals (6)
        anorm = np.deg2rad(30 + np.arange(0, 360, 60))
        # For each of A's 6 half-planes, required separation = sup_B - inf_A
        # sup of B in direction n = max_j cos(n - bdir_j)
        # inf of A in direction n = min_k cos(n - a0_k) where a0 = 0..
        a0 = np.deg2rad(np.arange(0, 360, 60))
        req = np.empty(6)
        for m in range(6):
            n = anorm[m]
            supB = np.max(np.cos(n - bdir))
            infA = np.min(np.cos(n - a0))
            req[m] = supB - infA
        vals[i] = np.max(req)
    return thetas, vals


_DMIN_TH, _DMIN_VAL = dmin_table()


def dmin(theta_deg):
    """Minimum center distance between two unit hexagons with relative angle theta_deg."""
    t = theta_deg % 60.0
    idx = int(np.clip(np.round(t / 60.0 * (len(_DMIN_TH) - 1)), 0, len(_DMIN_TH) - 1))
    return float(_DMIN_VAL[idx])


# --- Containment objective --------------------------------------------------

def outer_side_for(inner_pts, inner_ang, cx, cy, L, psi):
    """Return the minimal outer side length needed at given (cx,cy,psi) to contain
    all inner hexagons (positions inner_pts, rotations inner_ang in radians).
    Actually we want to *optimize* L; this computes required L for fixed psi,cx,cy."""
    norms = outer_normals(psi)
    req = 0.0
    for (px, py), phi in zip(inner_pts, inner_ang):
        for n in norms:
            nx, ny = math.cos(n), math.sin(n)
            # support of inner hex in direction n
            sup = support_unit_hex(n, phi)
            # distance from outer center to this half-plane along n
            dist = (px - cx) * nx + (py - cy) * ny
            need = dist + sup
            if need > req:
                req = need
    return req


def cost_outer(params, inner_pts, inner_ang):
    """params = [cx, cy, L, psi]. Objective = L + penalty if any hex escapes."""
    cx, cy, L, psi = params
    # required side length at this orientation/center
    req = outer_side_for(inner_pts, inner_ang, cx, cy, L, psi)
    # We want L >= req; objective encourages small L but penalizes violation.
    # Smooth: use softplus on (req - L) plus L.
    viol = req - L
    soft = 100.0 * math.log(1.0 + math.exp(min(viol, 30.0) * 10.0))
    return L + soft


# --- Overlap penalty for inner cluster --------------------------------------

def overlap_penalty(pts, angs):
    """Penalty for pairs of inner hexagons that are too close (hard-ish)."""
    n = len(pts)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d = math.hypot(dx, dy)
            rel = (angs[i] - angs[j]) * 180.0 / math.pi
            dm = dmin(rel)
            if d < dm:
                pen += 1000.0 * (dm - d) ** 2
    return pen


# --- Initial hexagonal cluster ---------------------------------------------

def initial_cluster():
    """11 hexagons in a hexagonal arrangement (1 center + 2 rings), side 1."""
    # Axial coordinates for hex radius 2 (12 cells) minus one? We need 11.
    # Standard "hex of side 2" has 1+6+12=19. We want 11: use center + 6 + 4 of ring2? 
    # Simpler: center + 6 neighbors (ring1) + 4 extra placed in gaps.
    # Actually a compact 11-hex cluster: use the 11 cells of a "hexagon of side 2" 
    # is 19. Instead use a 3x3-ish block. Let's use axial coords within radius 1 
    # (7 cells) plus 4 more.
    # For a clean start, use 7 (radius-1 hex) + 4 = 11, placed to be compact.
    pts = []
    angs = []
    # radius-1 hexagon (7 cells)
    axial = [(0, 0)]
    for k in range(6):
        a = np.deg2rad(60 * k)
        axial.append((math.cos(a), math.sin(a)))  # unit axial -> cartesian approx
    # Convert axial (cube) to cartesian for pointy-top: x = (q + r/2)*sqrt(3), y = r*1.5
    # Use a proper axial set for radius 1:
    axial_r1 = [(0,0), (1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    def axial_to_xy(q, r):
        return (math.sqrt(3) * (q + r / 2.0), 1.5 * r)
    for (q, r) in axial_r1:
        x, y = axial_to_xy(q, r)
        pts.append([x, y])
        angs.append(0.0)
    # 4 more cells: extend ring 2 partially to keep compact (pick 4 of the 12 ring-2 cells)
    # ring-2 axial coords (radius 2)
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                ring2.append((q, r))
    # pick 4 that are "aligned" to reduce bounding box: choose those with small |y|
    ring2.sort(key=lambda t: abs(1.5 * t[1]))
    for (q, r) in ring2[:4]:
        x, y = axial_to_xy(q, r)
        pts.append([x, y])
        angs.append(0.0)
    return pts, angs


# --- Optimization -----------------------------------------------------------

def optimize_outer(pts, angs, t0):
    """Optimize outer hexagon (cx,cy,L,psi) for fixed inner cluster."""
    from scipy.optimize import minimize
    # Initial guess: center at centroid, L large, psi=0
    cx0 = float(np.mean([p[0] for p in pts]))
    cy0 = float(np.mean([p[1] for p in pts]))
    # rough bounding: max extent
    ext = max(max(abs(p[0] - cx0) for p in pts) + 2.0,
              max(abs(p[1] - cy0) for p in pts) + 2.0)
    L0 = ext
    x0 = np.array([cx0, cy0, L0, 0.0])
    bounds = [(-50, 50), (-50, 50), (2.0, 60.0), (-np.pi / 2, np.pi / 2)]
    best = None
    best_val = float('inf')
    def obj(x):
        if time.time() - t0 > _DEADLINE - 5:
            raise KeyboardInterrupt
        return cost_outer(x, pts, angs)
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 400, 'maxfun': 2000})
        if time.time() - t0 < _DEADLINE - 5:
            v = obj(res.x)
            if v < best_val:
                best_val = v
                best = res.x
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    if best is None:
        best = x0
    return best


def refine_inner(pts, angs, outer, t0):
    """Light refinement of inner cluster positions + angles, keeping outer fixed."""
    from scipy.optimize import minimize
    cx, cy, L, psi = outer
    n = len(pts)
    def flat(x):
        p = []
        a = []
        for i in range(n):
            p.append([x[2 * i], x[2 * i + 1]])
            a.append(x[2 * n + i])
        return p, a
    def obj(x):
        if time.time() - t0 > _DEADLINE - 3:
            raise KeyboardInterrupt
        p, a = flat(x)
        # required outer side at current outer orientation/center
        req = outer_side_for(p, a, cx, cy, L, psi)
        # objective: minimize req (the side needed) + overlap penalty
        pen = overlap_penalty(p, a)
        return req + pen
    x0 = np.array([c for pt in pts for c in pt] + list(angs))
    bounds = [(-30, 30)] * (2 * n) + [(-np.pi / 2, np.pi / 2)] * n
    best = x0
    best_val = obj(x0)
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'maxfun': 1000})
        if time.time() - t0 < _DEADLINE - 3:
            v = obj(res.x)
            if v < best_val:
                best_val = v
                best = res.x
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    return flat(best)


def best_outer_for(pts, angs, t0):
    """Re-optimize outer after inner refinement, return (outer_params, side)."""
    outer = optimize_outer(pts, angs, t0)
    cx, cy, L, psi = outer
    req = outer_side_for(pts, angs, cx, cy, L, psi)
    # final side is max(L, req) to be safe
    side = max(L, req)
    return outer, side


def construct():
    t0 = time.time()
    pts, angs = initial_cluster()
    # First outer fit
    outer1, side1 = best_outer_for(pts, angs, t0)
    best_side = side1
    best_pts, best_angs, best_outer = pts, angs, outer1
    if time.time() - t0 < _DEADLINE - 10:
        # Refine inner
        pts2, angs2 = refine_inner(pts, angs, outer1, t0)
        outer2, side2 = best_outer_for(pts2, angs2, t0)
        if side2 < best_side:
            best_side = side2
            best_pts, best_angs, best_outer = pts2, angs2, outer2
    # Build return: inner rows (x, y, angle_degrees)
    inner = []
    for (x, y), a in zip(best_pts, best_angs):
        inner.append([float(x), float(y), float(math.degrees(a))])
    cx, cy, L, psi = best_outer
    return inner, (float(cx), float(cy)), float(best_side), float(math.degrees(psi))

# EVOLVE-BLOCK-END
