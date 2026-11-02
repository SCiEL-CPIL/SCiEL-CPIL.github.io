"""Fresh packing design: honeycomb init + fast SAT + SA + Nelder-Mead polish."""
import time
import math
import numpy as np

N = 12

# ---------- geometry ----------
# Unit regular hexagon (side 1) vertices in the "flat-top" orientation.
# Circumradius = 1, inradius = sqrt(3)/2.
_COS60 = 0.5
_SIN60 = math.sqrt(3.0) / 2.0
_VERTS = np.array([
    [1.0, 0.0],
    [_COS60, _SIN60],
    [-_COS60, _SIN60],
    [-1.0, 0.0],
    [-_COS60, -_SIN60],
    [_COS60, -_SIN60],
])
# Outward unit normals for each edge (edge i connects vertex i -> i+1).
# For a CCW polygon the outward normal of an edge is (dy, -dx) normalized.
_NORMS = np.empty((6, 2))
for i in range(6):
    a = _VERTS[i]
    b = _VERTS[(i + 1) % 6]
    d = b - a
    n = np.array([d[1], -d[0]])
    n /= np.linalg.norm(n)
    _NORMS[i] = n


def _rot(v, c, s):
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _inner_verts(cx, cy, ang):
    """Vertices of a unit hexagon centered at (cx,cy) with rotation ang (rad)."""
    c = math.cos(ang)
    s = math.sin(ang)
    return (cx + c * _VERTS[:, 0] - s * _VERTS[:, 1],
            cy + s * _VERTS[:, 0] + c * _VERTS[:, 1])


def _inner_edge_normals(ang):
    """Outward edge normals for a unit hexagon rotated by ang."""
    c = math.cos(ang)
    s = math.sin(ang)
    out = np.empty((6, 2))
    for i in range(6):
        n = _NORMS[i]
        out[i] = [c * n[0] - s * n[1], s * n[0] + c * n[1]]
    return out


def _hexagon_verts(cx, cy, side, ang):
    """Vertices of a regular hexagon (side length `side`) centered at (cx,cy)."""
    c = math.cos(ang)
    s = math.sin(ang)
    return (cx + side * (c * _VERTS[:, 0] - s * _VERTS[:, 1]),
            cy + side * (s * _VERTS[:, 0] + c * _VERTS[:, 1]))


def _hexagon_edge_normals(side, ang):
    c = math.cos(ang)
    s = math.sin(ang)
    out = np.empty((6, 2))
    for i in range(6):
        n = _NORMS[i]
        out[i] = [c * n[0] - s * n[1], s * n[0] + c * n[1]]
    return out


def _separating_gap(A, B):
    """Return the minimal signed distance separating hexagon A from B.
    A = (verts(6,2), normals(6,2)); B = (verts(6,2), normals(6,2)).
    Positive value => separated by that gap (non-overlapping).
    """
    A_v, A_n = A
    B_v, B_n = B
    gap = 0.0
    # Project B's vertices onto A's outward normals.
    for i in range(6):
        n = A_n[i]
        proj = B_v @ n
        maxproj = proj.max()
        # A's support in direction n is at A_v @ n (all equal to inradius*side... use max)
        a_support = (A_v @ n).max()
        # gap in direction n: if maxproj <= a_support -> separated by (a_support - maxproj)
        g = a_support - maxproj
        if g > gap:
            gap = g
    for i in range(6):
        n = B_n[i]
        proj = A_v @ n
        maxproj = proj.max()
        b_support = (B_v @ n).max()
        g = b_support - maxproj
        if g > gap:
            gap = g
    return gap


def _pair_gap(A, B):
    return _separating_gap(A, B)


def _inner_contained_gap(hexA, outer):
    """Gap such that hexA is inside outer when gap >= 0.
    outer = (verts, normals); hexA = (verts, normals).
    For hexA to be inside outer, for every outward normal n of outer,
    all vertices of hexA must lie on or below outer's support in n.
    gap = min over n of (outer_support(n) - max_vertex_of_A_in_n).
    """
    O_v, O_n = outer
    A_v, A_n = hexA
    gap = float('inf')
    for i in range(6):
        n = O_n[i]
        o_support = (O_v @ n).max()
        a_max = (A_v @ n).max()
        g = o_support - a_max
        if g < gap:
            gap = g
    return gap


# ---------- objective ----------
def _evaluate(x, y, ang, side, oang):
    """Return (violation, side) where violation = max(0, needed_side - side).
    x,y,ang are 1D arrays of length N.
    """
    # Precompute inner hexagon geometry.
    inners = []
    for k in range(N):
        vx, vy = _inner_verts(x[k], y[k], ang[k])
        nn = _inner_edge_normals(ang[k])
        inners.append((np.column_stack((vx, vy)), nn))
    # Outer hexagon geometry.
    ovx, ovy = _hexagon_verts(0.0, 0.0, side, oang)
    onn = _hexagon_edge_normals(side, oang)
    outer = (np.column_stack((ovx, ovy)), onn)

    need = 0.0
    # Containment of each inner in outer.
    for k in range(N):
        g = _inner_contained_gap(inners[k], outer)
        # If g < 0, inner pokes out; need to enlarge side.
        # Required extra: the deficit g is a distance; enlarging side by d
        # increases inradius by d, so we need d >= -g.
        if g < 0.0:
            need = max(need, -g)
    # Pairwise non-overlap.
    for i in range(N):
        for j in range(i + 1, N):
            g = _pair_gap(inners[i], inners[j])
            if g < 0.0:
                need = max(need, -g)
    return need, side


def _objective(params):
    """params: array of length 3N+1 = [x0..xN, y0..yN, ang0..angN, side]."""
    x = params[0:N]
    y = params[N:2 * N]
    ang = params[2 * N:3 * N]
    side = params[3 * N]
    need, s = _evaluate(x, y, ang, s, 0.0)
    # Objective: minimize side, penalize violation heavily.
    return side + 1000.0 * need


# ---------- initial configuration ----------
def _honeycomb_init():
    """Honeycomb arrangement of 12 unit hexagons around a central hole."""
    # Distance between centers of adjacent hexagons in a honeycomb = 2*inradius = sqrt(3).
    d = math.sqrt(3.0)
    pts = []
    # Ring of 6 around center at radius d.
    for k in range(6):
        a = math.pi / 3.0 * k
        pts.append((d * math.cos(a), d * math.sin(a)))
    # Second ring: 6 more at radius 2*d but offset, choose those that don't overlap.
    # Standard honeycomb: 6 at radius d (angles 0,60,...), 6 at radius 2d (angles 30,90,...)
    for k in range(6):
        a = math.pi / 3.0 * k + math.pi / 6.0
        pts.append((2 * d * math.cos(a), 2 * d * math.sin(a)))
    # This gives 12 points. Angles all 0.
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    ang = np.zeros(N)
    # Estimate initial side: bounding of points + inradius.
    rmax = np.sqrt(x ** 2 + y ** 2).max()
    side0 = rmax + 1.0 + 0.5
    return np.concatenate([x, y, ang, [side0]])


# ---------- simulated annealing ----------
def _simulated_annealing(params0, deadline):
    rng = np.random.default_rng(12345)
    params = params0.copy()
    f = _objective(params)
    best = params.copy()
    best_f = f
    T = 1.0
    # Adaptive: start with a step that produces moderate violations.
    step = 0.5
    n_iter = 0
    while time.time() < deadline:
        # Perturb a random subset.
        n_perturb = rng.integers(1, 4)
        idx = rng.integers(0, len(params), size=n_perturb)
        new = params.copy()
        for i in idx:
            new[i] += rng.normal(0.0, step)
        # Keep side positive and angles bounded.
        new[3 * N] = max(0.5, new[3 * N])
        new[2 * N:3 * N] = np.mod(new[2 * N:3 * N], 2 * math.pi)
        f_new = _objective(new)
        if f_new < f or rng.random() < math.exp(-(f_new - f) / max(T, 1e-9)):
            params = new
            f = f_new
            if f < best_f:
                best_f = f
                best = params.copy()
        else:
            # occasional larger step if stuck
            pass
        # Cool down
        T *= 0.9995
        n_iter += 1
        # Adapt step: if T too small, reduce step.
        if T < 1e-3:
            step *= 0.9
    return best, best_f


# ---------- Nelder-Mead polish ----------
def _nelder_mead_polish(params, deadline, max_iter=2000):
    """Simple Nelder-Mead on the objective."""
    n = len(params)
    # Build initial simplex.
    simplex = [params.copy()]
    for i in range(n):
        p = params.copy()
        delta = 0.1 if i < 3 * N else 0.05
        p[i] += delta
        simplex.append(p)
    fvals = [_objective(p) for p in simplex]
    # Order.
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    it = 0
    while time.time() < deadline and it < max_iter:
        # Centroid of all but worst.
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        f_worst = fvals[-1]
        # Reflection.
        x_r = centroid + alpha * (centroid - worst)
        f_r = _objective(x_r)
        if f_r < fvals[0]:
            # Expansion.
            x_e = centroid + gamma * (centroid - worst)
            f_e = _objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction.
            x_c = centroid + rho * (worst - centroid)
            f_c = _objective(x_c)
            if f_c < f_worst:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink.
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = _objective(simplex[i])
        # Reorder.
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        it += 1
    return simplex[0], fvals[0]


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill.

    # Initial configuration.
    params = _honeycomb_init()

    # Quick local polish to get a decent baseline.
    params, f = _nelder_mead_polish(params, min(time.time() + 20.0, deadline - 100.0), max_iter=500)

    # Simulated annealing with time budget.
    sa_deadline = deadline - 60.0  # reserve 60s for polish.
    if time.time() < sa_deadline:
        params, f = _simulated_annealing(params, sa_deadline)

    # Final high-precision polish.
    polish_deadline = deadline - 5.0
    params, f = _nelder_mead_polish(params, polish_deadline, max_iter=5000)

    # Extract and return.
    x = params[0:N]
    y = params[N:2 * N]
    ang = params[2 * N:3 * N]
    side = params[3 * N]

    # Convert angles to degrees.
    ang_deg = np.degrees(ang) % 360.0

    inner = [[float(x[i]), float(y[i]), float(ang_deg[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(side), 0.0
