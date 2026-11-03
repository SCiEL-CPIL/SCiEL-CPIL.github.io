"""Circle packing in unit square: 26 circles, maximize sum of radii.

Strategy: reduce search space to 52 center coordinates. Radii are derived
from the Voronoi / nearest-neighbor geometry:
    r_i = min( dist_to_boundary_i,  min_{j != i} dist(c_i, c_j) / 2 )
which guarantees non-overlapping circles by construction. We maximize the
sum of these radii over the center positions using a smooth log-sum-exp
(soft-min) surrogate with an analytic gradient, then polish with a
coordinate-descent / local-search pass.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 142.0  # safety margin before the 150 s wall-clock kill

# EVOLVE-BLOCK-START


def _radii_from_centers(c, soft=False, beta=None):
    """Compute maximal non-overlapping radii for centers c (n,2).

    Exact (soft=False): r_i = min(boundary_i, min_j d_ij/2).
    Smooth (soft=True): soft-min via log-sum-exp with scale beta.
    Returns (r, boundary, half_min_dist).
    """
    n = c.shape[0]
    x = c[:, 0]
    y = c[:, 1]
    # distance to each wall
    boundary = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    # pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    half_min = np.min(d, axis=1) / 2.0
    if not soft:
        r = np.minimum(boundary, half_min)
    else:
        # soft-min of the two "constraint values"
        # softmin(a,b) = -1/beta * log(exp(-beta a)+exp(-beta b))
        a = boundary
        b = half_min
        m = np.minimum(a, b)
        r = m - (1.0 / beta) * np.log(np.exp(-beta * (a - m)) + np.exp(-beta * (b - m)))
    return np.maximum(r, 0.0), boundary, half_min


def _objective(c, beta, penalty=0.0):
    """Objective value = -sum(r_i) (minimize) plus soft boundary penalty.
    c is flat (2n,) of centers."""
    c2 = c.reshape(N, 2)
    r, boundary, _ = _radii_from_centers(c2, soft=True, beta=beta)
    # ensure positive radius contribution
    val = -np.sum(r)
    if penalty > 0.0:
        # soft boundary: penalize if a circle would exceed its boundary limit
        # r already <= boundary by construction, so this is 0; keep for safety
        pass
    return val


def _gradient(c, beta):
    """Analytic gradient of -softmin(boundary_i, halfmin_i) w.r.t. centers.

    softmin(a,b) = m - (1/beta) log( e^{-beta(a-m)} + e^{-beta(b-m)} )
    where m=min(a,b).
    Let A = e^{-beta(a-m)}, B = e^{-beta(b-m)}.
    d softmin/d a = -A/(A+B) * (-beta)/beta ... derive carefully.
    softmin = m - (1/beta) log(A+B)
    d/da: dm/da (1 if a is min else 0) - (1/beta)*(1/(A+B))*dA/da
    dA/da = -beta A  (since a-m term derivative is 1 when a is min, else 0)
    So d/da softmin = (dm/da) + A/(A+B)   [when a is min: 1 + A/(A+B)]
    Actually when a is min, m=a, dm/da=1, and (a-m)=0 so A=1.
    Let's just compute numerically-safe using weights.
    """
    n = N
    c2 = c.reshape(N, 2)
    x = c2[:, 0]
    y = c2[:, 1]
    grad = np.zeros_like(c)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy + 1e-14)
    np.fill_diagonal(d, np.inf)

    boundary = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    half_min = np.min(d, axis=1) / 2.0

    a = boundary
    b = half_min
    m = np.minimum(a, b)
    A = np.exp(-beta * (a - m))
    B = np.exp(-beta * (b - m))
    S = A + B

    # weight on 'a' (boundary) and 'b' (pairwise) in the soft-min
    # d softmin/da = 1_{a<m? } ... use: d softmin/da = A/(A+B) (with sign +)
    # Because softmin decreases when a increases? softmin(a,b) is min-like,
    # increasing a (the larger one) doesn't change min; increasing the smaller
    # (a when a<b) increases softmin. d softmin/da = A/(A+B) where A is weight
    # of a. Since softmin ~ a when a<b, d/da ~1 = A/S when A>>B. Good.
    wa = A / S   # weight for boundary term
    wb = B / S   # weight for pairwise term

    # gradient of -softmin w.r.t. x_i,y_i
    # d(-softmin)/dx = -wa * d(boundary)/dx - wb * d(half_min)/dx
    # boundary gradient:
    dbx = np.zeros(n)
    dby = np.zeros(n)
    # boundary = min(x,1-x,y,1-y); subgradient: pick the active wall
    # For smoothness use soft handling: we treat boundary gradient as the
    # gradient of the currently-minimizing wall.
    walls = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
    active = np.argmin(walls, axis=1)
    for i in range(n):
        if active[i] == 0:
            dbx[i] = 1.0
        elif active[i] == 1:
            dbx[i] = -1.0
        elif active[i] == 2:
            dby[i] = 1.0
        else:
            dby[i] = -1.0

    # half_min gradient: d/dx_i (min_j d_ij/2) = (1/2) * (x_i - x_j*)/d_ij*
    # where j* is argmin
    jstar = np.argmin(d, axis=1)
    dhx = np.zeros(n)
    dhy = np.zeros(n)
    for i in range(n):
        j = jstar[i]
        dij = d[i, j]
        if dij > 1e-12:
            dhx[i] = 0.5 * (x[i] - x[j]) / dij
            dhy[i] = 0.5 * (y[i] - y[j]) / dij

    # assemble: grad of objective = -[ wa*db + wb*dh ]
    gx = -(wa * dbx + wb * dhx)
    gy = -(wa * dby + wb * dhy)
    grad[0::2] = gx
    grad[1::2] = gy
    return grad


def _validate(xyz):
    """Check configuration validity: inside unit square, non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True


def _enforce_validity(xyz):
    """Force a configuration to be valid by clipping positions and shrinking
    radii to the true geometric maximum (Voronoi-limited)."""
    p = xyz.copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # set each radius to the exact geometric maximum for its center
    for _ in range(3):
        r, boundary, _ = _radii_from_centers(p[:, :2], soft=False)
        p[:, 2] = np.maximum(r, 1e-6)
    return p


def _initial_hex():
    """Hexagonal lattice initial centers, scaled into the unit square."""
    r0 = 0.13
    centers = []
    for row in range(12):
        for col in range(12):
            x = col * 2.0 * r0
            y = row * np.sqrt(3.0) * r0
            if row % 2 == 1:
                x += r0
            centers.append((x, y))
    centers.sort(key=lambda p: (p[0] - 0.5) ** 2 + (p[1] - 0.5) ** 2)
    centers = centers[:N]
    xs = np.array([c[0] for c in centers])
    ys = np.array([c[1] for c in centers])
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    scale = min(0.8 / max(maxx - minx, 1e-9), 0.8 / max(maxy - miny, 1e-9))
    c = np.empty((N, 2))
    for i, (x, y) in enumerate(centers):
        c[i, 0] = (x - minx) * scale + 0.1
        c[i, 1] = (y - miny) * scale + 0.1
    return c


def _initial_grid():
    """Square grid initial centers."""
    side = int(np.ceil(np.sqrt(N)))
    c = np.empty((N, 2))
    k = 0
    for i in range(side):
        for j in range(side):
            if k >= N:
                break
            c[k, 0] = (i + 0.5) / side
            c[k, 1] = (j + 0.5) / side
            k += 1
    return c


def _local_search(c, beta, time_budget):
    """Refine centers with L-BFGS-B on the smooth surrogate, then a
    coordinate-descent polish on the exact objective."""
    start = time.time()
    params = c.flatten().copy()
    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = params.copy()
    best_val = -np.sum(_radii_from_centers(best_params.reshape(N, 2), soft=False)[0])

    # L-BFGS-B on smooth surrogate
    try:
        res = minimize(_objective, best_params, jac=_gradient, args=(beta,),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9})
        v = -np.sum(_radii_from_centers(res.x.reshape(N, 2), soft=False)[0])
        if v < best_val:
            best_params = res.x.copy()
            best_val = v
    except Exception:
        pass

    # Coordinate-descent polish on exact objective
    def exact_val(p):
        return -np.sum(_radii_from_centers(p.reshape(N, 2), soft=False)[0])

    for _ in range(400):
        if time.time() - start > time_budget:
            break
        improved = False
        for i in range(N):
            for axis in range(2):
                base = best_params.copy()
                lo, hi = best_params[i * 2 + axis] - 0.05, best_params[i * 2 + axis] + 0.05
                lo = max(lo, 0.0)
                hi = min(hi, 1.0)
                # golden-section-ish: sample a few points
                for frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    trial = best_params.copy()
                    trial[i * 2 + axis] = lo + frac * (hi - lo)
                    if exact_val(trial) < best_val - 1e-12:
                        best_params = trial
                        best_val = exact_val(trial)
                        improved = True
        if not improved:
            break

    return best_params.reshape(N, 2), best_val


def _build_xyz(c):
    r, _, _ = _radii_from_centers(c, soft=False)
    xyz = np.empty((N, 3))
    xyz[:, 0] = c[:, 0]
    xyz[:, 1] = c[:, 1]
    xyz[:, 2] = np.maximum(r, 1e-6)
    return xyz


def construct():
    global_start = time.time()

    best_x = None
    best_score = -np.inf

    inits = [_initial_hex(), _initial_grid()]
    # add a jittered hex restart for diversity
    rng = np.random.default_rng(12345)
    jittered = _initial_hex() + 0.02 * rng.standard_normal((N, 2))
    jittered = np.clip(jittered, 0.05, 0.95)
    inits.append(jittered)

    for c0 in inits:
        if time.time() - global_start > DEADLINE - 20.0:
            break
        budget = min(40.0, DEADLINE - (time.time() - global_start) - 15.0)
        if budget < 5.0:
            break
        c_opt, val = _local_search(c0, beta=8.0, time_budget=budget)
        xyz = _build_xyz(c_opt)
        xyz = _enforce_validity(xyz)
        score = np.sum(xyz[:, 2])
        if _validate(xyz) and score > best_score:
            best_score = score
            best_x = xyz.copy()

    # fallback if nothing valid
    if best_x is None or not _validate(best_x):
        c0 = _initial_hex()
        xyz = _build_xyz(c0)
        xyz = _enforce_validity(xyz)
        if not _validate(xyz):
            # absolute fallback: tiny circles on a grid
            side = int(np.ceil(np.sqrt(N)))
            xyz = np.empty((N, 3))
            k = 0
            for i in range(side):
                for j in range(side):
                    if k >= N:
                        break
                    xyz[k] = [(i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.5]
                    k += 1
        best_x = xyz

    return best_x


# EVOLVE-BLOCK-END
