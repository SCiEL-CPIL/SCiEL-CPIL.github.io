"""Model-and-optimize packer for 200 equal circles in a unit square.
Smooth log-sum-exp surrogate for the nonsmooth min + constrained SLSQP
solve + high-precision active-set polish. Keeps construct() interface."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200
T = 0.004  # soft-min temperature (controls sharpness of the smooth min)
EPS = 1e-9

def _elapsed(deadline):
    return time.time() >= deadline

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Largest common radius: min over pairs (d/2) and over walls."""
    diff = centers[:, None, :] - centers[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    wall = np.min(np.concatenate([
        0.5 - centers[:, 0], 0.5 + centers[:, 0],
        0.5 - centers[:, 1], 0.5 + centers[:, 1],
    ]))
    return min(min_pair, wall)

def _energy(x):
    """Smooth surrogate: log-sum-exp of -distances (pairs & walls) / T.
    Minimizing this pushes the minimum distance (and wall) up."""
    c = x.reshape(-1, 2)
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1e12)
    d = np.sqrt(d2)
    iu = np.triu_indices(N, 1)
    pair = d[iu]
    wall = np.concatenate([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
    ])
    a = -pair / T
    b = -wall / T
    m = np.concatenate([a, b])
    mmax = np.max(m)
    return T * (mmax + np.log(np.sum(np.exp(m - mmax))))

def _initialize_hexagonal():
    """Dense hexagonal lattice, clipped to the square."""
    r = 1.0 / np.sqrt(2.0 * N)  # rough spacing
    dx = r
    dy = r * np.sqrt(3.0)
    xs = np.arange(-0.5, 0.5 + 1e-9, dx)
    ys = np.arange(-0.5, 0.5 + 1e-9, dy)
    pts = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            ox = 0.0 if (j % 2 == 0) else dx / 2.0
            pts.append((x + ox, y))
    pts = np.array(pts)
    # Keep only points inside the square
    mask = (pts[:, 0] >= -0.5) & (pts[:, 0] <= 0.5) & (pts[:, 1] >= -0.5) & (pts[:, 1] <= 0.5)
    pts = pts[mask]
    if len(pts) < N:
        # pad with a jittered grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        return grid
    # take the N most "central" points (closest to origin) for better balance
    order = np.argsort(np.sum(pts**2, axis=1))
    return pts[order[:N]]

def _local_optimize(x0, deadline):
    """Constrained SLSQP: maximize r s.t. all pair & wall constraints >= r,
    with r as an auxiliary variable (epigraph)."""
    n = 2 * N
    # variables: 2N centers + 1 radius r
    def obj(v):
        # maximize r  -> minimize -r
        return -v[n]

    cons = []
    # wall constraints: each center within square with margin r
    # 0.5 - x >= r  -> 0.5 - x - r >= 0
    # x + 0.5 - r >= 0, etc.
    A = np.zeros((4 * N, n + 1))
    for i in range(N):
        # x_i
        A[4 * i, 2 * i] = -1.0;   A[4 * i, n] = -1.0; A[4 * i, -1] = 0.5
        A[4 * i + 1, 2 * i] = 1.0;  A[4 * i + 1, n] = -1.0
        # y_i
        A[4 * i + 2, 2 * i + 1] = -1.0; A[4 * i + 2, n] = -1.0; A[4 * i + 2, -1] = 0.5
        A[4 * i + 3, 2 * i + 1] = 1.0;  A[4 * i + 3, n] = -1.0
    # (A @ v + b >= 0) with b stored in last column; SLSQP uses fun(x)>=0
    def wall_fun(v):
        return A[:, :n] @ v + A[:, -1]
    cons.append({'type': 'ineq', 'fun': wall_fun})

    # pairwise: |c_i - c_j|^2 >= r^2  ->  |c_i-c_j|^2 - r^2 >= 0
    iu = np.triu_indices(N, 1)
    ii, jj = iu
    def pair_fun(v):
        c = v[:n].reshape(-1, 2)
        d2 = np.sum((c[ii] - c[jj]) ** 2, axis=1)
        return d2 - v[n] ** 2
    cons.append({'type': 'ineq', 'fun': pair_fun})

    # bounds: centers in square, r >= 0
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    v0 = np.concatenate([x0.ravel(), [_compute_radius(x0) * 0.99]])

    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 300, 'ftol': 1e-10, 'iprint': 0})
        if res.x is not None:
            return res.x[:n].reshape(-1, 2)
    except Exception:
        pass
    return x0

def _polish(centers, deadline):
    """High-precision active-set polish: repeatedly push the tightest
    (pair or wall) constraint outward with tiny steps, accept if radius up."""
    c = centers.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    step = 1e-4
    for _ in range(4000):
        if _elapsed(deadline):
            break
        diff = c[:, None, :] - c[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        pi, pj = np.unravel_index(np.argmin(d2), d2.shape)
        pair_dist = np.sqrt(d2[pi, pj])
        wall = np.concatenate([
            0.5 - c[:, 0], 0.5 + c[:, 0],
            0.5 - c[:, 1], 0.5 + c[:, 1],
        ])
        wi = int(np.argmin(wall))
        wall_dist = wall[wi]
        cur_r = min(pair_dist / 2.0, wall_dist)

        if pair_dist / 2.0 <= wall_dist:
            # push pair apart
            d = c[pi] - c[pj]
            nrm = np.linalg.norm(d)
            if nrm < 1e-12:
                d = np.array([1.0, 0.0]); nrm = 1.0
            d = d / nrm
            c[pi] += 0.5 * step * d
            c[pj] -= 0.5 * step * d
        else:
            # push wall point inward
            # determine which of 4 walls is tightest for point wi
            x, y = c[wi]
            dleft = 0.5 - x; dright = 0.5 + x
            ddown = 0.5 - y; dup = 0.5 + y
            m = min(dleft, dright, ddown, dup)
            if m == dleft:
                c[wi, 0] += step
            elif m == dright:
                c[wi, 0] -= step
            elif m == ddown:
                c[wi, 1] += step
            else:
                c[wi, 1] -= step

        c = np.clip(c, -0.5, 0.5)
        new_r = _compute_radius(c)
        if new_r > best_r + 1e-15:
            best_r = new_r
            best_c = c.copy()
        else:
            # no progress, shrink step
            step *= 0.7
            if step < 1e-9:
                break
    return best_c, best_r

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin under 150s
    best_c = None
    best_r = -1.0

    # Phase 1: L-BFGS on smooth surrogate from several inits
    inits = [_initialize_hexagonal()]
    for sd in (1, 7, 42):
        np.random.seed(sd)
        g = _initialize_hexagonal().copy()
        g += np.random.uniform(-0.01, 0.01, g.shape)
        g = np.clip(g, -0.49, 0.49)
        inits.append(g)

    for x0 in inits:
        if _elapsed(deadline):
            break
        try:
            res = minimize(_energy, x0.ravel(), method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10})
            xc = np.clip(res.x.reshape(-1, 2), -0.5, 0.5)
            r = _compute_radius(xc)
            if r > best_r:
                best_r = r
                best_c = xc.copy()
        except Exception:
            continue

    if best_c is None:
        best_c = _initialize_hexagonal()
        best_r = _compute_radius(best_c)

    # Phase 2: constrained SLSQP refine the best
    if not _elapsed(deadline):
        refined = _local_optimize(best_c, deadline)
        refined = np.clip(refined, -0.5, 0.5)
        r = _compute_radius(refined)
        if r > best_r:
            best_r = r
            best_c = refined.copy()

    # Phase 3: high-precision active-set polish
    if not _elapsed(deadline):
        best_c, best_r = _polish(best_c, deadline)

    return np.clip(best_c, -0.5, 0.5)

# EVOLVE-BLOCK-END
