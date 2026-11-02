"""Heilbronn 11-point max-min triangle area: warm start + Nelder-Mead refinement."""
import time
import itertools
import numpy as np

N = 11
SQ3 = np.sqrt(3.0)

# Triangle vertices
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQ3 / 2.0]])


def _barycentric(P):
    """Convert cartesian P (n,2) to barycentric (n,3) w.r.t. V."""
    # Solve P = a*V0 + b*V1 + c*V2, a+b+c=1
    # Use the standard formula with edge vectors
    v0 = V[0]
    v1 = V[1]
    v2 = V[2]
    d00 = np.dot(v1 - v0, v1 - v0)
    d01 = np.dot(v1 - v0, v2 - v0)
    d11 = np.dot(v2 - v0, v2 - v0)
    d0011 = d00 * d11 - d01 * d01
    P = np.asarray(P, dtype=float)
    rel = P - v0
    b = (d11 * np.einsum('ij,j->i', rel, v1 - v0) - d01 * np.einsum('ij,j->i', rel, v2 - v0)) / d0011
    c = (d00 * np.einsum('ij,j->i', rel, v2 - v0) - d01 * np.einsum('ij,j->i', rel, v1 - v0)) / d0011
    a = 1.0 - b - c
    return np.stack([a, b, c], axis=1)


def _cartesian(B):
    """Convert barycentric B (n,3) to cartesian (n,2)."""
    B = np.asarray(B, dtype=float)
    return B[:, 0] * V[0] + B[:, 1] * V[1] + B[:, 2] * V[2]


def _min_area(P):
    """Minimum triangle area among all triples of P (n,2)."""
    n = len(P)
    best = np.inf
    # Vectorize over pairs: area of (P[i],P[j],P[k])
    # For each pair (i,j), cross with all k, take min abs over k != i,j
    idx = np.arange(n)
    # Compute all pairwise differences
    # area(i,j,k) = 0.5 * | (P[j]-P[i]) x (P[k]-P[i]) |
    # We'll iterate i, then vectorize j,k
    for i in range(n):
        d = P - P[i]              # (n,2)
        # cross[i][j] = d[i] x d[j]  -> for fixed i, cross of (P[j]-P[i]) and (P[k]-P[i])
        cross = d[:, 0] * d[:, 1] - d[:, 1] * d[:, 0]  # (n,n), cross[j,k]
        np.fill_diagonal(cross, np.inf)
        cross[i, :] = np.inf
        cross[:, i] = np.inf
        m = np.min(np.abs(cross))
        if m < best:
            best = m
    return 0.5 * best


def _min_area_log(P):
    m = _min_area(P)
    if m <= 0:
        return -1e6
    return np.log(m)


def _inside(P, eps=1e-9):
    """Check all points strictly inside triangle (barycentric > eps)."""
    B = _barycentric(P)
    return bool(np.all(B > eps))


def _project_inside(P, eps=1e-6):
    """Clamp barycentric coords to [eps, 1-2eps] and renormalize."""
    B = _barycentric(P)
    B = np.clip(B, eps, 1.0 - 2 * eps)
    B = B / B.sum(axis=1, keepdims=True)
    return _cartesian(B)


def _init_11():
    """Known good 11-point layout: center, 3 edge-midpoints, 6 near-corner points."""
    # 3 corner-adjacent pairs along each edge near the corners, plus mid-edge
    # Use a symmetric pattern:
    # center
    c = np.array([0.5, SQ3 / 6.0])
    # 3 edge midpoints
    m01 = np.array([0.5, 0.0])
    m12 = np.array([0.75, SQ3 / 4.0])
    m20 = np.array([0.25, SQ3 / 4.0])
    # 6 points: 2 near each corner, pulled inward
    # Near corner 0 (0,0): points at (0.15,0) and (0.075, SQ3*0.075)
    a = 0.18
    p0a = np.array([a, 0.0])
    p0b = np.array([0.5 * a, SQ3 / 2.0 * a])
    # Near corner 1 (1,0)
    p1a = np.array([1.0 - a, 0.0])
    p1b = np.array([1.0 - 0.5 * a, SQ3 / 2.0 * a])
    # Near corner 2 (0.5, SQ3/2)
    p2a = np.array([0.5 + a, SQ3 / 2.0 - SQ3 / 2.0 * a])
    p2b = np.array([0.5 + 0.5 * a, SQ3 / 2.0 - SQ3 / 2.0 * 0.5 * a])
    pts = np.array([c, m01, m12, m20, p0a, p0b, p1a, p1b, p2a, p2b])
    return pts


def _nelder_mead(f, x0, max_iter=400, step=0.05, deadline=None, tol=1e-9):
    """Simple Nelder-Mead in cartesian space with interior projection."""
    n = len(x0)
    # Build simplex: x0 plus perturbations
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += step
        simplex.append(xi)
    vals = [f(s) for s in simplex]

    def sort_simplex():
        order = np.argsort(vals)
        simplex = [simplex[i].copy() for i in order]
        vals = [vals[i] for i in order]
        return simplex, vals

    simplex, vals = sort_simplex()
    best_x = simplex[0].copy()
    best_v = vals[0]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    it = 0
    while it < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        simplex, vals = sort_simplex()
        if vals[0] < best_v:
            best_v = vals[0]
            best_x = simplex[0].copy()
        if abs(vals[-1] - vals[0]) < tol * (1 + abs(vals[0])):
            break
        # centroid of all but worst
        c = np.mean(simplex[:-1], axis=0)
        # reflection
        xr = c + alpha * (c - simplex[-1])
        fr = f(xr)
        if fr < vals[0]:
            # expansion
            xe = c + gamma * (c - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe; vals[-1] = fe
            else:
                simplex[-1] = xr; vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr; vals[-1] = fr
        else:
            # contraction
            if fr < vals[-1]:
                xc = c + rho * (simplex[-1] - c)
                fc = f(xc)
                if fc <= fr:
                    simplex[-1] = xc; vals[-1] = fc
                else:
                    # shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        vals[i] = f(simplex[i])
            else:
                xc = c - rho * (c - simplex[-1])
                fc = f(xc)
                if fc < vals[-1]:
                    simplex[-1] = xc; vals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        vals[i] = f(simplex[i])
        it += 1
    return best_x, best_v


def _refine(P, deadline):
    """Iteratively refine P using Nelder-Mead on log-min-area, with restarts."""
    P = _project_inside(P)
    cur_val = _min_area(P)
    best = P.copy()

    def obj(x):
        x = _project_inside(x)
        return -_min_area_log(x)  # minimize negative log min area

    # Several rounds with decreasing step
    steps = [0.10, 0.05, 0.025, 0.012, 0.006, 0.003]
    for st in steps:
        if time.time() > deadline:
            break
        x, v = _nelder_mead(obj, P, max_iter=300, step=st, deadline=deadline)
        x = _project_inside(x)
        mv = _min_area(x)
        if mv > cur_val:
            cur_val = mv
            best = x.copy()
            P = x
    return best, cur_val


def _swap_improve(P, deadline):
    """Try pairwise swaps of point labels / small perturbations to escape local minima."""
    best = P.copy()
    best_val = _min_area(P)
    n = len(P)
    rng = np.random.default_rng(12345)
    attempts = 0
    while time.time() < deadline and attempts < 4000:
        attempts += 1
        # random small perturbation of one point
        i = rng.integers(0, n)
        Q = P.copy()
        Q[i] = Q[i] + rng.normal(0, 0.02, size=2)
        Q = _project_inside(Q)
        v = _min_area(Q)
        if v > best_val:
            best_val = v
            best = Q
            P = Q
        # occasionally a swap of two points' positions (symmetry breaking)
        if attempts % 7 == 0:
            i, j = rng.choice(n, size=2, replace=False)
            Q = P.copy()
            Q[[i, j]] = P[[j, i]]
            Q = _project_inside(Q)
            v = _min_area(Q)
            if v > best_val:
                best_val = v
                best = Q
                P = Q
    return best, best_val


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s

    # Warm start from known good layout
    P = _init_11()
    P = _project_inside(P)
    val = _min_area(P)

    # Main refinement
    P, val = _refine(P, deadline)

    # Swap-based local search to escape local minima
    if time.time() < deadline:
        P, val = _swap_improve(P, deadline)

    # Final safety: ensure strictly inside
    P = _project_inside(P)
    val = _min_area(P)

    return P
