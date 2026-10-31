"""Circle packing in unit square - 26 circles maximizing sum of radii.

Combined strategy:
  * Smooth surrogate objective (soft-max / soft-min via log-sum-exp) with a
    vectorized penalty term, optimized with L-BFGS-B (fast, gradient-based,
    no explicit constraint list).
  * SLSQP local search with explicit boundary + overlap constraints for a
    high-precision polish.
  * A dedicated radius-growth pass that fixes positions and squeezes radii
    into the gaps left by the neighbors.
  * Multiple diverse initializations + a light global perturbation loop.

Vectorized parameter layout: x = [x1..xN, y1..yN, r1..rN]
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Smooth helpers
# ---------------------------------------------------------------------------

def _softplus(x, beta=50.0):
    """Numerically stable softplus: smooth max(0, x)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    big = x > 20.0 / beta
    out[big] = x[big]
    small = ~big
    out[small] = np.log1p(np.exp(beta * x[small])) / beta
    return out


def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) via softplus."""
    return _softplus(v, beta)


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _to_circles(x):
    """x (vector, layout [x.., y.., r..]) -> list of (x, y, r)."""
    pos = x[:2 * N].reshape(N, 2)
    rad = x[2 * N:]
    return [(float(pos[i, 0]), float(pos[i, 1]), float(max(rad[i], 0.0)))
            for i in range(N)]


def _to_vec(circles):
    out = np.empty(3 * N)
    for i, (x, y, r) in enumerate(circles):
        out[i] = x
        out[N + i] = y
        out[2 * N + i] = r
    return out


# ---------------------------------------------------------------------------
# Smooth objective (for L-BFGS-B)
# ---------------------------------------------------------------------------

def _objective(x, penalty_weight=200.0, beta=40.0):
    """minimize -sum(radii) + penalty * smooth_violations (vectorized)."""
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-8)

    # Smooth boundary: soft-min of the four wall distances
    d = np.stack([pos[:, 0], 1.0 - pos[:, 0], pos[:, 1], 1.0 - pos[:, 1]], axis=1)
    neg_d = -d
    m = np.max(neg_d, axis=1, keepdims=True)
    neg_d_clipped = np.clip(neg_d, -50.0, 50.0)
    logsum = m + np.log(np.sum(np.exp(neg_d_clipped - m), axis=1, keepdims=True))
    soft_min_d = -logsum[:, 0]
    boundary_pen = np.sum(_soft_max(radii - soft_min_d, beta))

    # Smooth overlap: required = r_i + r_j, dist = ||p_i - p_j||
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol_raw = min_dist - dist
    overlap_pen = np.sum(_soft_max(viol_raw[mask], beta))

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)


# ---------------------------------------------------------------------------
# Validity / post-processing
# ---------------------------------------------------------------------------

def _is_valid(circles, tol=1e-8):
    """Check validity: inside unit square, non-overlapping, positive radii."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist < ri + rj - tol:
                return False
    return True


def _post_process(circles, tol=1e-8):
    """Shrink radii to guarantee strict validity, clip positions to bounds."""
    out = []
    for x, y, r in circles:
        r = max(0.0, r - tol)
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        out.append((x, y, r))
    return out


def _sum_radii(circles):
    return sum(c[2] for c in circles)


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------

def _init_hexagonal():
    """Hexagonal packing initial layout for 26 circles."""
    circles = []
    r = 0.075
    rows = [5, 5, 5, 5, 6]
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r

    row_idx = 0
    for count in rows:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.03
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if len(circles) < N:
                circles.append((x, y, r))
        row_idx += 1

    circles = circles[:N]
    while len(circles) < N:
        circles.append((0.5, 0.5, 0.01))
    return circles


def _init_grid():
    """Simple grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.4 / side
        circles.append((x, y, r))
    return circles


def _init_corners():
    """Initialization with circles at corners and edges."""
    circles = []
    circles.append((0.08, 0.08, 0.08))
    circles.append((0.92, 0.08, 0.08))
    circles.append((0.08, 0.92, 0.08))
    circles.append((0.92, 0.92, 0.08))
    circles.append((0.5, 0.05, 0.06))
    circles.append((0.5, 0.95, 0.06))
    circles.append((0.05, 0.5, 0.06))
    circles.append((0.95, 0.5, 0.06))

    random.seed(42)
    while len(circles) < N:
        x = random.uniform(0.15, 0.85)
        y = random.uniform(0.15, 0.85)
        r = random.uniform(0.04, 0.07)
        circles.append((x, y, r))
    return circles


def _init_random(seed=None):
    """Random initialization."""
    if seed is not None:
        random.seed(seed)
    circles = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.03, 0.08)
        circles.append((x, y, r))
    return circles


def _init_best_fit():
    """Greedy best-fit initialization: place each circle where it fits largest."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    pos[0] = [0.5, 0.5]
    radii[0] = 0.25

    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)

        for _ in range(3000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)

            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r

            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True

        if not found or best_r < 0.003:
            best_x, best_y, best_r = 0.5, 0.5, 0.003

        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97

    return _to_circles(_to_vec([(pos[i, 0], pos[i, 1], radii[i]) for i in range(N)]))


# ---------------------------------------------------------------------------
# L-BFGS-B on the smooth objective
# ---------------------------------------------------------------------------

def _lbfgsb_solve(x, deadline, penalty_weight=200.0, beta=40.0):
    """Vectorized L-BFGS-B on the smooth penalty objective."""
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-4, 0.5)] * N

    def obj(p):
        return _objective(p, penalty_weight, beta)

    try:
        result = minimize(
            obj, x, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-9,
                     'maxfun': 20000, 'maxls': 40}
        )
        return result.x
    except Exception:
        return x


# ---------------------------------------------------------------------------
# SLSQP local search with explicit constraints (high-precision polish)
# ---------------------------------------------------------------------------

def _slsqp_constraints():
    """Build the constraint list once (boundary + pairwise overlap)."""
    constraints = []
    dim = 3 * N

    # Boundary constraints: x>=r, x<=1-r, y>=r, y<=1-r
    for i in range(N):
        ix, iy, ir = 3 * i, 3 * i + 1, 3 * i + 2
        # x - r >= 0
        j = np.zeros(dim); j[ix] = 1.0; j[ir] = -1.0
        constraints.append({'type': 'ineq',
                            'fun': lambda p, ix=ix, ir=ir: p[ix] - p[ir],
                            'jac': lambda p, j=j: j})
        # 1 - x - r >= 0
        j = np.zeros(dim); j[ix] = -1.0; j[ir] = -1.0
        constraints.append({'type': 'ineq',
                            'fun': lambda p, ix=ix, ir=ir: 1.0 - p[ix] - p[ir],
                            'jac': lambda p, j=j: j})
        # y - r >= 0
        j = np.zeros(dim); j[iy] = 1.0; j[ir] = -1.0
        constraints.append({'type': 'ineq',
                            'fun': lambda p, iy=iy, ir=ir: p[iy] - p[ir],
                            'jac': lambda p, j=j: j})
        # 1 - y - r >= 0
        j = np.zeros(dim); j[iy] = -1.0; j[ir] = -1.0
        constraints.append({'type': 'ineq',
                            'fun': lambda p, iy=iy, ir=ir: 1.0 - p[iy] - p[ir],
                            'jac': lambda p, j=j: j})

    # Overlap constraints: (dx)^2 + (dy)^2 - (ri+rj)^2 >= 0
    for i in range(N):
        for j in range(i + 1, N):
            ixi, iyi, iri = 3 * i, 3 * i + 1, 3 * i + 2
            ixj, iyj, irj = 3 * j, 3 * j + 1, 3 * j + 2

            def overlap_fun(p, ixi=ixi, iyi=iyi, iri=iri,
                            ixj=ixj, iyj=iyj, irj=irj):
                dx = p[ixi] - p[ixj]
                dy = p[iyi] - p[iyj]
                return dx * dx + dy * dy - (p[iri] + p[irj]) ** 2

            def overlap_jac(p, ixi=ixi, iyi=iyi, iri=iri,
                            ixj=ixj, iyj=iyj, irj=irj):
                jac = np.zeros(dim)
                dx = p[ixi] - p[ixj]
                dy = p[iyi] - p[iyj]
                sum_r = p[iri] + p[irj]
                jac[ixi] = 2 * dx
                jac[ixj] = -2 * dx
                jac[iyi] = 2 * dy
                jac[iyj] = -2 * dy
                jac[iri] = -2 * sum_r
                jac[irj] = -2 * sum_r
                return jac

            constraints.append({'type': 'ineq', 'fun': overlap_fun,
                                'jac': overlap_jac})

    return constraints


_SLSQP_CONSTR = _slsqp_constraints()
_SLSQP_BOUNDS = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * N


def _slsqp_solve(params, maxiter=300, ftol=1e-12):
    """Solve SLSQP with explicit constraints."""
    def objective(p):
        return -sum(p[3 * i + 2] for i in range(N))

    def jac(p):
        g = np.zeros(3 * N)
        g[2::3] = -1.0
        return g

    try:
        result = minimize(
            objective, params, jac=jac,
            method='SLSQP', bounds=_SLSQP_BOUNDS, constraints=_SLSQP_CONSTR,
            options={'maxiter': maxiter, 'ftol': ftol}
        )
        return result.x
    except Exception:
        return params


def _local_search(circles, deadline, maxiter=300, ftol=1e-12):
    """SLSQP local search from a starting configuration."""
    params = _to_vec(circles)
    new_params = _slsqp_solve(params, maxiter=maxiter, ftol=ftol)
    return _to_circles(new_params)


# ---------------------------------------------------------------------------
# Radius-growth pass: fix positions, maximize radii into the gaps
# ---------------------------------------------------------------------------

def _radius_growth(circles, deadline, maxiter=200):
    """Fix positions, grow radii subject to boundary + overlap constraints."""
    params = _to_vec(circles)
    dim = 3 * N

    bounds = []
    for i in range(N):
        x, y, r = circles[i]
        bounds.append((x, x))       # x fixed
        bounds.append((y, y))       # y fixed
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((1e-5, max(max_r, 1e-5)))

    def objective(p):
        return -sum(p[3 * i + 2] for i in range(N))

    def jac(p):
        g = np.zeros(dim)
        g[2::3] = -1.0
        return g

    try:
        result = minimize(
            objective, params, jac=jac, method='SLSQP',
            bounds=bounds, constraints=_SLSQP_CONSTR,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        return _to_circles(result.x)
    except Exception:
        return circles


# ---------------------------------------------------------------------------
# Light global perturbation (replaces heavy simulated annealing)
# ---------------------------------------------------------------------------

def _perturb(circles, scale=0.05):
    """Small random jitter of positions and radii."""
    out = []
    for x, y, r in circles:
        nx = x + random.gauss(0, scale * 0.5)
        ny = y + random.gauss(0, scale * 0.5)
        nr = max(0.01, min(0.4, r + random.gauss(0, scale * 0.3)))
        nx = max(nr, min(1.0 - nr, nx))
        ny = max(nr, min(1.0 - nr, ny))
        out.append((nx, ny, nr))
    return out


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under the 150s kill

    inits = [
        _init_hexagonal(),
        _init_grid(),
        _init_corners(),
        _init_best_fit(),
        _init_random(42),
        _init_random(123),
        _init_random(777),
        _init_random(999),
    ]

    best_circles = inits[0]
    best_sum = _sum_radii(best_circles)

    def consider(cand):
        nonlocal best_circles, best_sum
        if _is_valid(cand):
            s = _sum_radii(cand)
            if s > best_sum:
                best_circles = cand
                best_sum = s

    # Phase 1: L-BFGS-B on the smooth objective from each diverse start.
    for init_circles in inits:
        if time.time() > deadline - 40.0:
            break
        x = _to_vec(init_circles)
        x_opt = _lbfgsb_solve(x, deadline)
        consider(_to_circles(x_opt))

    # Phase 2: SLSQP polish + radius-growth on the current best.
    if time.time() < deadline - 20.0:
        polished = _local_search(best_circles, deadline, maxiter=300, ftol=1e-12)
        consider(polished)
        grown = _radius_growth(best_circles, deadline, maxiter=200)
        consider(grown)

    # Phase 3: a few perturbation -> L-BFGS-B -> SLSQP rounds while time lasts.
    random.seed(2024)
    rounds = 0
    while time.time() < deadline - 15.0 and rounds < 8:
        rounds += 1
        cand = _perturb(best_circles, scale=0.05)
        x_opt = _lbfgsb_solve(_to_vec(cand), deadline)
        cand = _to_circles(x_opt)
        polished = _local_search(cand, deadline, maxiter=200, ftol=1e-12)
        consider(polished)

    # Phase 4: final high-precision polish + radius growth.
    if time.time() < deadline - 5.0:
        final = _local_search(best_circles, deadline, maxiter=400, ftol=1e-13)
        consider(final)
        final = _radius_growth(best_circles, deadline, maxiter=200)
        consider(final)
        final = _local_search(best_circles, deadline, maxiter=200, ftol=1e-13)
        consider(final)

    # Post-process for strict validity.
    best_circles = _post_process(best_circles)

    if not _is_valid(best_circles):
        best_circles = _post_process(_init_grid())

    return np.array(best_circles)

# EVOLVE-BLOCK-END
