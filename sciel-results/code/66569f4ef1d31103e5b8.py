"""Circle packing in unit square - 26 circles maximizing sum of radii.

Strategy: Differential-Evolution global search over the full (x,y,r) space
with a smooth penalty objective, followed by L-BFGS local refinement and a
final hard-constraint SLSQP polish.  This is structurally different from the
current operator-alternation / force-directed approach: DE is a population-
based derivative-free global optimizer that can jump over many local optima
that sequential local search gets stuck in.  The penalty is smooth (softplus
of violations) so L-BFGS can polish each DE individual efficiently.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26
DIM = 3 * N  # 78

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid(circles, tol=1e-7):
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
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True


def _post_process(circles, tol=1e-7):
    """Shrink circles slightly to ensure strict validity, clip to bounds."""
    out = []
    for x, y, r in circles:
        r = max(0.0, r - tol)
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        out.append((x, y, r))
    return out


def _sum_radii(circles):
    return sum(c[2] for c in circles)


def _circles_to_vec(circles):
    return np.array([v for c in circles for v in c], dtype=float)


def _vec_to_circles(v):
    return [(v[3*i], v[3*i+1], v[3*i+2]) for i in range(N)]


# ---------------------------------------------------------------------------
# Smooth penalty objective
# ---------------------------------------------------------------------------

def _softplus(z, k=50.0):
    """Numerically stable softplus: max(z,0) + log(1+exp(-k*|z|))/k."""
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-k * np.abs(z))) / k


def _penalty_objective(vec, k=50.0):
    """
    Smooth penalty objective = -sum(r) + penalty(violations).
    Lower is better.  Returns (value, violations_dict) for debugging.
    """
    x = vec[0::3]
    y = vec[1::3]
    r = vec[2::3]

    # --- boundary violations (circle must be inside [0,1]^2) ---
    # x - r >= 0,  x + r <= 1,  y - r >= 0,  y + r <= 1
    bnd = np.concatenate([
        -(x - r),          # violation if x < r
        (x + r) - 1.0,     # violation if x + r > 1
        -(y - r),
        (y + r) - 1.0,
    ])
    bnd_pen = np.sum(_softplus(bnd, k))

    # --- collision violations: dist_ij >= r_i + r_j  =>  dist^2 - (r_i+r_j)^2 >= 0 ---
    # Use squared form to avoid sqrt in gradient; still smooth.
    col_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            s = r[i] + r[j]
            viol = s**2 - (dx*dx + dy*dy)  # positive when overlapping
            col_pen += _softplus(viol, k)

    # --- negative radius violation ---
    neg_pen = np.sum(_softplus(-r, k))

    penalty = bnd_pen + col_pen + neg_pen
    obj = -np.sum(r) + penalty
    return obj


def _penalty_gradient(vec, k=50.0):
    """Analytic gradient of _penalty_objective."""
    x = vec[0::3].copy()
    y = vec[1::3].copy()
    r = vec[2::3].copy()

    grad = np.zeros(DIM)

    # --- objective part: d(-sum r)/dr_i = -1 ---
    grad[2::3] = -1.0

    # --- boundary gradients ---
    # For softplus(z, k): d/dz = sigmoid(k*z)
    def _sig(z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-k * z))

    # x - r >= 0  =>  z = -(x-r) = r - x ;  dz/dx = -1, dz/dr = 1
    z1 = r - x
    g1 = _sig(z1)
    grad[0::3] += -g1 * 1.0   # d/dx of softplus(r-x) = sig * (-1)
    grad[2::3] += g1 * 1.0    # d/dr of softplus(r-x) = sig * 1

    # x + r <= 1  =>  z = x+r-1 ;  dz/dx=1, dz/dr=1
    z2 = x + r - 1.0
    g2 = _sig(z2)
    grad[0::3] += g2 * 1.0
    grad[2::3] += g2 * 1.0

    # y - r >= 0  =>  z = r - y
    z3 = r - y
    g3 = _sig(z3)
    grad[1::3] += -g3 * 1.0
    grad[2::3] += g3 * 1.0

    # y + r <= 1  =>  z = y + r - 1
    z4 = y + r - 1.0
    g4 = _sig(z4)
    grad[1::3] += g4 * 1.0
    grad[2::3] += g4 * 1.0

    # --- collision gradients ---
    # viol_ij = (r_i+r_j)^2 - (dx^2+dy^2)
    # d/dx_i = -2*dx * sig,  d/dx_j = +2*dx * sig
    # d/dr_i = 2*(r_i+r_j) * sig,  d/dr_j = 2*(r_i+r_j) * sig
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            s = r[i] + r[j]
            viol = s**2 - (dx*dx + dy*dy)
            sig = _sig(viol)
            if sig < 1e-12:
                continue
            # x
            grad[3*i]   += -2.0 * dx * sig
            grad[3*j]   +=  2.0 * dx * sig
            # y
            grad[3*i+1] += -2.0 * dy * sig
            grad[3*j+1] +=  2.0 * dy * sig
            # r
            grad[3*i+2] += 2.0 * s * sig
            grad[3*j+2] += 2.0 * s * sig

    # --- negative radius gradient ---
    # z = -r ;  dz/dr = -1
    z5 = -r
    g5 = _sig(z5)
    grad[2::3] += -g5 * 1.0

    return grad


# ---------------------------------------------------------------------------
# Hard-constraint radius growth (positions fixed)
# ---------------------------------------------------------------------------

def _max_feasible_radius(i, circles, tol=1e-10):
    x, y, _ = circles[i]
    max_r = min(x, 1.0 - x, y, 1.0 - y)
    for j in range(N):
        if j == i:
            continue
        xj, yj, rj = circles[j]
        dist = np.sqrt((x - xj)**2 + (y - yj)**2)
        if dist > 0:
            max_r = min(max_r, dist - rj)
        else:
            max_r = 0.0
    return max(0.0, max_r)


def _grow_radii(circles, deadline, tol=1e-10):
    circles = [list(c) for c in circles]
    for _ in range(200):
        if time.time() > deadline - 15.0:
            break
        changed = False
        for i in range(N):
            max_r = _max_feasible_radius(i, circles, tol)
            if max_r > circles[i][2] + 1e-12:
                circles[i][2] = max_r
                changed = True
            elif circles[i][2] > max_r + 1e-12:
                circles[i][2] = max(0.0, max_r)
                changed = True
        if not changed:
            break
    return [tuple(c) for c in circles]


# ---------------------------------------------------------------------------
# L-BFGS local polish (smooth penalty)
# ---------------------------------------------------------------------------

def _lbfgs_polish(vec, deadline, maxiter=300):
    """Run L-BFGS-B on the smooth penalty objective."""
    def obj(v):
        return _penalty_objective(v)

    def jac(v):
        return _penalty_gradient(v)

    # bounds: x,y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))   # x
        bounds.append((0.0, 1.0))   # y
        bounds.append((0.0, 0.5))   # r

    try:
        res = minimize(
            obj, vec, jac=jac, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x
    except Exception:
        return vec


# ---------------------------------------------------------------------------
# SLSQP hard-constraint polish
# ---------------------------------------------------------------------------

def _slsqp_polish(circles, deadline):
    """SLSQP with hard pairwise and boundary constraints."""
    def objective(params):
        return -sum(params[3*i+2] for i in range(N))

    def _jac_overlap(p, i, j):
        jac = np.zeros(3 * N)
        xi, yi, ri = p[3*i], p[3*i+1], p[3*i+2]
        xj, yj, rj = p[3*j], p[3*j+1], p[3*j+2]
        jac[3*i]   = 2 * (xi - xj)
        jac[3*j]   = -2 * (xi - xj)
        jac[3*i+1] = 2 * (yi - yj)
        jac[3*j+1] = -2 * (yi - yj)
        jac[3*i+2] = -2 * (ri + rj)
        jac[3*j+2] = -2 * (ri + rj)
        return jac

    cons = []
    for i in range(N):
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2],
                     'jac': lambda p, i=i: np.array([1.0 if k == 3*i else (-1.0 if k == 3*i+2 else 0.0) for k in range(3*N)])})
        # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2],
                     'jac': lambda p, i=i: np.array([(-1.0 if k == 3*i else (-1.0 if k == 3*i+2 else 0.0)) for k in range(3*N)])})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2],
                     'jac': lambda p, i=i: np.array([1.0 if k == 3*i+1 else (-1.0 if k == 3*i+2 else 0.0) for k in range(3*N)])})
        # 1 - y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2],
                     'jac': lambda p, i=i: np.array([(-1.0 if k == 3*i+1 else (-1.0 if k == 3*i+2 else 0.0)) for k in range(3*N)])})
        # r >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2],
                     'jac': lambda p, i=i: np.array([1.0 if k == 3*i+2 else 0.0 for k in range(3*N)])})

    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2,
                'jac': lambda p, i=i, j=j: _jac_overlap(p, i, j)
            })

    params = np.array([val for circle in circles for val in circle])
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N

    try:
        result = minimize(
            objective, params, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 80, 'ftol': 1e-10}
        )
        if result.success:
            new_circles = [(result.x[3*i], result.x[3*i+1], result.x[3*i+2]) for i in range(N)]
            if _is_valid(new_circles):
                return new_circles
    except Exception:
        pass
    return circles


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------

def _init_hexagonal():
    rows_counts = [5, 5, 5, 5, 6]
    circles = []
    r = 0.08
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r
    row_idx = 0
    for count in rows_counts:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.02
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if len(circles) < N:
                circles.append([x, y, r])
        row_idx += 1
    circles = circles[:N]
    while len(circles) < N:
        circles.append([0.5, 0.5, 0.01])
    return np.array(circles, dtype=float)


def _init_grid():
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.45 / side
        circles.append([x, y, r])
    return np.array(circles, dtype=float)


def _init_corners():
    circles = []
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for cx, cy in corners:
        circles.append([cx, cy, 0.06])
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for ex, ey in edges:
        circles.append([ex, ey, 0.05])
    inner = [(0.5, 0.5), (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7),
             (0.5, 0.3), (0.5, 0.7), (0.3, 0.5), (0.7, 0.5),
             (0.4, 0.4), (0.6, 0.4), (0.4, 0.6), (0.6, 0.6),
             (0.35, 0.65), (0.65, 0.35), (0.35, 0.35), (0.65, 0.65)]
    for ix, iy in inner:
        if len(circles) < N:
            circles.append([ix, iy, 0.04])
    circles = circles[:N]
    while len(circles) < N:
        circles.append([0.5, 0.5, 0.01])
    return np.array(circles, dtype=float)


def _init_random(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    circles = []
    for _ in range(N):
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        r = random.uniform(0.02, 0.10)
        circles.append([x, y, r])
    return np.array(circles, dtype=float)


# ---------------------------------------------------------------------------
# Differential Evolution
# ---------------------------------------------------------------------------

def _differential_evolution(vecs, deadline, pop_size=20, F=0.8, CR=0.9,
                            max_gen=200, lbfgs_every=5):
    """
    Custom DE on the smooth penalty objective.
    vecs: list of initial candidate vectors (DIM,).
    Returns best vector.
    """
    # Build initial population
    pop = []
    for v in vecs:
        pop.append(np.clip(v, 0, 1).copy())
    # Pad with random
    rng = np.random.RandomState(42)
    while len(pop) < pop_size:
        v = np.zeros(DIM)
        for i in range(N):
            v[3*i]   = rng.uniform(0.05, 0.95)
            v[3*i+1] = rng.uniform(0.05, 0.95)
            v[3*i+2] = rng.uniform(0.02, 0.10)
        pop.append(v)

    # Evaluate
    fvals = np.array([_penalty_objective(v) for v in pop])
    best_idx = int(np.argmin(fvals))
    best_vec = pop[best_idx].copy()
    best_f = fvals[best_idx]

    gen = 0
    while gen < max_gen:
        if time.time() > deadline - 25.0:
            break

        new_pop = [p.copy() for p in pop]
        for i in range(pop_size):
            # Pick three distinct random indices different from i
            a, b, c = rng.choice([k for k in range(pop_size) if k != i], size=3, replace=False)

            # Mutation
            mutant = pop[a] + F * (pop[b] - pop[c])

            # Crossover (binomial)
            trial = pop[i].copy()
            j_rand = rng.randint(0, DIM - 1)
            for j in range(DIM):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # Bounds
            trial = np.clip(trial, 0, 1)
            # Radii upper bound
            for k in range(N):
                trial[3*k+2] = min(trial[3*k+2], 0.5)

            # Evaluate
            f_trial = _penalty_objective(trial)
            if f_trial <= fvals[i]:
                new_pop[i] = trial
                fvals[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_vec = trial.copy()
                    best_idx = i

        pop = new_pop
        gen += 1

        # Periodic L-BFGS polish on best few
        if gen % lbfgs_every == 0 and time.time() < deadline - 30.0:
            # Polish top 3
            order = np.argsort(fvals)
            for idx in order[:3]:
                polished = _lbfgs_polish(pop[idx], deadline, maxiter=100)
                f_pol = _penalty_objective(polished)
                if f_pol < fvals[idx]:
                    pop[idx] = polished
                    fvals[idx] = f_pol
                    if f_pol < best_f:
                        best_f = f_pol
                        best_vec = polished.copy()

    return best_vec


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    np.random.seed(42)
    random.seed(42)

    # Diverse initializations
    inits = [
        _init_hexagonal(),
        _init_grid(),
        _init_corners(),
        _init_random(42),
        _init_random(123),
        _init_random(777),
        _init_random(999),
        _init_random(314),
        _init_random(271),
    ]

    # Convert to vectors
    init_vecs = [_circles_to_vec(c) for c in inits]

    # --- Phase 1: DE global search ---
    best_vec = _differential_evolution(init_vecs, deadline,
                                        pop_size=20, F=0.8, CR=0.9,
                                        max_gen=200, lbfgs_every=5)

    best_circles = _vec_to_circles(best_vec)

    # --- Phase 2: L-BFGS polish ---
    if time.time() < deadline - 30.0:
        polished_vec = _lbfgs_polish(best_vec, deadline, maxiter=300)
        polished_circles = _vec_to_circles(polished_vec)
        # Check if improved
        if _penalty_objective(polished_vec) < _penalty_objective(best_vec):
            best_vec = polished_vec
            best_circles = polished_circles

    # --- Phase 3: Hard-constraint radius growth ---
    best_circles = _grow_radii(best_circles, deadline)

    # --- Phase 4: SLSQP hard-constraint polish ---
    if time.time() < deadline - 15.0:
        polished = _slsqp_polish(best_circles, deadline)
        if _sum_radii(polished) > _sum_radii(best_circles) and _is_valid(polished):
            best_circles = polished

    # --- Phase 5: Final radius growth ---
    best_circles = _grow_radii(best_circles, deadline)

    # --- Post-process ---
    best_circles = _post_process(best_circles)

    # --- Verify ---
    if not _is_valid(best_circles):
        # Fallback
        fallback = _post_process(_init_grid().tolist())
        best_circles = fallback

    return np.array(best_circles)

# EVOLVE-BLOCK-END
