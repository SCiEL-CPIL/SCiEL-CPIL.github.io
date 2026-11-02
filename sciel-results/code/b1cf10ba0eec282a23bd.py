import math
import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 150.0
MARGIN = 12.0
EPS = 1e-9

# A regular hexagon of side 1, oriented so its flats face the 6 axial directions
# (pointy-top). These are the 6 unit normal directions (apothem directions).
N6 = np.array([
    [math.cos(math.pi / 2), math.sin(math.pi / 2)],
    [math.cos(7 * math.pi / 6), math.sin(7 * math.pi / 6)],
    [math.cos(11 * math.pi / 6), math.sin(11 * math.pi / 6)],
    [math.cos(3 * math.pi / 6), math.sin(3 * math.pi / 6)],
    [math.cos(5 * math.pi / 6), math.sin(5 * math.pi / 6)],
    [math.cos(9 * math.pi / 6), math.sin(9 * math.pi / 6)],
])


def axial_to_vec(q, r):
    return (math.sqrt(3) * (q + 0.5 * r), 1.5 * r)


def compute_side(positions, beta):
    """Outer side length (apothem = max over 6 dirs of |projection| + 1)."""
    cx = float(np.mean(positions[:, 0]))
    cy = float(np.mean(positions[:, 1]))
    c = np.array([cx, cy])
    R = np.array([[math.cos(beta), -math.sin(beta)], [math.sin(beta), math.cos(beta)]])
    N = N6 @ R.T
    p = positions - c
    proj = p @ N.T
    maxproj = float(np.max(np.abs(proj)))
    return maxproj + 1.0, (cx, cy)


def feasible(positions):
    """No-overlap check via axial grid (valid for hexagons sharing one orientation)."""
    inv = np.array([
        [1.0, 0.0],
        [-1.0 / (2.0 * math.sqrt(3)), 1.0 / (1.5)],
    ])
    A = positions @ inv.T
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            dq = A[i, 0] - A[j, 0]
            dr = A[i, 1] - A[j, 1]
            # overlap iff max(|dq|, |dr|, |dq+dr|) < 1
            if max(abs(dq), abs(dr), abs(dq + dr)) < 1.0 - 1e-9:
                return False
    return True


def initial_pop(n):
    pop = np.empty((n, 22))
    for i in range(n):
        rng = np.random.RandomState(i * 2654435761 + 12345)
        pts = np.array([axial_to_vec(q, r) for q, r in _cluster_seed()])
        pts = pts + 0.5 * rng.randn(11, 2)
        pop[i] = pts.reshape(-1)
    return pop


def _cluster_seed():
    # 11 hexagons: a 3x3 block minus 1 corner, plus 2 neighbours (no overlap)
    return [(0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (-1, 1), (3, 1)]


def penalty(positions):
    inv = np.array([
        [1.0, 0.0],
        [-1.0 / (2.0 * math.sqrt(3)), 1.0 / (1.5)],
    ])
    A = positions @ inv.T
    pen = 0.0
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            d = max(abs(A[i, 0] - A[j, 0]), abs(A[i, 1] - A[j, 1]),
                    abs((A[i, 0] - A[j, 0]) + (A[i, 1] - A[j, 1])))
            if d < 1.0:
                pen += (1.0 - d) ** 2
    return pen


def evaluate(vec, beta):
    pts = vec.reshape(11, 2)
    if not feasible(pts):
        return None, None
    side, cen = compute_side(pts, beta)
    return side, (pts, cen)


def refine_beta(pts, beta0, budget=0.8):
    def f(b):
        s, _ = compute_side(pts, b)
        return s
    best_b, best_v = beta0, f(beta0)
    step = 0.01
    t_end = time.time() + budget
    for _ in range(400):
        if time.time() > t_end:
            break
        for b in (best_b + step, best_b - step):
            v = f(b)
            if v < best_v:
                best_b, best_v = b, v
                step *= 0.5
            else:
                step *= 0.5
        if step < 1e-9:
            break
    # fine golden-section polish
    a, b = best_b - 1e-3, best_b + 1e-3
    gr = (math.sqrt(5) - 1) / 2
    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(60):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - gr * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + gr * (b - a)
            f2 = f(x2)
    best_b = 0.5 * (a + b)
    return best_b, f(best_b)


def construct():
    if time.time() - START_TIME > TIME_LIMIT - MARGIN:
        # fallback: a valid non-overlapping configuration
        pts = np.array([axial_to_vec(q, r) for q, r in _cluster_seed()])
        side, cen = compute_side(pts, 0.0)
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [float(cen[0]), float(cen[1])], float(side), 0.0

    best_side = float('inf')
    best_pts = None
    best_beta = 0.0
    best_cen = None

    pop = initial_pop(48)
    np.random.seed(7)
    beta_choices = [0.0, math.pi / 12, math.pi / 6, math.pi / 18, 5 * math.pi / 36]

    t_end = time.time() + (TIME_LIMIT - MARGIN)
    it = 0
    while time.time() < t_end:
        it += 1
        # differential evolution step
        n = len(pop)
        new_pop = np.empty_like(pop)
        for i in range(n):
            a, b, c = (np.random.randint(n) for _ in range(3))
            while b == i or c == i:
                b, c = np.random.randint(n), np.random.randint(n)
            trial = pop[a] + 0.8 * (pop[b] - pop[c])
            # occasional recombination + mutation
            r = np.random.rand()
            if r < 0.05:
                trial = pop[i] + 0.3 * np.random.randn(22)
            new_pop[i] = trial

        beta = beta_choices[it % len(beta_choices)]
        # evaluate current pop and trials; keep best per slot
        for i in range(n):
            s_cur, _ = evaluate(pop[i], beta)
            s_tr, _ = evaluate(new_pop[i], beta)
            if s_cur is None:
                s_cur = 1e6 + penalty(pop[i].reshape(11, 2))
            if s_tr is None:
                s_tr = 1e6 + penalty(new_pop[i].reshape(11, 2))
            if s_tr <= s_cur:
                pop[i] = new_pop[i]
            # track global best on exact side
            if s_cur is not None and s_cur < best_side:
                best_side, best_pts = s_cur, pop[i].reshape(11, 2)
                best_beta = beta
            if s_tr is not None and s_tr < best_side:
                best_side, best_pts = s_tr, new_pop[i].reshape(11, 2)
                best_beta = beta

        # refine rotation for the best candidate when we have time
        if best_pts is not None and time.time() < t_end - 3.0 and it % 8 == 0:
            rb, rv = refine_beta(best_pts, best_beta, budget=0.8)
            if rv < best_side:
                best_side, best_beta = rv, rb

    # final polish on the best configuration
    if best_pts is not None:
        rb, rv = refine_beta(best_pts, best_beta, budget=1.5)
        if rv < best_side:
            best_side, best_beta = rv, rb

    if best_pts is None:
        pts = np.array([axial_to_vec(q, r) for q, r in _cluster_seed()])
        best_pts, best_side, best_beta = pts, compute_side(pts, 0.0)[0], 0.0

    _, best_cen = compute_side(best_pts, best_beta)
    inner = [[float(x), float(y), float(math.degrees(best_beta))] for x, y in best_pts]
    return inner, [float(best_cen[0]), float(best_cen[1])], float(best_side), float(math.degrees(best_beta))
