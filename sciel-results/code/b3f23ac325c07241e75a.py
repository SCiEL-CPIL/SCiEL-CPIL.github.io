# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 620.0  # wall-clock safety margin (evaluator kills at ~630s)
_T0 = time.time()


def _hex_vertices(cx, cy, s, a_deg):
    a = math.radians(a_deg)
    out = np.empty((6, 2))
    for i in range(6):
        th = a + math.radians(60.0 * i)
        out[i, 0] = cx + s * math.cos(th)
        out[i, 1] = cy + s * math.sin(th)
    return out


def _separation_ok(A, B):
    """True if the two unit hexagons (each 6x2 vertex array) are separated
    on at least one of the 12 edge normals (gap > 1e-6)."""
    for M in (A, B):
        for i in range(6):
            p1 = M[i]
            p2 = M[(i + 1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            na = n @ A
            nb = n @ B
            if (na.max() - nb.max()) > 1e-6 or (nb.max() - na.max()) > 1e-6:
                return True
    return False


def _check_all_separated(V):
    for i in range(11):
        for j in range(i + 1, 11):
            if not _separation_ok(V[i], V[j]):
                return False
    return True


def _min_side(V):
    """Smallest enclosing regular hexagon side (center at origin).
    Outer side s with angle a contains every vertex iff, for every
    edge-normal direction n_k(a), max_k dot(p, n_k) <= s. Min over a."""
    p = V.reshape(-1, 2)
    best = math.inf
    for m in range(12):
        a = math.radians(30.0 * m)
        s = 0.0
        for k in range(6):
            th = a + math.radians(60.0 * k)
            n = np.array([math.cos(th), math.sin(th)])
            d = p @ n
            s = max(s, d.max())
        if s < best:
            best = s
    return best


def _evaluate(V):
    """Return (side, valid). V: (11,6,2)."""
    if not _check_all_separated(V):
        return math.inf, False
    s = _min_side(V)
    return s, True


def _positions_to_vertices(pos):
    V = np.empty((11, 6, 2))
    for i in range(11):
        V[i] = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, 0.0)
    return V


def _sa_from(start, budget):
    pos = start.copy()
    V = _positions_to_vertices(pos)
    s, valid = _evaluate(V)
    best = (s if valid else math.inf, pos.copy())
    if not valid:
        # kick until feasible
        t = 2.0
        for _ in range(30000):
            if time.time() > budget:
                break
            k = np.random.randint(11)
            pos[k] += np.random.normal(0, 0.6, 2)
            V = _positions_to_vertices(pos)
            s2, v2 = _evaluate(V)
            if v2:
                s, valid = s2, True
                best = (s, pos.copy())
                t = 0.8
                break
    if not valid:
        return best
    t = 0.6
    step = 0.35
    while time.time() < budget:
        k = np.random.randint(11)
        pos[k] += np.random.normal(0, step, 2)
        V = _positions_to_vertices(pos)
        s2, v2 = _evaluate(V)
        if v2 and (s2 < s or np.random.rand() < math.exp((s - s2) / t)):
            s, valid = s2, True
            if s2 < best[0]:
                best = (s2, pos.copy())
        elif not v2:
            if np.random.rand() < math.exp(-3.0 / t):
                s = math.inf
            else:
                pos[k] -= np.random.normal(0, step, 2)
        t *= 0.9995
        if t < 1e-4:
            t = 1e-4
        step *= 0.9999
    return best


def hexagon_packing_11():
    rng = np.random.default_rng(12345)
    # diverse initial seeds: hexagonal-cluster-like spreads of varying radius
    def make_seed(scale):
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0], [-1.0, 0.0],
            [0.5, 0.8660254], [-0.5, 0.8660254], [0.5, -0.8660254], [-0.5, -0.8660254],
            [2.0, 0.0], [-2.0, 0.0], [1.0, 1.7320508], [-1.0, 1.7320508],
        ]) * scale
        return pts

    global_best = (math.inf, None)
    budget = _DEADLINE - 20.0

    # Phase 1: many diverse coarse starts
    scales = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2]
    for sc in scales:
        for rep in range(2):
            if time.time() > budget:
                break
            start = make_seed(sc)
            start += rng.normal(0, 0.15, start.shape)
            res = _sa_from(start, budget)
            if res[0] < global_best[0]:
                global_best = res

    # Phase 2: refine around the best found, with several restarts
    if global_best[1] is not None:
        for rep in range(6):
            if time.time() > _DEADLINE - 30.0:
                break
            start = global_best[1].copy()
            start += rng.normal(0, 0.08, start.shape)
            res = _sa_from(start, _DEADLINE - 30.0)
            if res[0] < global_best[0]:
                global_best = res

    pos = global_best[1]
    if pos is None:
        # absolute fallback: guaranteed-large valid packing
        pos = make_seed(3.0)
        s = math.inf
    else:
        s, valid = _evaluate(_positions_to_vertices(pos))
        if not valid:
            pos = make_seed(3.0)
            s = _min_side(_positions_to_vertices(pos))

    # center the cluster at origin for the outer hexagon
    V = _positions_to_vertices(pos)
    c = V.reshape(-1, 2).mean(axis=0)
    pos = pos - c
    side, _ = _evaluate(_positions_to_vertices(pos))

    inner = np.zeros((11, 3))
    inner[:, 0] = pos[:, 0]
    inner[:, 1] = pos[:, 1]
    inner[:, 2] = 0.0
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(side)


# EVOLVE-BLOCK-END
