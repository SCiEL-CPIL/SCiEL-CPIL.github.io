"""Combined: deterministic hexagonal-blob construction + fast local search + simulated annealing.
Target: 32 points on triangular lattice, maximize unit-distance edges (Harborth optimum 76)."""
import time
import math
import random

_START = None
_DEADLINE = None

def _now():
    return time.time()

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _hexdist(p, q):
    da = p[0] - q[0]
    db = p[1] - q[1]
    return (abs(da) + abs(da + db) + abs(db)) // 2

def _edge_count(Sset):
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    if len(Sset) != 32:
        return False
    return all(abs(a) <= 64 and abs(b) <= 64 for a, b in Sset)

def _local_search(Sset, pool, swap_cand, deadline, rng, max_passes=300):
    """Swap-based local search: try to improve edge count by swapping one point
    for another within a small radius."""
    S = list(Sset)
    Sset = set(S)
    cur = _edge_count(Sset)
    for _ in range(max_passes):
        if _now() > deadline - 1.0:
            break
        improved = False
        order = list(range(len(S)))
        rng.shuffle(order)
        for i in order:
            p = S[i]
            for q in swap_cand[p]:
                if q in Sset:
                    continue
                newS = set(Sset)
                newS.discard(p)
                newS.add(q)
                ne = _edge_count(newS)
                if ne > cur:
                    Sset = newS
                    S[i] = q
                    cur = ne
                    improved = True
                    break
        if not improved:
            break
    return Sset, cur

def _simulated_annealing(Sset, pool, swap_cand, deadline, rng, max_iter=8000):
    """Simulated annealing with swap moves to escape local optima."""
    S = list(Sset)
    Sset = set(S)
    cur = _edge_count(Sset)
    best = cur
    best_Sset = set(Sset)
    T = 3.0
    cooling = 0.9995

    for _ in range(max_iter):
        if _now() > deadline - 1.0:
            break
        i = rng.randrange(len(S))
        p = S[i]
        if not swap_cand[p]:
            continue
        q = rng.choice(swap_cand[p])
        if q in Sset:
            continue

        newS = set(Sset)
        newS.discard(p)
        newS.add(q)
        ne = _edge_count(newS)
        delta = ne - cur

        if delta >= 0 or rng.random() < math.exp(delta / T):
            Sset = newS
            S[i] = q
            cur = ne
            if cur > best:
                best = cur
                best_Sset = set(Sset)

        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_Sset, best

def construct():
    global _START, _DEADLINE
    if _START is None:
        _START = time.time()
        _DEADLINE = _START + 110.0  # safety margin before 120s contract limit

    rng = random.Random(20240517)

    # Candidate pool: hexagon of radius 6 (plenty of room; all coords <= 6 <= 64).
    pool = []
    for a in range(-6, 7):
        for b in range(-6, 7):
            if (abs(a) + abs(a + b) + abs(b)) // 2 <= 6:
                pool.append((a, b))
    poolset = set(pool)

    # Precompute swap candidates: pool points within hex-distance 2 of each pool point.
    swap_cand = {}
    for p in pool:
        swap_cand[p] = [q for q in pool if q != p and _hexdist(p, q) <= 2]

    best = None
    best_edges = -1

    def consider(Sset):
        nonlocal best, best_edges
        if _valid(Sset):
            e = _edge_count(Sset)
            if e > best_edges:
                best_edges = e
                best = [list(t) for t in Sset]

    # Seed sets: hexagon radius 2 (19 pts), plus random seeds.
    center = (0, 0)
    hex2 = [p for p in pool if _hexdist(p, center) <= 2]  # 19 points
    hex3 = [p for p in pool if _hexdist(p, center) <= 3]  # 37 points

    def grow_to_32(seed):
        Sset = set(seed)
        Sset = {p for p in Sset if p in poolset}
        if len(Sset) >= 32:
            return set(list(Sset)[:32])
        while len(Sset) < 32:
            if _now() > _DEADLINE - 3.0:
                return Sset
            best_pts = []
            best_new = -1
            for q in pool:
                if q in Sset:
                    continue
                new_edges = sum(1 for n in _neighbors(q) if n in Sset)
                if new_edges > best_new:
                    best_new = new_edges
                    best_pts = [q]
                elif new_edges == best_new:
                    best_pts.append(q)
            if not best_pts:
                break
            Sset.add(rng.choice(best_pts))
        return Sset

    # Phase 1: deterministic seeds + local search
    seeds = [
        list(hex2),
        list(hex3[:32]),
        [(0, 0), (1, 0), (0, 1)],
    ]
    for _ in range(6):
        seeds.append(rng.sample(hex2, rng.randint(3, 8)))

    for seed in seeds:
        if _now() > _DEADLINE - 8.0:
            break
        Sset = grow_to_32(seed)
        if len(Sset) != 32:
            continue
        Sset, _ = _local_search(Sset, pool, swap_cand, _DEADLINE, rng)
        consider(Sset)
        if best_edges >= 76:
            break

    # Phase 2: simulated annealing from best found so far
    if best_edges < 76 and best is not None:
        if _now() < _DEADLINE - 10.0:
            Sset, _ = _simulated_annealing(set(best), pool, swap_cand, _DEADLINE, rng)
            consider(Sset)

    # Phase 3: additional randomized restarts while time remains
    while _now() < _DEADLINE - 5.0:
        seed = rng.sample(hex3, rng.randint(4, 10))
        Sset = grow_to_32(seed)
        if len(Sset) != 32:
            continue
        Sset, _ = _local_search(Sset, pool, swap_cand, _DEADLINE, rng)
        consider(Sset)
        if best_edges >= 76:
            break
        # Occasionally try SA from a different starting point
        if _now() < _DEADLINE - 10.0 and rng.random() < 0.3:
            Sset_sa, _ = _simulated_annealing(Sset, pool, swap_cand, _DEADLINE, rng)
            consider(Sset_sa)
            if best_edges >= 76:
                break

    # Final polish: intensive local search on best solution
    if best is not None and best_edges < 76 and _now() < _DEADLINE - 2.0:
        Sset, _ = _local_search(set(best), pool, swap_cand, _DEADLINE, rng, max_passes=500)
        consider(Sset)

    if best is None:
        # Guaranteed-valid fallback
        best = [[a, b] for a in range(16) for b in range(2)]

    # Final sanity checks
    assert len(best) == 32
    assert len(set(tuple(p) for p in best)) == 32
    assert all(abs(p[0]) <= 64 and abs(p[1]) <= 64 for p in best)
    return best
