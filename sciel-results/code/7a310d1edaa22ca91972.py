"""Fresh design: factor-heavy values + greedy collision-maximizing edges + local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 14
M = 35
MAXV = 10000
ALL_EDGES = list(combinations(range(N), 2))  # 91 possible edges


def _score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        a = values[i]
        b = values[j]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))


def _score_parts(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        a = values[i]
        b = values[j]
        sums.add(a + b)
        prods.add(a * b)
    return len(sums), len(prods)


def _greedy_edges(values, seed_rng):
    """Pick M edges maximizing product collisions (tiebreak by sum collisions)."""
    # Precompute product and sum for every edge
    edge_prod = {}
    edge_sum = {}
    for (i, j) in ALL_EDGES:
        a = values[i]
        b = values[j]
        edge_prod[(i, j)] = a * b
        edge_sum[(i, j)] = a + b

    chosen = []
    prod_count = {}
    sum_count = {}
    # Sort candidate edges by a heuristic: prefer edges whose product is already
    # common or can become common. We do a greedy: at each step pick the edge that
    # maximizes (number of already-chosen edges sharing its product) then (sharing sum).
    remaining = list(ALL_EDGES)
    # Precompute for each edge the set of other edges sharing product / sum for speed
    # Build product->edges and sum->edges maps
    prod_to_edges = {}
    sum_to_edges = {}
    for (i, j) in ALL_EDGES:
        p = edge_prod[(i, j)]
        s = edge_sum[(i, j)]
        prod_to_edges.setdefault(p, []).append((i, j))
        sum_to_edges.setdefault(s, []).append((i, j))

    chosen_set = set()
    for _ in range(M):
        best_edge = None
        best_key = None
        for (i, j) in remaining:
            if (i, j) in chosen_set:
                continue
            p = edge_prod[(i, j)]
            s = edge_sum[(i, j)]
            # count how many chosen edges share this product / sum
            prod_share = 0
            for e2 in prod_to_edges[p]:
                if e2 in chosen_set:
                    prod_share += 1
            sum_share = 0
            for e2 in sum_to_edges[s]:
                if e2 in chosen_set:
                    sum_share += 1
            # key: maximize product share, then sum share
            key = (prod_share, sum_share)
            if best_key is None or key > best_key:
                best_key = key
                best_edge = (i, j)
        chosen.append(best_edge)
        chosen_set.add(best_edge)
        remaining.remove(best_edge)
    return chosen


def _refine(values, edges, budget, rng):
    """Local search improving raw value by swapping values/edges."""
    cur = _score(values, edges)
    values = list(values)
    edges = [list(e) for e in edges]
    # candidate pool of extra values to swap in
    pool = [v for v in range(1, MAXV + 1) if v not in set(values)]
    # keep pool bounded for speed
    if len(pool) > 400:
        pool = rng.sample(pool, 400)

    deadline = time.time() + budget
    edges_set = set(tuple(e) for e in edges)

    def try_accept(new_vals, new_edges):
        nonlocal cur, values, edges, edges_set
        s, p = _score_parts(new_vals, new_edges)
        new_raw = max(s, p)
        if new_raw < cur:
            cur = new_raw
            values = new_vals
            edges = [list(e) for e in new_edges]
            edges_set = set(tuple(e) for e in new_edges)
            return True
        return False

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # --- value swap moves ---
        for idx in range(N):
            for cand in pool:
                if cand == values[idx]:
                    continue
                new_vals = list(values)
                new_vals[idx] = cand
                if len(set(new_vals)) != N:
                    continue
                if try_accept(new_vals, edges):
                    # update pool: remove cand, add old value
                    pool = [v for v in pool if v != cand]
                    pool.append(values[idx])
                    improved = True
                    break
            if improved:
                break
        if time.time() >= deadline:
            break
        # --- edge swap moves: replace one edge with another ---
        for _ in range(200):
            if time.time() >= deadline:
                break
            old = rng.choice(edges)
            # pick a candidate edge not currently present
            cand = rng.choice(ALL_EDGES)
            if tuple(cand) in edges_set:
                continue
            new_edges = [e for e in edges if e != old]
            new_edges.append(list(cand))
            if try_accept(values, new_edges):
                improved = True
    return values, edges, cur


def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin under 90s
    rng = random.Random(12345)

    # Build a factor-heavy candidate value set: multiples of small primes
    # We want 14 distinct values with lots of shared factors.
    # Strategy: take multiples of 2,3,4,5,6,7 within range, ensure distinctness.
    base_factors = [2, 3, 4, 5, 6, 7]
    candidates = set()
    for f in base_factors:
        for k in range(1, 100):
            v = f * k
            if 1 <= v <= MAXV:
                candidates.add(v)
    candidates = sorted(candidates)

    # Pick 14 values that maximize internal product-collision potential.
    # Greedy: start with a seed, add value that maximizes collisions with existing.
    def collision_score(vals, new_v):
        # count how many pairs among vals+new_v share a product
        prods = {}
        for a in vals:
            for b in vals:
                if a != b:
                    prods[a * b] = prods.get(a * b, 0) + 1
        # add new_v
        for a in vals:
            p = a * new_v
            prods[p] = prods.get(p, 0) + 1
        # sum of (count choose 2) over products = number of colliding pairs
        total = 0
        for c in prods.values():
            total += c * (c - 1) // 2
        return total

    selected = []
    # start with a random-ish seed from candidates
    seed_val = rng.choice(candidates[:200])
    selected.append(seed_val)
    while len(selected) < N:
        best_v = None
        best_s = -1
        # sample a subset of candidates for speed
        sample = rng.sample(candidates, min(len(candidates), 300))
        for v in sample:
            if v in selected:
                continue
            s = collision_score(selected, v)
            if s > best_s:
                best_s = s
                best_v = v
        if best_v is None:
            # fallback: pick any unused
            for v in candidates:
                if v not in selected:
                    best_v = v
                    break
        selected.append(best_v)
        if time.time() > deadline - 20:
            break
    # ensure 14 distinct
    while len(selected) < N:
        for v in range(1, MAXV + 1):
            if v not in selected:
                selected.append(v)
                break
    values = sorted(selected[:N])

    # Initial edges via greedy collision maximization
    edges = _greedy_edges(values, rng)
    cur = _score(values, edges)

    # Refine with local search
    remaining = deadline - time.time() - 5.0
    if remaining > 2.0:
        values, edges, cur = _refine(values, edges, remaining, rng)

    # Final validity check
    values = [int(v) for v in values]
    edges = [[int(i), int(j)] for i, j in edges]
    assert len(values) == N and len(set(values)) == N
    assert all(1 <= v <= MAXV for v in values)
    assert len(edges) == M
    seen = set()
    for i, j in edges:
        assert i != j
        t = (min(i, j), max(i, j))
        assert t not in seen
        seen.add(t)

    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
