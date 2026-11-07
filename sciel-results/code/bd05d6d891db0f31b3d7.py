"""Fresh design: Simulated Annealing on edge selection with diverse value sets."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000
ALL_EDGE_PAIRS = list(combinations(range(N), 2))  # 91 pairs
ALL_EDGE_SET = set(ALL_EDGE_PAIRS)

def _raw_value(values, edges):
    sums = set()
    prods = set()
    for (i, j) in edges:
        a = values[i]
        b = values[j]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _smoothed(values, edges, t):
    """Smooth surrogate: soft-max of distinct sum/product counts via log-sum-exp on counts."""
    sums = {}
    prods = {}
    for (i, j) in edges:
        a = values[i]
        b = values[j]
        s = a + b
        p = a * b
        sums[s] = sums.get(s, 0) + 1
        prods[p] = prods.get(p, 0) + 1
    # log-sum-exp of counts for sums
    sc = list(sums.values())
    pc = list(prods.values())
    def lse(arr):
        if not arr:
            return 0.0
        m = max(arr)
        return m + math.log(sum(math.exp(x - m) for x in arr))
    ls = lse(sc)
    lp = lse(pc)
    # soft max of the two
    mm = max(ls, lp)
    return mm + math.log(math.exp(ls - mm) + math.exp(lp - mm))

def _gen_value_set(rng):
    """Generate a random set of N distinct positive integers <= MAX_VAL."""
    # Mix strategies for diversity
    strat = rng.random()
    if strat < 0.3:
        # consecutive-ish
        start = rng.randint(1, 200)
        step = rng.randint(2, 30)
        vals = set()
        while len(vals) < N:
            vals.add(start + step * rng.randint(0, 50))
        vals = list(vals)
    elif strat < 0.5:
        # geometric
        base = rng.uniform(1.0, 5.0)
        ratio = rng.uniform(1.05, 1.8)
        vals = []
        for k in range(N):
            v = int(base * (ratio ** k))
            if 1 <= v <= MAX_VAL:
                vals.append(v)
        # fill if needed
        while len(vals) < N:
            v = rng.randint(1, MAX_VAL)
            if v not in vals:
                vals.append(v)
    elif strat < 0.7:
        # prime-heavy
        def is_prime(n):
            if n < 2:
                return False
            for p in range(2, int(n**0.5) + 1):
                if n % p == 0:
                    return False
            return True
        vals = []
        attempts = 0
        while len(vals) < N and attempts < 5000:
            v = rng.randint(2, MAX_VAL)
            if is_prime(v) and v not in vals:
                vals.append(v)
            attempts += 1
        while len(vals) < N:
            v = rng.randint(1, MAX_VAL)
            if v not in vals:
                vals.append(v)
    else:
        # uniform random
        vals = set()
        while len(vals) < N:
            vals.add(rng.randint(1, MAX_VAL))
        vals = list(vals)
    # ensure distinct and in range
    vals = list(set(v for v in vals if 1 <= v <= MAX_VAL))
    while len(vals) < N:
        vals.append(rng.randint(1, MAX_VAL))
    vals = vals[:N]
    # ensure distinct
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    while len(out) < N:
        v = rng.randint(1, MAX_VAL)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out[:N]

def _init_edges(rng):
    """Random 35-edge subset."""
    edges = set(rng.sample(ALL_EDGE_PAIRS, M))
    return edges

def _edge_swap(edges, rng):
    """Return a new edge set with one edge removed and one added."""
    edge_list = list(edges)
    rem = edge_list[rng.randrange(len(edge_list))]
    # pick a non-edge
    non_edges = [e for e in ALL_EDGE_PAIRS if e not in edges]
    if not non_edges:
        return edges
    add = rng.choice(non_edges)
    new_edges = set(edges)
    new_edges.discard(rem)
    new_edges.add(add)
    return new_edges

def _sa_search(values, rng, deadline):
    """Simulated annealing on edge set."""
    edges = _init_edges(rng)
    best_edges = set(edges)
    best_val = _raw_value(values, best_edges)
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    steps = 0
    while time.time() < deadline:
        new_edges = _edge_swap(edges, rng)
        new_smooth = _smoothed(values, new_edges, T)
        old_smooth = _smoothed(values, edges, T)
        delta = new_smooth - old_smooth
        if delta < 0 or rng.random() < math.exp(-delta / T):
            edges = new_edges
            new_val = _raw_value(values, edges)
            if new_val < best_val:
                best_val = new_val
                best_edges = set(edges)
        T = max(T_min, T * cooling)
        steps += 1
        if steps % 5000 == 0:
            # occasional restart
            if rng.random() < 0.1:
                edges = _init_edges(rng)
    return best_edges, best_val

def construct():
    start = time.time()
    deadline = start + 80.0  # 10s safety margin
    rng = random.Random(42)
    best_overall = None
    best_score = float('inf')
    # Generate diverse value sets
    value_sets = []
    for _ in range(8):
        vs = _gen_value_set(rng)
        value_sets.append(vs)
    # Also include a simple consecutive set
    value_sets.append(list(range(1, N + 1)))
    for vs in value_sets:
        if time.time() > deadline - 5.0:
            break
        # Quick greedy init
        edges = _init_edges(rng)
        cur_val = _raw_value(vs, edges)
        # Local improvement: try to swap edges to reduce
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            edge_list = list(edges)
            for rem in edge_list:
                if time.time() > deadline - 5.0:
                    break
                non_edges = [e for e in ALL_EDGE_PAIRS if e not in edges]
                for add in non_edges:
                    if time.time() > deadline - 5.0:
                        break
                    new_edges = set(edges)
                    new_edges.discard(rem)
                    new_edges.add(add)
                    nv = _raw_value(vs, new_edges)
                    if nv < cur_val:
                        edges = new_edges
                        cur_val = nv
                        improved = True
                        break
        # Then SA
        sa_edges, sa_val = _sa_search(vs, rng, deadline - 5.0)
        if sa_val < cur_val:
            edges = sa_edges
            cur_val = sa_val
        if cur_val < best_score:
            best_score = cur_val
            best_overall = (list(vs), list(edges))
    if best_overall is None:
        # fallback
        values = list(range(1, N + 1))
        edges = [list(e) for e in ALL_EDGE_PAIRS[:M]]
        return {'values': values, 'edges': edges}
    values, edges = best_overall
    # Ensure types
    values = [int(v) for v in values]
    edges = [[int(i), int(j)] for (i, j) in edges]
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
