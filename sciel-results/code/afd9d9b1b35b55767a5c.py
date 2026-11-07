"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

N = 14
M = 35
MAXV = 10000

# Base value set: powers of two (products all distinct; sums collide on doubles)
BASE_VALUES = [1 << i for i in range(N)]  # 1..8192, all <=10000, distinct, positive


def _raw_value(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        a = values[i]
        b = values[j]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))


def _raw_parts(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        a = values[i]
        b = values[j]
        sums.add(a + b)
        prods.add(a * b)
    return len(sums), len(prods)


def _greedy_edges(values, m):
    """Pick m edges to maximize sum/product collisions (minimize distinct count)."""
    n = len(values)
    all_pairs = list(combinations(range(n), 2))
    chosen = []
    sums = set()
    prods = set()
    # Precompute pair sums/prods
    pair_info = [(i, j, values[i] + values[j], values[i] * values[j]) for i, j in all_pairs]
    for _ in range(m):
        best = None
        best_key = None
        for (i, j, s, p) in pair_info:
            # cost = number of NEW distinct values introduced
            new_s = 1 if s not in sums else 0
            new_p = 1 if p not in prods else 0
            # prefer reusing both; tie-break by smaller raw magnitude
            key = (new_s + new_p, s, p)
            if best_key is None or key < best_key:
                best_key = key
                best = (i, j, s, p)
        if best is None:
            break
        i, j, s, p = best
        chosen.append((i, j))
        sums.add(s)
        prods.add(p)
    return chosen


def _sa_optimize(values, edges, time_budget):
    """Simulated annealing over edge swaps to reduce raw value."""
    n = len(values)
    edges_set = set(tuple(e) for e in edges)
    all_pairs = list(combinations(range(n), 2))
    pair_set = set(all_pairs)
    cur = list(edges)
    cur_val = _raw_value(values, cur)
    best = list(cur)
    best_val = cur_val
    rng = random.Random(12345)
    t = 2.0
    cool = 0.999
    deadline = time.time() + time_budget
    while time.time() < deadline and t > 0.01:
        # pick a random edge to remove and a random non-edge to add
        i = rng.randrange(len(cur))
        e_rm = tuple(cur[i])
        # pick a candidate to add not already present
        for _ in range(20):
            cand = tuple(rng.choice(all_pairs))
            if cand not in edges_set:
                break
        else:
            t *= cool
            continue
        new_edges = [list(x) for x in cur if x != e_rm]
        new_edges.append(list(cand))
        new_val = _raw_value(values, new_edges)
        delta = new_val - cur_val
        if delta <= 0 or rng.random() < math.exp(-delta / t):
            cur = new_edges
            edges_set.discard(e_rm)
            edges_set.add(cand)
            cur_val = new_val
            if cur_val < best_val:
                best_val = cur_val
                best = [list(x) for x in cur]
        t *= cool
    return best, best_val


def construct():
    deadline = time.time() + 80.0  # safety margin under 90s
    best_result = None
    best_score = float('inf')

    # Phase 1: greedy edge selection on base values
    edges = _greedy_edges(BASE_VALUES, M)
    score = _raw_value(BASE_VALUES, edges)
    if score < best_score:
        best_score = score
        best_result = {'values': list(BASE_VALUES), 'edges': [list(e) for e in edges]}

    # Phase 2: SA polish on edges (values fixed at base)
    if time.time() < deadline - 5:
        budget = min(40.0, max(5.0, deadline - time.time() - 5))
        edges2, score2 = _sa_optimize(BASE_VALUES, edges, budget)
        if score2 < best_score:
            best_score = score2
            best_result = {'values': list(BASE_VALUES), 'edges': [list(e) for e in edges2]}

    # Phase 3: try a couple of alternative value sets with greedy+short SA
    alt_value_sets = []
    # set of small distinct ints (dense sums)
    alt_value_sets.append(list(range(1, N + 1)))
    # powers of two shifted
    alt_value_sets.append([2 * (1 << i) for i in range(N)])  # 2..16384 -> some >10000
    # filter valid
    alt_value_sets = [v for v in alt_value_sets if all(1 <= x <= MAXV for x in v) and len(set(v)) == N]

    for vals in alt_value_sets:
        if time.time() > deadline - 10:
            break
        e = _greedy_edges(vals, M)
        s = _raw_value(vals, e)
        if s < best_score:
            best_score = s
            best_result = {'values': list(vals), 'edges': [list(x) for x in e]}
        if time.time() < deadline - 8:
            e2, s2 = _sa_optimize(vals, e, 8.0)
            if s2 < best_score:
                best_score = s2
                best_result = {'values': list(vals), 'edges': [list(x) for x in e2]}

    if best_result is None:
        best_result = {'values': list(BASE_VALUES), 'edges': [list(e) for e in edges]}

    # Ensure exact types: Python int, list
    vals_out = [int(v) for v in best_result['values']]
    edges_out = [[int(i), int(j)] for i, j in best_result['edges']]
    return {'values': vals_out, 'edges': edges_out}
# EVOLVE-BLOCK-END
