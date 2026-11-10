"""Construct a 14-node, 35-edge graph with 14 distinct positive ints <=10000 minimizing
max(#distinct edge sums, #distinct edge products)."""

import random
from itertools import combinations, product as iproduct


def _score(values, edges):
    if len(set(values)) != len(values):
        return 10**9
    sset = set()
    pset = set()
    for u, v in edges:
        sset.add(values[u] + values[v])
        pset.add(values[u] * values[v])
    return max(len(sset), len(pset))


def _rand_walk(vals, edges, iters=20000, temp=20.0, rng=random.Random(0)):
    best = vals
    best_s = _score(vals, edges)
    cur = list(vals)
    cur_s = best_s
    n = len(vals)
    for it in range(iters):
        t = temp * (1.0 - it / iters)
        move = rng.random()
        new = list(cur)
        if move < 0.7:
            # change one value
            i = rng.randrange(n)
            old = new[i]
            # pick new distinct value in 1..10000
            while True:
                cand = rng.randint(1, 10000)
                if cand != old and cand not in new:
                    break
            new[i] = cand
        elif move < 0.9:
            # swap two values
            i, j = rng.sample(range(n), 2)
            new[i], new[j] = new[j], new[i]
        else:
            # change two values
            i, j = rng.sample(range(n), 2)
            oi, oj = new[i], new[j]
            while True:
                ci = rng.randint(1, 10000)
                cj = rng.randint(1, 10000)
                if ci != oi and cj != oj and ci != cj and ci not in new and cj not in new:
                    break
            new[i], new[j] = ci, cj

        ns = _score(new, edges)
        if ns < cur_s or rng.random() < pow(2.718281828, (cur_s - ns) / max(t, 1e-6)):
            cur = new
            cur_s = ns
            if cur_s < best_s:
                best = list(cur)
                best_s = cur_s
                if best_s <= 14:
                    break
    return best, best_s


def _try_graph(values, edges_seed, iters=20000, temp=20.0, rng=random.Random(0)):
    best_vals, best_s = _rand_walk(values, edges_seed, iters=iters, temp=temp, rng=rng)
    return best_vals, best_s


def _full_graph():
    return list(combinations(range(14), 2))


def _almost_full_graph():
    # remove a few edges to reduce constraint (but task requires exactly 35 edges, so we can't)
    # Instead, for optimization we can try different edge sets of size 35
    all_edges = list(combinations(range(14), 2))
    rng = random.Random(0)
    rng.shuffle(all_edges)
    return all_edges[:35]


def construct():
    # We must return exactly 35 edges. We'll optimize over value assignments for a fixed
    # edge set (the first 35 of all combinations). We can also try several edge sets
    # to find one that allows fewer distinct sums/products.
    #
    # Strategy: try a few candidate edge sets (full graph is one; random subsets of size 35
    # are others). For each, do simulated annealing on values.
    #
    # Since 35 is exactly C(14,3)? No, C(14,2)=91. 35 is C(7,2)=21? No.
    # Actually, 35 = C(10,2)? No, C(10,2)=45. C(9,2)=36. Close.
    # Let's just try the "first 35" edge set and a couple of others.

    candidate_edge_sets = [
        list(combinations(range(14), 2))[:35],  # first 35 in lexicographic order
    ]

    # Add a few random edge sets of size 35
    rng = random.Random(42)
    for seed in [1, 7, 13]:
        r = random.Random(seed)
        all_e = list(combinations(range(14), 2))
        r.shuffle(all_e)
        candidate_edge_sets.append(all_e[:35])

    best_result = None
    best_score = 10**9

    for es in candidate_edge_sets:
        for trial_seed in [0, 1, 2]:
            rng = random.Random(trial_seed)
            # Start with a random initial value assignment
            init_vals = rng.sample(range(1, 10001), 14)
            vals, sc = _try_graph(init_vals, es, iters=15000, temp=15.0, rng=rng)
            if sc < best_score:
                best_score = sc
                best_result = (list(vals), [list(e) for e in es])
            if best_score <= 14:
                break
        if best_score <= 14:
            break

    if best_result is None:
        # Fallback: simple construction
        vals = list(range(1, 15))
        es = list(combinations(range(14), 2))[:35]
        best_result = (vals, [list(e) for e in es])

    values, edges = best_result
    # Ensure types
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    return {'values': values, 'edges': edges}
