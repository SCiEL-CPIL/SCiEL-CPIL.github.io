"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 14
M = 35
MAXV = 10000

def _score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods)), len(sums), len(prods)

def _optimize_edges(values, deadline=None):
    """Greedy + local search on edges given fixed values."""
    all_edges = list(combinations(range(N), 2))
    best_edges = None
    best_score = None

    # Greedy construction
    used = set()
    edges = []
    sums = set()
    prods = set()
    # Precompute
    edge_sum = {}
    edge_prod = {}
    for i, j in all_edges:
        edge_sum[(i, j)] = values[i] + values[j]
        edge_prod[(i, j)] = values[i] * values[j]

    for _ in range(M):
        best_e = None
        best_new = None
        best_key = None
        for e in all_edges:
            if e in used:
                continue
            s = edge_sum[e]
            p = edge_prod[e]
            new_s = 0 if s in sums else 1
            new_p = 0 if p in prods else 1
            key = (new_s + new_p, new_p, new_s, random.random())
            if best_key is None or key < best_key:
                best_key = key
                best_e = e
                best_new = (s, p)
        if best_e is None:
            break
        used.add(best_e)
        edges.append(best_e)
        sums.add(best_new[0])
        prods.add(best_new[1])

    cur_score, _, _ = _score(values, edges)
    best_edges = [list(e) for e in edges]
    best_score = cur_score

    # Local search on edges: swap one edge for another
    if deadline is not None and time.time() > deadline:
        return best_edges, best_score

    edges_set = set(edges)
    improved = True
    iters = 0
    max_iters = 200
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        edges_list = list(edges)
        random.shuffle(edges_list)
        for idx, e in enumerate(edges_list):
            # Try removing e and adding a different edge
            new_edges = edges_list[:idx] + edges_list[idx+1:]
            base_sums = set()
            base_prods = set()
            for a, b in new_edges:
                base_sums.add(values[a] + values[b])
                base_prods.add(values[a] * values[b])
            best_add = None
            best_key = None
            for f in all_edges:
                if f in edges_set:
                    continue
                s = values[f[0]] + values[f[1]]
                p = values[f[0]] * values[f[1]]
                new_s = 0 if s in base_sums else 1
                new_p = 0 if p in base_prods else 1
                key = (new_s + new_p, new_p, new_s, random.random())
                if best_key is None or key < best_key:
                    best_key = key
                    best_add = f
            if best_add is None:
                continue
            # Evaluate
            new_edges_full = new_edges + [best_add]
            sc, _, _ = _score(values, new_edges_full)
            if sc < cur_score:
                edges = new_edges_full
                edges_set = set(edges)
                cur_score = sc
                if cur_score < best_score:
                    best_score = cur_score
                    best_edges = [list(e) for e in edges]
                improved = True
                break
    return best_edges, best_score

def _local_search_values(values, edges, deadline):
    """Local search on values, re-optimizing edges."""
    best = list(values)
    best_edges = [list(e) for e in edges]
    best_score, _, _ = _score(best, best_edges)

    iters = 0
    max_iters = 500
    while iters < max_iters:
        iters += 1
        if time.time() > deadline:
            break
        # Perturb one value
        idx = random.randrange(N)
        old = best[idx]
        # Try a few random replacements
        found = False
        for _ in range(30):
            new_val = random.randint(1, MAXV)
            if new_val == old:
                continue
            if new_val in best:
                continue
            cand = list(best)
            cand[idx] = new_val
            cand_edges, cand_score = _optimize_edges(cand, deadline)
            if cand_score < best_score:
                best = cand
                best_edges = cand_edges
                best_score = cand_score
                found = True
                break
        if not found:
            # Try a small perturbation
            for delta in [1, -1, 2, -2, 3, -3, 5, -5, 10, -10]:
                new_val = old + delta
                if new_val < 1 or new_val > MAXV:
                    continue
                if new_val in best:
                    continue
                cand = list(best)
                cand[idx] = new_val
                cand_edges, cand_score = _optimize_edges(cand, deadline)
                if cand_score < best_score:
                    best = cand
                    best_edges = cand_edges
                    best_score = cand_score
                    found = True
                    break

    return best, best_edges, best_score

def construct():
    start = time.time()
    deadline = start + 75  # safety margin

    random.seed(42)

    # Initial candidate: use divisors of a highly composite number
    # 5040 has many divisors. Let's pick 14 distinct divisors.
    # Divisors of 5040: 1,2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,21,24,...
    # Let's pick values that allow many shared products.
    # Use pairs that multiply to a common value.
    # Strategy: pick 7 pairs (a, C/a) for several C values.

    # Simpler: start with a random structured set and local search.
    best_values = None
    best_edges = None
    best_score = 10**9

    # Try several initializations
    inits = []

    # Init 1: divisors of 5040
    divs_5040 = [d for d in range(1, 5041) if 5040 % d == 0]
    if len(divs_5040) >= 14:
        inits.append(divs_5040[:14])

    # Init 2: powers of 2 and 3
    vals = set()
    for a in range(5):
        for b in range(5):
            v = (2**a) * (3**b)
            if 1 <= v <= MAXV:
                vals.add(v)
    if len(vals) >= 14:
        inits.append(sorted(vals)[:14])

    # Init 3: random
    for _ in range(5):
        inits.append(random.sample(range(1, MAXV+1), 14))

    # Init 4: structured - two groups sharing products
    # Group A: 7 small numbers, Group B: 7 numbers = C / a_i
    for C in [100, 200, 500, 1000, 2000, 5000]:
        group_a = random.sample(range(2, 50), 7)
        group_b = []
        valid = True
        for a in group_a:
            if C % a == 0:
                group_b.append(C // a)
            else:
                valid = False
                break
        if valid and len(set(group_a + group_b)) == 14 and all(1 <= v <= MAXV for v in group_a + group_b):
            inits.append(group_a + group_b)

    for init_vals in inits:
        if time.time() > deadline:
            break
        vals = list(init_vals)
        edges, score = _optimize_edges(vals, deadline)
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges

    # Local search on values
    if time.time() < deadline - 5:
        best_values, best_edges, best_score = _local_search_values(best_values, best_edges, deadline)

    # Final polish: try a few more random restarts if time permits
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline:
                break
            vals = random.sample(range(1, MAXV+1), 14)
            edges, score = _optimize_edges(vals, deadline)
            if score < best_score:
                best_score = score
                best_values = vals
                best_edges = edges

    # Ensure types are correct
    values_out = [int(v) for v in best_values]
    edges_out = [[int(i), int(j)] for i, j in best_edges]

    return {'values': values_out, 'edges': edges_out}
# EVOLVE-BLOCK-END
