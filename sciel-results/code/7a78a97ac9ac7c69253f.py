"""Structural rewrite: use structured value sets + optimal edge selection via collision-aware greedy,
with simulated annealing on values using fast incremental score."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

N = 14
M = 35
MAXV = 10000

def _raw_score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges():
    return list(itertools.combinations(range(N), 2))

def _edge_features(values, all_edges):
    """Return list of (edge, sum, product) for all edges."""
    feats = []
    for u, v in all_edges:
        a, b = values[u], values[v]
        feats.append(((u, v), a + b, a * b))
    return feats

def _select_edges_by_collision(feats, target_m):
    """Greedy: pick target_m edges to minimize max(distinct_sums, distinct_products).
    At each step, pick the edge whose addition causes the smallest increase in max(distinct)."""
    sums = set()
    prods = set()
    chosen = []
    remaining = list(feats)
    # Sort initially by how 'central' the sum/product is (appears more often) to break ties
    sum_count = {}
    prod_count = {}
    for _, s, p in feats:
        sum_count[s] = sum_count.get(s, 0) + 1
        prod_count[p] = prod_count.get(p, 0) + 1
    # Greedy with lookahead-free: pick edge minimizing resulting max(distinct)
    for _ in range(target_m):
        best_edge = None
        best_cost = float('inf')
        for idx, (e, s, p) in enumerate(remaining):
            new_sums = len(sums) + (0 if s in sums else 1)
            new_prods = len(prods) + (0 if p in prods else 1)
            cost = max(new_sums, new_prods)
            # Tie-break: prefer edges whose sum/prod appear more often (more future collisions)
            tie = -(sum_count[s] + prod_count[p])
            if (cost, tie) < (best_cost, 0) if best_edge is None else (cost, tie) < (best_cost, 0):
                best_cost = cost
                best_edge = idx
        # Simpler: track best properly
        best_edge = None
        best_key = None
        for idx, (e, s, p) in enumerate(remaining):
            new_sums = len(sums) + (0 if s in sums else 1)
            new_prods = len(prods) + (0 if p in prods else 1)
            cost = max(new_sums, new_prods)
            tie = -(sum_count[s] + prod_count[p])
            key = (cost, tie, idx)
            if best_key is None or key < best_key:
                best_key = key
                best_edge = idx
        e, s, p = remaining.pop(best_edge)
        chosen.append(e)
        sums.add(s)
        prods.add(p)
    return chosen

def _score_with_best_edges(values):
    """Given values, pick optimal 35 edges and return (score, edges)."""
    all_e = _all_edges()
    feats = _edge_features(values, all_e)
    edges = _select_edges_by_collision(feats, M)
    score = _raw_score(values, edges)
    return score, edges

def _make_geometric_values(scale=1, base=2):
    vals = []
    v = scale
    for i in range(N):
        if v > MAXV:
            break
        vals.append(v)
        v *= base
    while len(vals) < N:
        vals.append(vals[-1] * 2)
    # Ensure distinct and <= MAXV
    vals = sorted(set(vals[:N]))
    while len(vals) < N:
        vals.append(vals[-1] + 1)
    return vals[:N]

def _make_arithmetic_values(start=1, step=1):
    return [start + i * step for i in range(N)]

def _make_mixed_values():
    """Geometric with common factor to reduce product distinctness."""
    # values = d * b^i ; products = d^2 * b^(i+j) -> collide by i+j
    d = 1
    b = 2
    vals = [d * (b ** i) for i in range(N)]
    if max(vals) > MAXV:
        b = 2
        vals = [b ** i for i in range(N)]
    return vals

def _make_two_cluster_values():
    """Two clusters of values to create sum collisions."""
    # e.g., {1,2,3,4, 100,101,102,103, ...} - sums within cluster collide less,
    # but cross-cluster sums are large and distinct.
    # Better: use values where a_i + a_j = a_k + a_l for many pairs.
    # Arithmetic progression gives sum collisions: i+j = k+l.
    return list(range(1, N + 1))

def _random_values():
    return sorted(random.sample(range(1, MAXV + 1), N))

def _anneal_values(initial_vals, all_e, time_budget):
    """Simulated annealing on values with fast score via best-edge selection."""
    vals = initial_vals[:]
    score, edges = _score_with_best_edges(vals)
    best_vals = vals[:]
    best_score = score
    best_edges = edges[:]
    
    T = 1000.0
    cooling = 0.995
    start = time.time()
    
    while T > 1.0:
        if time.time() - start > time_budget:
            break
        # Perturb: change one value
        new_vals = vals[:]
        idx = random.randint(0, N - 1)
        old = new_vals[idx]
        # Try a range of new values
        delta = random.choice([-50, -10, -5, -2, -1, 1, 2, 5, 10, 50])
        new_vals[idx] = max(1, min(MAXV, old + delta))
        if new_vals[idx] == old:
            new_vals[idx] = max(1, min(MAXV, old + random.randint(-100, 100)))
        
        if len(set(new_vals)) < N or max(new_vals) > MAXV:
            T *= cooling
            continue
        
        new_score, new_edges = _score_with_best_edges(new_vals)
        delta_s = new_score - score
        if delta_s < 0 or random.random() < pow(2.718281828, -delta_s / max(T, 1e-9)):
            vals = new_vals
            score = new_score
            edges = new_edges
            if score < best_score:
                best_score = score
                best_vals = vals[:]
                best_edges = edges[:]
        T *= cooling
    
    return best_vals, best_score, best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin
    
    all_e = _all_edges()
    
    best_result = None
    best_score = float('inf')
    
    # Structured initializations
    inits = []
    # Geometric (powers of 2)
    inits.append(_make_geometric_values(1, 2))
    # Arithmetic
    inits.append(_make_arithmetic_values(1, 1))
    # Arithmetic with step 2
    inits.append(_make_arithmetic_values(1, 2))
    # Geometric base 3
    inits.append(_make_geometric_values(1, 3))
    # Two-cluster
    inits.append(_make_two_cluster_values())
    # Small values 1..14
    inits.append(list(range(1, N + 1)))
    # Values with common factor: 2,4,6,...,28
    inits.append([2 * (i + 1) for i in range(N)])
    # Values: 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192
    inits.append([2 ** i for i in range(N)])
    # Mixed: some small, some large
    inits.append([1, 2, 3, 4, 5, 6, 7, 8, 100, 200, 400, 800, 1600, 3200])
    
    # Random inits
    for _ in range(10):
        inits.append(_random_values())
    
    # Phase 1: Evaluate each init with optimal edge selection
    for vals in inits:
        if time.time() > deadline:
            break
        if len(set(vals)) < N or max(vals) > MAXV:
            continue
        score, edges = _score_with_best_edges(vals)
        if score < best_score:
            best_score = score
            best_result = {'values': vals[:], 'edges': [list(e) for e in edges]}
    
    # Phase 2: Anneal from best inits
    if best_result is None:
        best_result = {'values': list(range(1, N + 1)), 'edges': [list(e) for e in all_e[:M]]}
        best_score = _raw_score(best_result['values'], best_result['edges'])
    
    # Anneal from top few initializations
    init_scores = []
    for vals in inits[:8]:
        if len(set(vals)) < N or max(vals) > MAXV:
            continue
        s, e = _score_with_best_edges(vals)
        init_scores.append((s, vals))
    init_scores.sort(key=lambda x: x[0])
    
    for s0, v0 in init_scores[:4]:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        budget = min(15.0, remaining - 5)
        if budget < 3:
            continue
        av, ascore, aedges = _anneal_values(v0, all_e, budget)
        if ascore < best_score:
            best_score = ascore
            best_result = {'values': av[:], 'edges': [list(e) for e in aedges]}
    
    # Phase 3: Final local search on best result
    if best_result and time.time() < deadline - 5:
        vals = best_result['values'][:]
        edges = [tuple(e) for e in best_result['edges']]
        cur_score = best_score
        remaining = deadline - time.time() - 5
        if remaining > 3:
            # Quick local search: try perturbing values and re-selecting edges
            for _ in range(int(remaining * 50)):
                if time.time() > deadline - 5:
                    break
                new_vals = vals[:]
                idx = random.randint(0, N - 1)
                new_vals[idx] = max(1, min(MAXV, new_vals[idx] + random.randint(-20, 20)))
                if len(set(new_vals)) < N or max(new_vals) > MAXV:
                    continue
                ns, ne = _score_with_best_edges(new_vals)
                if ns < cur_score:
                    vals = new_vals
                    cur_score = ns
                    edges = ne
        if cur_score < best_score:
            best_score = cur_score
            best_result = {'values': vals[:], 'edges': [list(e) for e in edges]}
    
    # Ensure valid output
    if best_result is None:
        vals = list(range(1, N + 1))
        edges = [list(e) for e in all_e[:M]]
        best_result = {'values': vals, 'edges': edges}
    
    # Final validation and type conversion
    vals = best_result['values']
    edges = best_result['edges']
    best_result['values'] = [int(v) for v in vals]
    best_result['edges'] = [[int(a), int(b)] for a, b in edges]
    
    return best_result
# EVOLVE-BLOCK-END
