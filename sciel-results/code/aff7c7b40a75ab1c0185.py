"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations
from math import log, exp

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
    """Fast greedy + local search on edges given fixed values."""
    all_edges = list(combinations(range(N), 2))
    # Precompute sum and product for every edge
    edge_data = {}
    for i, j in all_edges:
        edge_data[(i, j)] = (values[i] + values[j], values[i] * values[j])

    # Greedy construction: pick edges that reuse existing sums/products
    used = set()
    edges = []
    sums = set()
    prods = set()
    for _ in range(M):
        best_e = None
        best_key = None
        for e in all_edges:
            if e in used:
                continue
            s, p = edge_data[e]
            new_s = 0 if s in sums else 1
            new_p = 0 if p in prods else 1
            key = (new_s + new_p, new_p, new_s)
            if best_key is None or key < best_key:
                best_key = key
                best_e = e
        if best_e is None:
            break
        used.add(best_e)
        edges.append(best_e)
        s, p = edge_data[best_e]
        sums.add(s)
        prods.add(p)

    cur_score = max(len(sums), len(prods))
    best_edges = [list(e) for e in edges]
    best_score = cur_score

    # Local search: swap one edge for another
    if deadline is not None and time.time() > deadline:
        return best_edges, best_score

    edges_set = set(edges)
    improved = True
    iters = 0
    max_iters = 300
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        edges_list = list(edges)
        random.shuffle(edges_list)
        for idx, e in enumerate(edges_list):
            # Remove edge e, compute base sums/prods
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
                s, p = edge_data[f]
                new_s = 0 if s in base_sums else 1
                new_p = 0 if p in base_prods else 1
                key = (new_s + new_p, new_p, new_s)
                if best_key is None or key < best_key:
                    best_key = key
                    best_add = f
            if best_add is None:
                continue
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
    max_iters = 300
    while iters < max_iters:
        iters += 1
        if time.time() > deadline:
            break
        idx = random.randrange(N)
        old = best[idx]
        found = False
        # Try small deltas first (more likely to find improvements)
        for delta in [1, -1, 2, -2, 3, -3, 5, -5, 7, -7, 10, -10, 15, -15, 20, -20, 30, -30, 50, -50]:
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
        if not found:
            # Try a few random values
            for _ in range(5):
                new_val = random.randint(1, MAXV)
                if new_val == old or new_val in best:
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

def _build_collision_pool():
    """Build a candidate pool of integers with high collision density.
    Integers that are divisors of highly composite numbers have many
    multiplicative relationships, and small integers have many additive
    relationships."""
    pool = set()
    
    # Divisors of highly composite numbers
    for hcn in [5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400]:
        for d in range(1, int(hcn**0.5) + 1):
            if hcn % d == 0:
                if d <= MAXV:
                    pool.add(d)
                if hcn // d <= MAXV:
                    pool.add(hcn // d)
    
    # Small integers (high additive collision)
    for v in range(1, 200):
        pool.add(v)
    
    # Powers and products of small primes
    primes = [2, 3, 5, 7, 11]
    for a in range(6):
        for b in range(5):
            for c in range(4):
                v = (2**a) * (3**b) * (5**c)
                if 1 <= v <= MAXV:
                    pool.add(v)
    
    # Products of pairs of small numbers
    for i in range(2, 30):
        for j in range(i, 30):
            v = i * j
            if 1 <= v <= MAXV:
                pool.add(v)
    
    return sorted(pool)

def _collision_score_pair(a, b):
    """Score how well two values collide: count shared sums and products
    with other candidate pairs (approximated by checking if a+b and a*b
    can be formed by other pairs in the pool)."""
    # For efficiency, we'll use a simpler heuristic:
    # - Smaller values have more additive collisions
    # - Values with many divisors have more multiplicative collisions
    score = 0
    score += (200 - min(a, b))  # Prefer smaller values for additive collisions
    # Count divisors (proxy for multiplicative collision density)
    div_count = 0
    for d in range(1, int(min(a, b)**0.5) + 1):
        if min(a, b) % d == 0:
            div_count += 1
            if d != min(a, b) // d:
                div_count += 1
    score += div_count * 5
    return score

def _greedy_value_selection(pool, deadline):
    """Select 14 values from pool using greedy collision-density approach."""
    if time.time() > deadline:
        return None
    
    selected = []
    selected_set = set()
    
    # Precompute pair collision scores for pool
    # For efficiency, use a heuristic score
    scored_pool = []
    for v in pool:
        # Heuristic: prefer values that are divisors of many small numbers
        # and have many divisors themselves
        div_count = 0
        for d in range(1, int(v**0.5) + 1):
            if v % d == 0:
                div_count += 1
                if d != v // d:
                    div_count += 1
        # Also check if v is a divisor of 5040, 7560, etc.
        hcn_divisor_count = 0
        for hcn in [5040, 7560, 10080, 15120, 20160, 25200]:
            if hcn % v == 0:
                hcn_divisor_count += 1
        score = div_count * 3 + hcn_divisor_count * 10 + (200 - min(v, 200))
        scored_pool.append((score, v))
    
    scored_pool.sort(key=lambda x: -x[0])
    
    # Greedy selection with collision checking
    for score, v in scored_pool:
        if len(selected) >= N:
            break
        if v in selected_set:
            continue
        # Check if adding v creates good collisions with existing selected values
        if selected:
            good_collision = False
            for sv in selected:
                s = v + sv
                p = v * sv
                # Check if this sum or product can be formed by other pairs
                for sv2 in selected:
                    if sv2 == sv:
                        continue
                    if v + sv2 == s or v * sv2 == p:
                        good_collision = True
                        break
                    if sv + sv2 == s or sv * sv2 == p:
                        good_collision = True
                        break
                if good_collision:
                    break
            if not good_collision and len(selected) >= 3:
                # Still add if score is high enough
                if score < 30:
                    continue
        selected.append(v)
        selected_set.add(v)
    
    if len(selected) < N:
        # Fill with remaining high-score values
        for score, v in scored_pool:
            if len(selected) >= N:
                break
            if v not in selected_set:
                selected.append(v)
                selected_set.add(v)
    
    return selected[:N]

def _simulated_annealing(values, edges, deadline):
    """Simulated annealing over values with edge re-optimization."""
    best = list(values)
    best_edges = [list(e) for e in edges]
    best_score, _, _ = _score(best, best_edges)
    
    current = list(best)
    current_edges = [list(e) for e in best_edges]
    current_score, _, _ = _score(current, current_edges)
    
    T = 30.0
    cooling = 0.99
    iters = 0
    max_iters = 800
    
    while iters < max_iters:
        iters += 1
        if time.time() > deadline:
            break
        T *= cooling
        if T < 0.5:
            break
        
        # Perturb: change one value
        idx = random.randrange(N)
        old = current[idx]
        # Bias toward small perturbations
        if random.random() < 0.5:
            delta = random.choice([1, -1, 2, -2, 3, -3, 5, -5, 10, -10, 20, -20])
            new_val = old + delta
        else:
            new_val = random.randint(1, MAXV)
        if new_val < 1 or new_val > MAXV or new_val == old or new_val in current:
            continue
        cand = list(current)
        cand[idx] = new_val
        
        cand_edges, cand_score = _optimize_edges(cand, deadline)
        
        # Accept or reject
        delta = cand_score - current_score
        if delta < 0 or random.random() < exp(-delta / max(T, 0.01)):
            current = cand
            current_edges = cand_edges
            current_score = cand_score
            if current_score < best_score:
                best = list(current)
                best_edges = [list(e) for e in current_edges]
                best_score = current_score
    
    return best, best_edges, best_score

def _polish(values, edges, deadline):
    """Final polish: try small perturbations and edge swaps."""
    best = list(values)
    best_edges = [list(e) for e in edges]
    best_score, _, _ = _score(best, best_edges)
    
    all_edges = list(combinations(range(N), 2))
    edge_data = {}
    for i, j in all_edges:
        edge_data[(i, j)] = (best[i] + best[j], best[i] * best[j])
    
    iters = 0
    max_iters = 200
    while iters < max_iters:
        iters += 1
        if time.time() > deadline:
            break
        
        # Try small value perturbations
        idx = random.randrange(N)
        old = best[idx]
        for delta in [1, -1, 2, -2, 3, -3, 5, -5, 7, -7, 10, -10, 15, -15, 20, -20]:
            new_val = old + delta
            if new_val < 1 or new_val > MAXV or new_val in best:
                continue
            cand = list(best)
            cand[idx] = new_val
            cand_edges, cand_score = _optimize_edges(cand, deadline)
            if cand_score < best_score:
                best = cand
                best_edges = cand_edges
                best_score = cand_score
                # Rebuild edge_data
                edge_data = {}
                for i, j in all_edges:
                    edge_data[(i, j)] = (best[i] + best[j], best[i] * best[j])
                break
        
        # Try edge swaps
        if random.random() < 0.5:
            edges_list = list(best_edges)
            random.shuffle(edges_list)
            edges_set = set(tuple(e) for e in edges_list)
            for idx_e in range(len(edges_list)):
                e = edges_list[idx_e]
                new_edges = edges_list[:idx_e] + edges_list[idx_e+1:]
                base_sums = set()
                base_prods = set()
                for a, b in new_edges:
                    base_sums.add(best[a] + best[b])
                    base_prods.add(best[a] * best[b])
                best_add = None
                best_key = None
                for f in all_edges:
                    if f in edges_set:
                        continue
                    s, p = edge_data[f]
                    new_s = 0 if s in base_sums else 1
                    new_p = 0 if p in base_prods else 1
                    key = (new_s + new_p, new_p, new_s)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_add = f
                if best_add is not None:
                    new_edges_full = new_edges + [best_add]
                    sc, _, _ = _score(best, new_edges_full)
                    if sc < best_score:
                        best_edges = new_edges_full
                        best_score = sc
                        break
    
    return best, best_edges, best_score

def construct():
    start = time.time()
    deadline = start + 80  # Safety margin before 90s kill
    
    random.seed(42)
    
    best_values = None
    best_edges = None
    best_score = 10**9
    
    # Build collision pool
    pool = _build_collision_pool()
    
    # Multiple initialization strategies
    inits = []
    
    # Init 1: Greedy collision-density selection
    greedy_vals = _greedy_value_selection(pool, deadline)
    if greedy_vals:
        inits.append(greedy_vals)
    
    # Init 2: Divisors of 5040
    divs_5040 = [d for d in range(1, 5041) if 5040 % d == 0]
    if len(divs_5040) >= 14:
        inits.append(divs_5040[:14])
    
    # Init 3: Small integers
    inits.append(list(range(1, 15)))
    
    # Init 4: Powers of 2 and 3
    vals = set()
    for a in range(5):
        for b in range(5):
            v = (2**a) * (3**b)
            if 1 <= v <= MAXV:
                vals.add(v)
    if len(vals) >= 14:
        inits.append(sorted(vals)[:14])
    
    # Init 5: Random from pool
    for _ in range(8):
        if len(pool) >= 14:
            inits.append(random.sample(pool, 14))
    
    # Init 6: Structured pairs (product = constant)
    for C in [120, 180, 240, 360, 480, 600, 720, 840, 960, 1200, 1440, 1680, 1920, 2400, 2880, 3360, 4320, 5040]:
        divs = [d for d in range(2, C // 2 + 1) if C % d == 0]
        if len(divs) >= 14:
            # Pick 7 divisors and their complements
            random.shuffle(divs)
            group_a = divs[:7]
            group_b = [C // a for a in group_a]
            combined = group_a + group_b
            if len(set(combined)) == 14 and all(1 <= v <= MAXV for v in combined):
                inits.append(combined)
    
    # Init 7: Arithmetic-like sequences with small step
    for base in [1, 2, 3, 5, 7]:
        for step in [1, 2, 3, 4, 5]:
            seq = [base + i * step for i in range(N)]
            if all(1 <= v <= MAXV for v in seq) and len(set(seq)) == N:
                inits.append(seq)
    
    # Evaluate all initializations
    for init_vals in inits:
        if time.time() > deadline - 30:
            break
        vals = list(init_vals)
        # Ensure distinct
        if len(set(vals)) < N:
            continue
        if any(v < 1 or v > MAXV for v in vals):
            continue
        edges, score = _optimize_edges(vals, deadline)
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
    
    # Local search on values
    if time.time() < deadline - 15:
        best_values, best_edges, best_score = _local_search_values(best_values, best_edges, deadline - 10)
    
    # Simulated annealing
    if time.time() < deadline - 10:
        best_values, best_edges, best_score = _simulated_annealing(best_values, best_edges, deadline - 5)
    
    # Polish
    if time.time() < deadline - 3:
        best_values, best_edges, best_score = _polish(best_values, best_edges, deadline)
    
    # Ensure types are correct
    values_out = [int(v) for v in best_values]
    edges_out = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': values_out, 'edges': edges_out}
# EVOLVE-BLOCK-END
