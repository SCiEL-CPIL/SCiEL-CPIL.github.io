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
    """Greedy + local search on edges given fixed values."""
    all_edges = list(combinations(range(N), 2))
    best_edges = None
    best_score = None

    # Greedy construction
    used = set()
    edges = []
    sums = set()
    prods = set()
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
        idx = random.randrange(N)
        old = best[idx]
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
    
    T = 50.0
    cooling = 0.995
    iters = 0
    max_iters = 2000
    
    while iters < max_iters:
        iters += 1
        if time.time() > deadline:
            break
        T *= cooling
        if T < 0.1:
            break
        
        # Perturb: change one or two values
        if random.random() < 0.7:
            idx = random.randrange(N)
            old = current[idx]
            new_val = random.randint(1, MAXV)
            if new_val == old or new_val in current:
                continue
            cand = list(current)
            cand[idx] = new_val
        else:
            idx1 = random.randrange(N)
            idx2 = random.randrange(N)
            if idx1 == idx2:
                continue
            old1, old2 = current[idx1], current[idx2]
            new1 = random.randint(1, MAXV)
            new2 = random.randint(1, MAXV)
            if new1 == old1 or new2 == old2 or new1 in current or new2 in current:
                continue
            cand = list(current)
            cand[idx1] = new1
            cand[idx2] = new2
        
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
    
    iters = 0
    max_iters = 300
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
                break
        
        # Try edge swaps
        if random.random() < 0.5:
            edges_list = list(best_edges)
            random.shuffle(edges_list)
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
                all_edges = list(combinations(range(N), 2))
                # Convert to tuples for hashing
                edges_set = set(tuple(e) for e in edges_list)
                for f in all_edges:
                    if f in edges_set:
                        continue
                    s = best[f[0]] + best[f[1]]
                    p = best[f[0]] * best[f[1]]
                    new_s = 0 if s in base_sums else 1
                    new_p = 0 if p in base_prods else 1
                    key = (new_s + new_p, new_p, new_s, random.random())
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
    for _ in range(10):
        if len(pool) >= 14:
            inits.append(random.sample(pool, 14))
    
    # Init 6: Structured pairs
    for C in [100, 200, 500, 1000, 2000, 5000, 10000]:
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
    
    # Evaluate all initializations
    for init_vals in inits:
        if time.time() > deadline - 10:
            break
        vals = list(init_vals)
        # Ensure distinct
        if len(set(vals)) < N:
            continue
        edges, score = _optimize_edges(vals, deadline)
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
    
    # Simulated annealing
    if time.time() < deadline - 20:
        best_values, best_edges, best_score = _simulated_annealing(best_values, best_edges, deadline - 10)
    
    # Polish
    if time.time() < deadline - 5:
        best_values, best_edges, best_score = _polish(best_values, best_edges, deadline)
    
    # Ensure types are correct
    values_out = [int(v) for v in best_values]
    edges_out = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': values_out, 'edges': edges_out}
# EVOLVE-BLOCK-END
