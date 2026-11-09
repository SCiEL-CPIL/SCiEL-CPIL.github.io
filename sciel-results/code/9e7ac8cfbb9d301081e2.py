"""Rewritten program for small sums and products along selected graph edges."""
import time
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85  # 5s safety margin before 90s kill


def _score(values, edges):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))


def _greedy_edge_selection(values, m, n):
    """
    Greedy edge selection: iteratively pick the edge that minimizes the
    resulting max(distinct_sums, distinct_products).
    """
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    # Precompute sum and product for each edge
    edge_sum = {}
    edge_prod = {}
    for e in all_edges:
        u, v = e
        edge_sum[e] = values[u] + values[v]
        edge_prod[e] = values[u] * values[v]
    
    selected = []
    sum_set = set()
    prod_set = set()
    remaining = list(all_edges)
    
    for _ in range(m):
        if time.time() > _DEADLINE:
            break
        best_edge = None
        best_score = float('inf')
        
        for e in remaining:
            s = edge_sum[e]
            p = edge_prod[e]
            new_sums = len(sum_set | {s})
            new_prods = len(prod_set | {p})
            score = max(new_sums, new_prods)
            # Tie-break: prefer edges with smaller sum (more collision potential)
            tie_break = s
            if (score, tie_break) < (best_score, float('inf')) or best_edge is None:
                if best_edge is None or score < best_score or (score == best_score and tie_break < best_score):
                    best_score = score
                    best_edge = e
        
        if best_edge is None:
            break
        
        selected.append(best_edge)
        sum_set.add(edge_sum[best_edge])
        prod_set.add(edge_prod[best_edge])
        remaining.remove(best_edge)
    
    return selected


def _hill_climb_edges(values, edges, m, n, max_iter=50000):
    """
    Hill climbing on edge set: try swapping one edge for another to reduce score.
    """
    all_edges = list(combinations(range(n), 2))
    edge_set = set(edges)
    
    # Precompute
    edge_sum = {}
    edge_prod = {}
    for e in all_edges:
        u, v = e
        edge_sum[e] = values[u] + values[v]
        edge_prod[e] = values[u] * values[v]
    
    # Current state with counts
    sum_counts = {}
    prod_counts = {}
    for e in edge_set:
        s = edge_sum[e]
        p = edge_prod[e]
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    def _get_score():
        return max(len(sum_counts), len(prod_counts))
    
    current_score = _get_score()
    best_score = current_score
    best_edge_set = set(edge_set)
    best_sum_counts = dict(sum_counts)
    best_prod_counts = dict(prod_counts)
    
    # Build list of non-selected edges
    non_selected = [e for e in all_edges if e not in edge_set]
    
    # Precompute for each non-selected edge, which selected edges it could replace
    # to potentially reduce score
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if current_score <= 3:
            break
        
        # Pick a random edge to remove
        remove_edge = random.choice(list(edge_set))
        
        # Try several candidate edges to add
        tried = 0
        max_tries = 8
        found_improvement = False
        while tried < max_tries:
            if time.time() > _DEADLINE:
                break
            add_edge = random.choice(non_selected)
            tried += 1
            
            # Compute new score if we swap
            s_rem = edge_sum[remove_edge]
            p_rem = edge_prod[remove_edge]
            s_add = edge_sum[add_edge]
            p_add = edge_prod[add_edge]
            
            # Tentatively remove remove_edge
            sum_counts[s_rem] -= 1
            if sum_counts[s_rem] == 0:
                del sum_counts[s_rem]
            prod_counts[p_rem] -= 1
            if prod_counts[p_rem] == 0:
                del prod_counts[p_rem]
            
            # Tentatively add add_edge
            sum_counts[s_add] = sum_counts.get(s_add, 0) + 1
            prod_counts[p_add] = prod_counts.get(p_add, 0) + 1
            
            new_score = max(len(sum_counts), len(prod_counts))
            
            if new_score < current_score:
                # Accept the swap
                edge_set.discard(remove_edge)
                edge_set.add(add_edge)
                non_selected.remove(add_edge)
                non_selected.append(remove_edge)
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edge_set = set(edge_set)
                    best_sum_counts = dict(sum_counts)
                    best_prod_counts = dict(prod_counts)
                found_improvement = True
                break
            else:
                # Revert
                sum_counts[s_add] -= 1
                if sum_counts[s_add] == 0:
                    del sum_counts[s_add]
                prod_counts[p_add] -= 1
                if prod_counts[p_add] == 0:
                    del prod_counts[p_add]
                sum_counts[s_rem] = sum_counts.get(s_rem, 0) + 1
                prod_counts[p_rem] = prod_counts.get(p_rem, 0) + 1
        
        if not found_improvement:
            # No improvement found in this iteration; try a different strategy
            # Sometimes accept equal-score moves to escape local optima
            if random.random() < 0.1:
                remove_edge = random.choice(list(edge_set))
                add_edge = random.choice(non_selected)
                s_rem = edge_sum[remove_edge]
                p_rem = edge_prod[remove_edge]
                s_add = edge_sum[add_edge]
                p_add = edge_prod[add_edge]
                
                sum_counts[s_rem] -= 1
                if sum_counts[s_rem] == 0:
                    del sum_counts[s_rem]
                prod_counts[p_rem] -= 1
                if prod_counts[p_rem] == 0:
                    del prod_counts[p_rem]
                
                sum_counts[s_add] = sum_counts.get(s_add, 0) + 1
                prod_counts[p_add] = prod_counts.get(p_add, 0) + 1
                
                new_score = max(len(sum_counts), len(prod_counts))
                
                if new_score <= current_score:
                    edge_set.discard(remove_edge)
                    edge_set.add(add_edge)
                    non_selected.remove(add_edge)
                    non_selected.append(remove_edge)
                    current_score = new_score
                else:
                    # Revert
                    sum_counts[s_add] -= 1
                    if sum_counts[s_add] == 0:
                        del sum_counts[s_add]
                    prod_counts[p_add] -= 1
                    if prod_counts[p_add] == 0:
                        del prod_counts[p_add]
                    sum_counts[s_rem] = sum_counts.get(s_rem, 0) + 1
                    prod_counts[p_rem] = prod_counts.get(p_rem, 0) + 1
    
    return best_score, list(best_edge_set)


def _try_value_set(values, m, n):
    """Try a value set: greedy init + hill climb."""
    if time.time() > _DEADLINE:
        return None
    
    # Greedy initialization
    edges = _greedy_edge_selection(values, m, n)
    if len(edges) < m:
        # Pad with random edges
        all_edges = list(combinations(range(n), 2))
        edge_set = set(edges)
        for e in all_edges:
            if len(edge_set) >= m:
                break
            if e not in edge_set:
                edge_set.add(e)
        edges = list(edge_set)
    
    # Hill climb
    score, final_edges = _hill_climb_edges(values, edges, m, n, max_iter=30000)
    
    return score, values, final_edges


def _generate_value_candidates(n, max_val):
    """Generate diverse candidate value sets."""
    candidates = []
    
    # 1. Consecutive small integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Small even numbers
    candidates.append(list(range(2, 2 * n + 1, 2)))
    
    # 3. Highly composite-like: divisors of 2520 (highly composite)
    # 2520 = 2^3 * 3^2 * 5 * 7
    divisors_2520 = []
    for d in range(1, 2521):
        if 2520 % d == 0:
            divisors_2520.append(d)
    divisors_2520.sort()
    if len(divisors_2520) >= n:
        # Pick n divisors that are well-distributed
        # Prefer smaller ones for sum collisions
        candidates.append(divisors_2520[:n])
        # Also try a mix
        if len(divisors_2520) >= 2 * n:
            candidates.append(divisors_2520[:n//2] + divisors_2520[-n//2:])
    
    # 4. Divisors of 5040 (even more composite)
    divisors_5040 = []
    for d in range(1, 5041):
        if 5040 % d == 0:
            divisors_5040.append(d)
    divisors_5040.sort()
    if len(divisors_5040) >= n:
        candidates.append(divisors_5040[:n])
    
    # 5. Divisors of 7560
    divisors_7560 = []
    for d in range(1, 7561):
        if 7560 % d == 0:
            divisors_7560.append(d)
    divisors_7560.sort()
    if len(divisors_7560) >= n:
        candidates.append(divisors_7560[:n])
    
    # 6. Small primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    candidates.append(primes[:n])
    
    # 7. Arithmetic progressions
    for start in range(1, 8):
        for step in range(1, 5):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val and min(vals) >= 1:
                candidates.append(vals)
    
    # 8. Random small values
    random.seed(42)
    for _ in range(30):
        vals = sorted(random.sample(range(1, 40), n))
        candidates.append(vals)
    
    # 9. Clustered values
    for base in range(1, 15):
        vals = []
        for i in range(n):
            v = base + random.randint(0, 4)
            vals.append(v)
        vals = sorted(set(vals))
        while len(vals) < n:
            vals.append(base + random.randint(0, 8))
        vals = sorted(vals[:n])
        if len(set(vals)) == n and max(vals) <= max_val:
            candidates.append(vals)
    
    # 10. Multiples of small numbers
    for d in [2, 3, 4, 5, 6]:
        vals = [d * (i + 1) for i in range(n)]
        if max(vals) <= max_val and len(set(vals)) == n:
            candidates.append(vals)
    
    # 11. Mix of very small and some larger
    random.seed(123)
    for _ in range(15):
        small = sorted(random.sample(range(1, 10), 7))
        large = sorted(random.sample(range(10, 30), 7))
        vals = sorted(small + large)
        if len(set(vals)) == n and max(vals) <= max_val:
            candidates.append(vals)
    
    # 12. Values around a central highly composite number's divisors
    # Pick divisors of 840 (2^3*3*5*7)
    divisors_840 = []
    for d in range(1, 841):
        if 840 % d == 0:
            divisors_840.append(d)
    divisors_840.sort()
    if len(divisors_840) >= n:
        candidates.append(divisors_840[:n])
    
    # 13. Powers of 2 and 3 mixed
    vals = []
    for i in range(1, 11):
        vals.append(2 ** i)
    for i in range(1, 6):
        vals.append(3 ** i)
    vals = sorted(set(vals))
    if len(vals) >= n:
        candidates.append(vals[:n])
    
    # 14. Values that are all divisors of a common number
    for base in [120, 240, 360, 420, 600, 840, 1260, 1680]:
        divs = [d for d in range(1, base + 1) if base % d == 0]
        divs.sort()
        if len(divs) >= n:
            candidates.append(divs[:n])
    
    return candidates


def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m  # worst case
    best_result = None
    
    candidates = _generate_value_candidates(n, max_val)
    
    # Shuffle candidates for diversity
    random.seed(7)
    random.shuffle(candidates)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates
    
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
        
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        
        result = _try_value_set(vals, m, n)
        if result is None:
            continue
        score, values, edges = result
        if score < best_score:
            best_score = score
            best_result = (values, edges)
            if best_score <= 3:
                break
    
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
