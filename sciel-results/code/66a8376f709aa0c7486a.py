"""Search-based program: minimize max(distinct_sums, distinct_products) over 35 edges of 14 vertices."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_score(values, edges):
    """Compute the raw score: max(#distinct sums, #distinct products) over edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges(n):
    """Generate all C(n,2) possible edges."""
    return list(itertools.combinations(range(n), 2))

def _greedy_edge_selection(values, target_m, all_edges):
    """
    Greedy: pick edges that maximize reuse of existing sum/product values.
    """
    edges_chosen = []
    chosen_set = set()
    sum_counts = {}
    prod_counts = {}
    
    edge_info = []
    for u, v in all_edges:
        a, b = values[u], values[v]
        s = a + b
        p = a * b
        edge_info.append((u, v, s, p))
    
    for _ in range(target_m):
        best_edge = None
        best_key = None
        n_sums = len(sum_counts)
        n_prods = len(prod_counts)
        if n_sums >= n_prods:
            w_s, w_p = 3, 1
        else:
            w_s, w_p = 1, 3
        for (u, v, s, p) in edge_info:
            if (u, v) in chosen_set:
                continue
            s_count = sum_counts.get(s, 0)
            p_count = prod_counts.get(p, 0)
            key = (w_s * s_count + w_p * p_count, s_count + p_count, -s, -p)
            if best_key is None or key > best_key:
                best_key = key
                best_edge = (u, v, s, p)
        
        if best_edge is None:
            break
        
        u, v, s, p = best_edge
        edges_chosen.append((u, v))
        chosen_set.add((u, v))
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    return [list(e[:2]) for e in edges_chosen]

def _make_powers_values():
    candidates = []
    for a in range(0, 6):
        for b in range(0, 6):
            val = (2 ** a) * (3 ** b)
            if val > 0 and val <= 10000 and val not in candidates:
                candidates.append(val)
    candidates.sort()
    values = candidates[:14]
    assert len(set(values)) == 14
    assert all(isinstance(v, int) and 0 < v <= 10000 for v in values)
    return values

def _make_high_collision_values():
    return [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]

def _make_geometric_values():
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

def _make_arithmetic_values():
    return [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

def _make_mixed_values():
    return [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]

def _make_prime_based_values():
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]

def _make_composite_values():
    return [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24]

def _make_rational_powers():
    """Values from small rational powers for sum/product structure."""
    base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    return base

def _make_squared_values():
    """Squared numbers: products are 4th powers, sums structured."""
    vals = [i*i for i in range(1, 15)]
    return vals

def _make_multiples_values():
    """Multiples of small numbers for product structure."""
    return [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

def _try_value_set(values, all_edges, deadline):
    if time.time() > deadline:
        return None
    edges = _greedy_edge_selection(values, P['m'], all_edges)
    if len(edges) != P['m']:
        return None
    score = _raw_score(values, edges)
    return score, edges

def _fast_local_search(values, edges, all_edges, deadline, max_iters=200):
    """
    Fast local search: precompute all swap deltas, pick best improving swap.
    """
    n = len(values)
    m = len(edges)
    
    edge_attrs = {}
    for u, v in all_edges:
        a, b = values[u], values[v]
        edge_attrs[(u, v)] = (a + b, a * b)
    
    best_score = float('inf')
    best_edges = None
    all_edge_tuples = [tuple(e) for e in all_edges]
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            current_edges = [list(e) for e in edges]
        else:
            current_edges = [list(e) for e in edges]
            k = min(2, m)
            for _ in range(k):
                i = random.randint(0, m - 1)
                j = random.randint(0, m - 1)
                if i != j:
                    current_edges[i], current_edges[j] = current_edges[j], current_edges[i]
        
        edge_set = set(tuple(e) for e in current_edges)
        
        sum_counts = {}
        prod_counts = {}
        for u, v in current_edges:
            s, p = edge_attrs[(u, v)]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        n_sums = len(sum_counts)
        n_prods = len(prod_counts)
        current_score = max(n_sums, n_prods)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(m):
                u1, v1 = current_edges[i]
                e1 = (u1, v1)
                s1, p1 = edge_attrs[e1]
                
                best_swap = None
                best_delta = 0
                best_new_score = current_score
                
                for e2 in all_edge_tuples:
                    if e2 == e1 or e2 in edge_set:
                        continue
                    
                    s2, p2 = edge_attrs[e2]
                    
                    old_s_count = sum_counts[s1]
                    old_p_count = prod_counts[p1]
                    new_s_count = sum_counts.get(s2, 0)
                    new_p_count = prod_counts.get(p2, 0)
                    
                    # Compute new distinct counts after swap
                    ns_sums = n_sums
                    ns_prods = n_prods
                    
                    if old_s_count == 1:
                        ns_sums -= 1
                    if new_s_count == 0:
                        ns_sums += 1
                    if old_p_count == 1:
                        ns_prods -= 1
                    if new_p_count == 0:
                        ns_prods += 1
                    
                    new_score = max(ns_sums, ns_prods)
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_swap = e2
                        best_delta = new_score - current_score
                
                if best_swap is not None:
                    s2, p2 = edge_attrs[best_swap]
                    old_s_count = sum_counts[s1]
                    old_p_count = prod_counts[p1]
                    new_s_count = sum_counts.get(s2, 0)
                    new_p_count = prod_counts.get(p2, 0)
                    
                    sum_counts[s1] = old_s_count - 1
                    prod_counts[p1] = old_p_count - 1
                    if sum_counts[s1] == 0:
                        del sum_counts[s1]
                    if prod_counts[p1] == 0:
                        del prod_counts[p1]
                    
                    sum_counts[s2] = new_s_count + 1
                    prod_counts[p2] = new_p_count + 1
                    
                    n_sums = len(sum_counts)
                    n_prods = len(prod_counts)
                    current_score = max(n_sums, n_prods)
                    
                    current_edges[i] = list(best_swap)
                    edge_set.remove(e1)
                    edge_set.add(best_swap)
                    improved = True
                    break
            
            if not improved:
                break
        
        if current_score < best_score:
            best_score = current_score
            best_edges = [list(e) for e in current_edges]
    
    return best_score, best_edges

def _value_perturbation(values, all_edges, deadline, n_perturb=5):
    """Try small perturbations to values and re-optimize edges."""
    best_score = float('inf')
    best_edges = None
    best_values = None
    n = len(values)
    
    for _ in range(n_perturb):
        if time.time() > deadline:
            break
        
        new_values = values[:]
        idx = random.randint(0, n - 1)
        delta = random.choice([-1, 1, -2, 2, -3, 3, -5, 5])
        new_val = new_values[idx] + delta
        if new_val <= 0 or new_val > 10000 or new_val in new_values:
            continue
        new_values[idx] = new_val
        
        edges = _greedy_edge_selection(new_values, P['m'], all_edges)
        if len(edges) != P['m']:
            continue
        score = _raw_score(new_values, edges)
        if score < best_score:
            best_score = score
            best_edges = edges
            best_values = new_values[:]
    
    if best_edges is not None:
        return best_score, best_edges, best_values
    return None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline, 5s safety margin from 90s
    
    all_edges = _all_edges(P['n'])
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    candidates = []
    
    value_strategies = [
        _make_powers_values,
        _make_high_collision_values,
        _make_geometric_values,
        _make_arithmetic_values,
        _make_mixed_values,
        _make_prime_based_values,
        _make_composite_values,
        _make_rational_powers,
        _make_squared_values,
        _make_multiples_values,
    ]
    
    for make_values in value_strategies:
        if time.time() > deadline - 30:
            break
        
        values = make_values()
        if len(set(values)) != 14:
            continue
        if not all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            continue
        
        result = _try_value_set(values, all_edges, deadline)
        if result is not None:
            score, edges = result
            candidates.append((score, values[:], edges))
    
    candidates.sort(key=lambda x: x[0])
    
    for score, values, edges in candidates[:5]:
        if time.time() > deadline - 5:
            break
        
        ls_score, ls_edges = _fast_local_search(values, edges, all_edges, deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
            best_values = values[:]
        
        # Also try value perturbation on this value set
        if time.time() > deadline - 3:
            continue
        vp_score, vp_edges, vp_values = _value_perturbation(values, all_edges, deadline, n_perturb=3)
        if vp_score is not None and vp_score < best_score:
            best_score = vp_score
            best_edges = vp_edges
            best_values = vp_values[:]
    
    if best_edges is None:
        values1 = _make_powers_values()
        edges = _greedy_edge_selection(values1, P['m'], all_edges)
        best_values = values1
        best_edges = edges
    
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
