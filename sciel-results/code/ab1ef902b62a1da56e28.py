"""Rewritten program: structural change to simultaneous value+edge optimization via SA."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import Counter

def _score(values, edges):
    """Compute max(distinct_sums, distinct_products) for given values and edges."""
    if not edges:
        return 0
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _fast_score_delta(values, edge, sums_set, prods_set):
    """Check if adding edge increases the score. Returns (new_score, sum_added, prod_added)."""
    i, j = edge
    s = values[i] + values[j]
    p = values[i] * values[j]
    new_sums = len(sums_set) + (0 if s in sums_set else 1)
    new_prods = len(prods_set) + (0 if p in prods_set else 1)
    return max(new_sums, new_prods), s, p

def _build_sum_prod_tables(values):
    """Precompute sum and product for all pairs."""
    n = len(values)
    sum_table = {}
    prod_table = {}
    for i in range(n):
        for j in range(i+1, n):
            sum_table[(i, j)] = values[i] + values[j]
            prod_table[(i, j)] = values[i] * values[j]
    return sum_table, prod_table

def _score_from_tables(sum_table, prod_table, edges):
    """Compute score using precomputed tables."""
    sums = set()
    prods = set()
    for i, j in edges:
        key = (min(i,j), max(i,j))
        sums.add(sum_table[key])
        prods.add(prod_table[key])
    return max(len(sums), len(prods))

def _greedy_edge_selection(values, num_edges=35, seed=42):
    """Greedy: select edges to minimize distinct sums and products."""
    random.seed(seed)
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    sum_table, prod_table = _build_sum_prod_tables(values)
    
    selected = []
    sums = set()
    prods = set()
    current_score = 0
    
    # First pass: pick edges that don't increase score
    for e in all_edges:
        if len(selected) >= num_edges:
            break
        i, j = e
        key = (min(i,j), max(i,j))
        s = sum_table[key]
        p = prod_table[key]
        if s in sums and p in prods:
            selected.append((i, j))
            sums.add(s)
            prods.add(p)
    
    # Second pass: pick edges that increase score least
    if len(selected) < num_edges:
        remaining = [e for e in all_edges if e not in selected]
        random.shuffle(remaining)
        for e in remaining:
            if len(selected) >= num_edges:
                break
            i, j = e
            key = (min(i,j), max(i,j))
            s = sum_table[key]
            p = prod_table[key]
            new_sums = len(sums) + (0 if s in sums else 1)
            new_prods = len(prods) + (0 if p in prods else 1)
            # Prefer edges that add to the smaller set
            if new_sums <= new_prods:
                selected.append((i, j))
                sums.add(s)
                prods.add(p)
            else:
                selected.append((i, j))
                sums.add(s)
                prods.add(p)
    
    return selected[:num_edges]

def _local_search_edges(values, edges, max_iter=3000, deadline=None):
    """Local search: swap edges to improve score."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    sum_table, prod_table = _build_sum_prod_tables(values)
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    current_score = _score_from_tables(sum_table, prod_table, list(edge_set))
    
    in_list = list(edge_set)
    out_list = [e for e in all_edges if e not in edge_set]
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        if not out_list:
            break
        
        old = random.choice(in_list)
        new = random.choice(out_list)
        
        edge_set.discard(old)
        edge_set.add(new)
        new_score = _score_from_tables(sum_table, prod_table, list(edge_set))
        
        if new_score <= current_score:
            current_score = new_score
            in_list.remove(old)
            in_list.append(new)
            out_list.remove(new)
            out_list.append(old)
        else:
            edge_set.discard(new)
            edge_set.add(old)
    
    return [list(e) for e in edge_set], current_score

def _local_search_values(values, edges, max_iter=2000, deadline=None):
    """Local search: perturb values to improve score."""
    n = len(values)
    current_score = _score(values, edges)
    best_values = list(values)
    best_score = current_score
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Randomly perturb one value
        new_values = list(best_values)
        idx = random.randrange(n)
        old_val = new_values[idx]
        
        # Try small perturbations
        delta = random.choice([-3, -2, -1, 1, 2, 3, 5, 7, 11, -5, -7])
        new_val = old_val + delta
        if new_val < 1 or new_val > 10000:
            continue
        if new_val in new_values:
            continue
        
        new_values[idx] = new_val
        new_score = _score(new_values, edges)
        
        if new_score <= best_score:
            best_values = new_values
            best_score = new_score
    
    return best_values, best_score

def _simulated_annealing(values, edges, deadline):
    """SA over both values and edges."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    current_values = list(values)
    current_edges = set(tuple(sorted(e)) for e in edges)
    current_score = _score(current_values, list(current_edges))
    
    best_values = list(current_values)
    best_edges = list(current_edges)
    best_score = current_score
    
    start_temp = 10.0
    end_temp = 0.1
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        temp = start_temp * (end_temp / start_temp) ** (min(steps, 5000) / 5000.0)
        
        # Choose move type
        if random.random() < 0.5:
            # Edge swap
            if len(current_edges) < 2:
                continue
            in_list = list(current_edges)
            out_list = [e for e in all_edges if e not in current_edges]
            if not out_list:
                continue
            old = random.choice(in_list)
            new = random.choice(out_list)
            
            current_edges.discard(old)
            current_edges.add(new)
            new_score = _score(current_values, list(current_edges))
            
            delta = new_score - current_score
            if delta <= 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(current_values)
                    best_edges = list(current_edges)
            else:
                current_edges.discard(new)
                current_edges.add(old)
        else:
            # Value perturbation
            idx = random.randrange(n)
            old_val = current_values[idx]
            delta_val = random.choice([-3, -2, -1, 1, 2, 3, 5, 7, 11, -5, -7, 13, -13])
            new_val = old_val + delta_val
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current_values:
                continue
            
            current_values[idx] = new_val
            new_score = _score(current_values, list(current_edges))
            
            delta = new_score - current_score
            if delta <= 0 or random.random() < math.exp(-delta / max(temp, 0.01)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(current_values)
                    best_edges = list(current_edges)
            else:
                current_values[idx] = old_val
    
    return best_values, best_edges, best_score

def _make_candidate_values_1():
    """Small consecutive integers."""
    return list(range(1, 15))

def _make_candidate_values_2():
    """Powers and multiples for product structure."""
    return [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]

def _make_candidate_values_3():
    """Multiplicative structure."""
    return [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]

def _make_candidate_values_4():
    """Mixed structure."""
    return [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]

def _make_candidate_values_5():
    """Sparse with gaps."""
    return [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]

def _make_candidate_values_6():
    """Highly composite based."""
    return [1, 2, 3, 4, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108]

def construct():
    start = time.time()
    deadline = start + 75  # Safety margin before 90s kill
    
    best = None
    best_score = float('inf')
    
    candidates = [
        _make_candidate_values_1(),
        _make_candidate_values_2(),
        _make_candidate_values_3(),
        _make_candidate_values_4(),
        _make_candidate_values_5(),
        _make_candidate_values_6(),
    ]
    
    for ci, vals in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Try multiple seeds for greedy
        for seed in [1, 42, 99, 123]:
            if time.time() > deadline:
                break
            random.seed(seed)
            edges = _greedy_edge_selection(vals, 35, seed)
            
            # Local search on edges
            edges, s = _local_search_edges(vals, edges, max_iter=2000, deadline=deadline)
            
            if s < best_score:
                best_score = s
                best = (list(vals), [list(e) for e in edges])
        
        # SA on both values and edges (use remaining time)
        if time.time() < deadline - 10:
            sa_deadline = min(deadline - 2, time.time() + 15)
            vals_sa, edges_sa, s_sa = _simulated_annealing(vals, best[1] if best else [], sa_deadline)
            if s_sa < best_score:
                best_score = s_sa
                best = (vals_sa, [list(e) for e in edges_sa])
        
        # Local search on values
        if best is not None and time.time() < deadline - 5:
            vals_l, s_l = _local_search_values(best[0], best[1], max_iter=1500, deadline=deadline)
            if s_l < best_score:
                best_score = s_l
                best = (vals_l, list(best[1]))
    
    if best is None:
        vals = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        return {'values': vals, 'edges': edges}
    
    # Final polish
    if time.time() < deadline - 3:
        vals_f, s_f = _local_search_values(best[0], best[1], max_iter=1000, deadline=deadline)
        if s_f <= best_score:
            best = (vals_f, list(best[1]))
    
    return {'values': [int(v) for v in best[0]], 'edges': [[int(i), int(j)] for i, j in best[1]]}
# EVOLVE-BLOCK-END
