"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products))

def _greedy_edge_selection(values, num_edges=35):
    """Select edges greedily to minimize distinct sums and products."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    edge_props = {}
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_props[(i, j)] = (s, p)
        
    selected_edges = []
    selected_sums = set()
    selected_products = set()
    
    remaining_edges = set(all_edges)
    
    for _ in range(num_edges):
        best_edge = None
        best_score = float('inf')
        best_sums = None
        best_products = None
        
        for e in remaining_edges:
            s, p = edge_props[e]
            new_sums_size = len(selected_sums | {s})
            new_products_size = len(selected_products | {p})
            current_score = max(new_sums_size, new_products_size)
            
            if current_score < best_score:
                best_score = current_score
                best_edge = e
                best_sums = selected_sums | {s}
                best_products = selected_products | {p}
            elif current_score == best_score:
                total_new = new_sums_size + new_products_size
                total_best = len(best_sums) + len(best_products)
                if total_new < total_best:
                    best_score = current_score
                    best_edge = e
                    best_sums = selected_sums | {s}
                    best_products = selected_products | {p}
                    
        if best_edge is None:
            break
            
        selected_edges.append(best_edge)
        selected_sums = best_sums
        selected_products = best_products
        remaining_edges.remove(best_edge)
        
    return selected_edges, _score(values, selected_edges)

def _local_search_edges(values, edges, max_iters=2000):
    """Refine edge selection using local search."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    current_edges = [tuple(e) for e in edges]
    current_set = set(current_edges)
    current_score = _score(values, current_edges)
    
    best_edges = list(current_edges)
    best_score = current_score
    
    edge_props = {}
    for i, j in all_edges:
        edge_props[(i, j)] = (values[i] + values[j], values[i] * values[j])
        
    for _ in range(max_iters):
        if not current_edges:
            break
        idx = random.randint(0, len(current_edges) - 1)
        e_out = current_edges[idx]
        
        available = [e for e in all_edges if e not in current_set]
        if not available:
            break
        e_in = random.choice(available)
        
        new_edges = current_edges[:]
        new_edges[idx] = e_in
        new_set = set(new_edges)
        
        new_score = _score(values, new_edges)
        
        if new_score < current_score:
            current_edges = new_edges
            current_set = new_set
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = list(current_edges)
        elif new_score == current_score and random.random() < 0.15:
            current_edges = new_edges
            current_set = new_set
            
    return best_edges, best_score

def _multi_restart_search(values, num_restarts=4, iters_per_restart=1000):
    """Run multiple restarts of greedy + local search."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    best_edges = None
    best_score = float('inf')
    
    for restart in range(num_restarts):
        if restart == 0:
            edges, score = _greedy_edge_selection(values, 35)
        else:
            edges = [list(e) for e in random.sample(all_edges, 35)]
            score = _score(values, edges)
        
        edges, score = _local_search_edges(values, edges, max_iters=iters_per_restart)
        
        if score < best_score:
            best_score = score
            best_edges = [list(e) for e in edges]
            
        if best_score <= 25:
            break
            
    return best_edges, best_score

def construct():
    """Construct values and edges minimizing max(distinct sums, distinct products)."""
    random.seed(42)
    
    best_overall_score = float('inf')
    best_overall_result = None
    
    candidates = []
    candidates.append(list(range(1, 15)))
    candidates.append(list(range(2, 29, 2)))
    candidates.append(list(range(3, 43, 3)))
    candidates.append([1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24])
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 20, 24, 30, 36])
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    candidates.append([1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 7, 14])
    candidates.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    candidates.append([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42])
    
    for _ in range(8):
        vals = random.sample(range(1, 80), 14)
        vals.sort()
        candidates.append(vals)
    
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    for values in unique_candidates:
        if len(set(values)) != 14 or any(v <= 0 or v > 10000 for v in values):
            continue
            
        edges, score = _multi_restart_search(values, num_restarts=3, iters_per_restart=600)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_result = (list(values), edges)
            
        if best_overall_score <= 28:
            break
            
    if best_overall_result is None:
        values = list(range(1, 15))
        edges, _ = _greedy_edge_selection(values, 35)
        best_overall_result = (values, edges)
        
    return {'values': best_overall_result[0], 'edges': best_overall_result[1]}
# EVOLVE-BLOCK-END
