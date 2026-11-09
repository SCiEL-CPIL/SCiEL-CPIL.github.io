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
    
    # Precompute sums and products for all edges
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
        
        # To speed up, we might not check all remaining edges if num is large, 
        # but 91 edges is small, so checking all is fine.
        for e in remaining_edges:
            s, p = edge_props[e]
            new_sums_size = len(selected_sums | {s})
            new_products_size = len(selected_products | {p})
            current_score = max(new_sums_size, new_products_size)
            
            # Tie-breaking: prefer edges that add fewer total new elements
            # or just use the score.
            if current_score < best_score:
                best_score = current_score
                best_edge = e
                best_sums = selected_sums | {s}
                best_products = selected_products | {p}
            elif current_score == best_score:
                # Tie-breaker: minimize total distinct count
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

def _local_search_edges(values, edges, max_iters=500):
    """Refine edge selection using local search."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    current_edges = [tuple(e) for e in edges]
    current_set = set(current_edges)
    current_score = _score(values, current_edges)
    
    best_edges = list(current_edges)
    best_score = current_score
    
    # Precompute properties for faster evaluation
    edge_props = {}
    for i, j in all_edges:
        edge_props[(i, j)] = (values[i] + values[j], values[i] * values[j])
        
    for _ in range(max_iters):
        if not current_edges:
            break
        # Pick a random edge to remove
        idx = random.randint(0, len(current_edges) - 1)
        e_out = current_edges[idx]
        
        # Pick a random edge to add (not in current set)
        available = [e for e in all_edges if e not in current_set]
        if not available:
            break
        e_in = random.choice(available)
        
        # Evaluate new score
        new_edges = current_edges[:]
        new_edges[idx] = e_in
        new_set = set(new_edges)
        
        # Quick check: if e_in and e_out have same sum/product, score might not change or decrease
        # Just compute full score for simplicity, 35 edges is small
        new_score = _score(values, new_edges)
        
        if new_score < current_score:
            current_edges = new_edges
            current_set = new_set
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = list(current_edges)
        elif new_score == current_score and random.random() < 0.1:
            # Accept neutral moves to escape local optima
            current_edges = new_edges
            current_set = new_set
            # current_score stays same
            
    return best_edges, best_score

def construct():
    """Construct values and edges minimizing max(distinct sums, distinct products)."""
    random.seed(42)
    
    best_overall_score = float('inf')
    best_overall_result = None
    
    # Define candidate value sets
    # 1. Consecutive integers
    v1 = list(range(1, 15))
    # 2. Even integers
    v2 = list(range(2, 29, 2))
    # 3. Multiples of 3
    v3 = list(range(3, 43, 3))
    # 4. Small integers with high factorization (heuristic)
    v4 = [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]
    # 5. Powers of 2 (likely bad, but check)
    v5 = [2**i for i in range(1, 15)]
    # 6. Random small numbers
    v6 = random.sample(range(1, 50), 14)
    # 7. Random small numbers
    v7 = random.sample(range(1, 100), 14)
    
    candidates = [v1, v2, v3, v4, v5, v6, v7]
    
    for values in candidates:
        # Ensure distinct and positive
        if len(set(values)) != 14 or any(v <= 0 for v in values):
            continue
            
        # Greedy initialization
        edges, score = _greedy_edge_selection(values, 35)
        
        # Local search refinement
        edges, score = _local_search_edges(values, edges, max_iters=1000)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_result = (list(values), [list(e) for e in edges])
            
        # Early stopping if we hit a very low score
        if best_overall_score <= 30:
            break
            
    if best_overall_result is None:
        # Fallback
        values = list(range(1, 15))
        edges, _ = _greedy_edge_selection(values, 35)
        best_overall_result = (values, [list(e) for e in edges])
        
    return {'values': best_overall_result[0], 'edges': best_overall_result[1]}
# EVOLVE-BLOCK-END
