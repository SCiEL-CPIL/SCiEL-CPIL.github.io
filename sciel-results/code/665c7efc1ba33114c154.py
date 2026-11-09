"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import itertools
import random

def construct():
    n = 14
    m = 35
    
    # Generate diverse candidate value sets
    # Strategy: Use small numbers with many common factors to maximize product collisions
    # and small ranges to minimize sum collisions.
    
    candidate_sets = []
    
    # Helper to add unique sets
    def add_set(vals):
        if len(set(vals)) == 14 and all(1 <= v <= 10000 for v in vals):
            key = tuple(sorted(vals))
            if key not in seen_sets:
                seen_sets.add(key)
                candidate_sets.append(sorted(vals))
                
    seen_sets = set()
    
    # Set 1: Small integers 1..14
    add_set(list(range(1, 15)))
    
    # Set 2: Composites rich in factors (1-20 range, skipping primes where possible)
    add_set([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # Set 3: Divisors of 2520 (highly composite)
    divs_2520 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 56, 60, 63, 70, 72, 84, 90, 105, 120, 126, 140, 168, 180, 210, 252, 280, 315, 360, 420, 504, 630, 840, 1260, 2520]
    add_set(divs_2520[:14])
    
    # Set 4: Powers of 2 and 3 and 5 mixed
    add_set([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24])
    
    # Set 5: Randomized small composites
    base_composites = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30]
    for _ in range(10):
        random.shuffle(base_composites)
        add_set(base_composites[:14])
        
    # Set 6: Even numbers only (maximizes shared factor 2)
    add_set([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    
    # Set 7: Multiples of 3
    add_set([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42])

    def run_greedy(values, deterministic=True):
        all_edges = list(itertools.combinations(range(n), 2))
        
        distinct_sums = set()
        distinct_products = set()
        selected_edges = []
        
        # Precompute sum and product for each edge
        edge_info = {}
        for i, j in all_edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_info[(i, j)] = (s, p)
            
        # Precompute frequency of sums and products in the FULL set of edges
        sum_counts = {}
        prod_counts = {}
        for i, j in all_edges:
            s, p = edge_info[(i, j)]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
            
        remaining = set(all_edges)
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            best_tie_breaker = float('inf')
            
            for edge in remaining:
                s, p = edge_info[edge]
                new_sums = 0 if s in distinct_sums else 1
                new_prods = 0 if p in distinct_products else 1
                
                # Primary metric: max of distinct counts after addition
                new_max = max(len(distinct_sums) + new_sums, len(distinct_products) + new_prods)
                
                # Secondary metric: Prefer edges that don't add new distinct values.
                # Tertiary metric: If adding new, prefer those that are "common" (high count in full graph)
                tie_breaker = 0
                if new_sums:
                    tie_breaker -= sum_counts.get(s, 0)
                if new_prods:
                    tie_breaker -= prod_counts.get(p, 0)
                
                # If deterministic, use a stable tie-breaker (e.g., edge index)
                # Otherwise, add noise
                if deterministic:
                    total_score = (new_max, tie_breaker, edge)
                else:
                    noise = random.random() * 0.01
                    total_score = (new_max, tie_breaker + noise)
                
                if total_score < (best_score, best_tie_breaker):
                    best_score = new_max
                    best_tie_breaker = tie_breaker
                    best_edge = edge
            
            if best_edge is not None:
                selected_edges.append([best_edge[0], best_edge[1]])
                s, p = edge_info[best_edge]
                distinct_sums.add(s)
                distinct_products.add(p)
                remaining.discard(best_edge)
        
        # Calculate final score
        raw_value = max(len(distinct_sums), len(distinct_products))
        return raw_value, {'values': values, 'edges': selected_edges}

    best_raw_value = float('inf')
    best_result = None
    
    # Try deterministic run for each value set
    for vals in candidate_sets:
        raw_val, result = run_greedy(vals, deterministic=True)
        if raw_val < best_raw_value:
            best_raw_value = raw_val
            best_result = result
            
    # If time permits, try a few random restarts on the best value set
    if best_result is not None:
        best_vals = best_result['values']
        for _ in range(5):
            raw_val, result = run_greedy(best_vals, deterministic=False)
            if raw_val < best_raw_value:
                best_raw_value = raw_val
                best_result = result

    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result

# EVOLVE-BLOCK-END
