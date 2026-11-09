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
    
    # Set 1: Small integers 1..14 (Baseline)
    candidate_sets.append(list(range(1, 15)))
    
    # Set 2: Replace primes with composites to increase factor sharing
    # 11, 13 are primes. Replace with 15, 16, 18, 20 etc.
    candidate_sets.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # Set 3: Focus on divisors of 60 (1,2,3,4,5,6,10,12,15,20,30,60) + others
    # Need 14 distinct.
    candidate_sets.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24])
    
    # Set 4: Geometric-ish progression mixed with composites
    candidate_sets.append([1, 2, 4, 6, 8, 10, 12, 16, 18, 20, 24, 30, 32, 36])
    
    # Set 5: Randomized small composites
    base_composites = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30]
    for _ in range(5):
        random.shuffle(base_composites)
        # Ensure distinct and <= 10000
        vals = base_composites[:14]
        if len(set(vals)) == 14 and all(1 <= v <= 10000 for v in vals):
            candidate_sets.append(sorted(vals))
            
    # Remove duplicates in candidate_sets
    unique_sets = []
    seen = set()
    for s in candidate_sets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_sets.append(s)
    candidate_sets = unique_sets

    def run_greedy(values, seed=0):
        rng = random.Random(seed)
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
        # This helps in tie-breaking: prefer edges that use "common" sums/products
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
            candidates = []
            
            for edge in remaining:
                s, p = edge_info[edge]
                new_sums = 0 if s in distinct_sums else 1
                new_prods = 0 if p in distinct_products else 1
                
                # Primary metric: max of distinct counts after addition
                new_max = max(len(distinct_sums) + new_sums, len(distinct_products) + new_prods)
                
                # Secondary metric: Prefer edges that don't add new distinct values.
                # Tertiary metric: If adding new, prefer those that are "common" (high count in full graph)
                # because they are likely to be reused by other edges (though we remove edges, 
                # so this is a heuristic for "structural density").
                tie_breaker = 0
                if new_sums:
                    tie_breaker -= sum_counts.get(s, 0)
                if new_prods:
                    tie_breaker -= prod_counts.get(p, 0)
                    
                # Add small random noise to break ties dynamically and explore different paths
                noise = rng.random() * 0.01
                
                total_score = (new_max, tie_breaker + noise)
                
                if total_score < (best_score, best_tie_breaker):
                    best_score = new_max
                    best_tie_breaker = tie_breaker + noise
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
    
    # Try multiple seeds for each value set to escape local minima
    for vals in candidate_sets:
        for seed in range(5):
            raw_val, result = run_greedy(vals, seed)
            if raw_val < best_raw_value:
                best_raw_value = raw_val
                best_result = result
                
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result

# EVOLVE-BLOCK-END
