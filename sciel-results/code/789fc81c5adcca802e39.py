"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 14
    m = 35
    
    # Candidate value sets designed to minimize distinct sums and products
    # Set A: Standard 1..14
    # Set B: Replaced primes 11,13 with composites 15,16
    # Set C: Removed 7,11,13, added 15,16,18 (more composites)
    
    candidate_sets = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18],
    ]
    
    from itertools import combinations
    
    def run_greedy(values):
        all_edges = list(combinations(range(n), 2))
        
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
        # to estimate "reusability" of a new sum/product
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
                
                # Secondary metric: If new, prefer sums/products that are common in the remaining graph
                # This is a heuristic for "amortized cost"
                tie_breaker = 0
                if new_sums:
                    # Higher count is better (lower cost), so use -count or 1/count
                    # We want to minimize cost, so if count is high, cost is low.
                    # Let's use -count as a reward, or just count as a penalty for uniqueness.
                    # If s is new, and sum_counts[s] is 10, it's good.
                    # So we want to maximize sum_counts[s] for new sums.
                    # In a minimization loop, we can subtract it.
                    tie_breaker -= sum_counts.get(s, 0)
                if new_prods:
                    tie_breaker -= prod_counts.get(p, 0)
                    
                # Combined score for comparison
                # We want to minimize new_max.
                # If tie, we want to minimize tie_breaker (which is negative of rewards, so more negative is better).
                # So we want the smallest tie_breaker (most negative).
                
                if new_max < best_score or (new_max == best_score and tie_breaker < best_tie_breaker):
                    best_score = new_max
                    best_edge = edge
                    best_tie_breaker = tie_breaker
            
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
    
    for vals in candidate_sets:
        raw_val, result = run_greedy(vals)
        if raw_val < best_raw_value:
            best_raw_value = raw_val
            best_result = result
            
    return best_result

# EVOLVE-BLOCK-END
