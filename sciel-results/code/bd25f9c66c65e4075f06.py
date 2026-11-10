"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    # Try several candidate value sets and edge selections
    # Strategy: use values that create many sum collisions and product collisions
    
    best_score = float('inf')
    best_result = None
    
    # Candidate value sets
    candidate_values = []
    
    # Small consecutive
    candidate_values.append(list(range(1, 15)))
    
    # Symmetric around center: pairs sum to constant
    # e.g., 1,2,...,14 - already have
    # Try: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    candidate_values.append([2**i for i in range(14)])
    
    # Try values with more structure
    # Use values where v[i] + v[j] collides often
    # e.g., arithmetic progression
    candidate_values.append(list(range(1, 15)))
    
    # Try: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27
    candidate_values.append(list(range(1, 28, 2)))
    
    # Try: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (powers of 2)
    # already have
    
    # Try a set designed for sum collisions:
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
    # Pairs summing to 15: (1,14),(2,13),(3,12),(4,11),(5,10),(6,9),(7,8) = 7 edges
    # Pairs summing to 14: (1,13),(2,12),(3,11),(4,10),(5,9),(6,8) = 6 edges
    # etc.
    
    all_edges = list(itertools.combinations(range(n), 2))
    
    for values in candidate_values:
        # Try to select 35 edges that minimize max(distinct_sums, distinct_products)
        
        # Precompute sums and products for all edges
        edge_data = []
        for (i, j) in all_edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_data.append((i, j, s, p))
        
        # Greedy: try to pick edges that reuse sums and products
        # Sort edges by how "common" their sum and product are
        from collections import Counter
        sum_counts = Counter(s for _, _, s, _ in edge_data)
        prod_counts = Counter(p for _, _, _, p in edge_data)
        
        # Score each edge: prefer edges with common sums and products
        def edge_score(ed):
            i, j, s, p = ed
            return (sum_counts[s], prod_counts[p])
        
        # Try multiple random orderings with bias toward common sums/products
        for attempt in range(50):
            random.seed(attempt)
            # Shuffle but bias
            edges_copy = edge_data[:]
            # Sort by score descending, then add noise
            scored = [(edge_score(ed), random.random(), ed) for ed in edges_copy]
            scored.sort(key=lambda x: (-x[0][0], -x[0][1], x[1]))
            
            selected = []
            used_sums = set()
            used_prods = set()
            
            for _, _, ed in scored:
                if len(selected) >= m:
                    break
                i, j, s, p = ed
                selected.append((i, j))
                used_sums.add(s)
                used_prods.add(p)
            
            if len(selected) < m:
                continue
            
            score = max(len(used_sums), len(used_prods))
            if score < best_score:
                best_score = score
                best_result = {
                    'values': list(values),
                    'edges': [[i, j] for i, j in selected]
                }
    
    if best_result is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in all_edges[:m]]
        return {'values': values, 'edges': edges}
    
    return best_result
# EVOLVE-BLOCK-END
