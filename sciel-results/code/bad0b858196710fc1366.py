def construct():
    import itertools
    
    # Values 1 to 14
    values = list(range(1, 15))
    n = 14
    
    # Generate all edges
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Calculate sum and product for each edge
    edge_data = []
    for u, v in all_edges:
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_data.append((u, v, s, p))
        
    # We want to select 35 edges to minimize max(distinct_sums, distinct_products)
    # Strategy: Focus on central sums to minimize distinct sums.
    # Sums range from 3 to 27.
    # Counts per sum:
    # 13: 5, 14: 6, 15: 7, 16: 6, 17: 5, 18: 4, 19: 3
    # Total 36.
    
    # Filter edges with sums in [13, 19]
    candidate_edges = [e for e in edge_data if 13 <= e[2] <= 19]
    
    # We have 36 candidates. We need 35.
    # To minimize distinct products, we might want to drop the edge with the most "unique" product
    # or just pick the first 35.
    # Let's try picking the first 35.
    selected_edges = candidate_edges[:35]
    
    # Calculate distinct sums and products
    sums = set()
    products = set()
    edges_out = []
    for u, v, s, p in selected_edges:
        sums.add(s)
        products.add(p)
        edges_out.append([u, v])
        
    # Check if this is good
    # If distinct products is too high, maybe we should try a different set of values?
    # But 1..14 is the best for product collisions.
    
    # Let's also try to optimize the selection of the 35 edges from the 36 candidates
    # by trying all 36 subsets of size 35 (drop one edge) and picking the best one.
    best_score = float('inf')
    best_edges = None
    
    # There are only 36 choices to drop
    for i in range(len(candidate_edges)):
        # Drop candidate_edges[i]
        trial_edges = [e for j, e in enumerate(candidate_edges) if j != i]
        
        s_set = set()
        p_set = set()
        for u, v, s, p in trial_edges:
            s_set.add(s)
            p_set.add(p)
            
        score = max(len(s_set), len(p_set))
        if score < best_score:
            best_score = score
            best_edges = [[u, v] for u, v, s, p in trial_edges]
            
    return {
        'values': values,
        'edges': best_edges
    }
