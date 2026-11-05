def construct():
    # ... helper functions ...
    
    # 1. Find a valid graph on 20 vertices (Known to exist, should be fast)
    adj_20 = optimize(20, time_limit=20)
    if adj_20 is None:
        # Fallback to smaller? Or just return empty?
        # The problem says 1 <= n <= 22.
        # If we can't find 20, we try 19?
        # But 20 is known. If optimize fails, it's a bug or bad luck.
        # Let's try a few more restarts for 20 if it fails?
        # Or just return a small valid graph.
        return {'n': 5, 'edges': []}
        
    # We have a valid 20-vertex graph.
    result_20 = adj_20
    
    # 2. Try to find a valid graph on 21 vertices
    # Use remaining time.
    time_spent = time.time() - start_total
    remaining = 120 - time_spent - 5 # 5s buffer
    
    if remaining > 10:
        adj_21 = optimize(21, time_limit=remaining)
        if adj_21 is not None:
            # Return 21
            edges = []
            for i in range(21):
                for j in range(i + 1, 21):
                    if (adj_21[i] & (1 << j)):
                        edges.append([i, j])
            return {'n': 21, 'edges': edges}
            
    # 3. Try to find a valid graph on 22 vertices
    # If we have time left?
    # If we didn't find 21, we probably don't have time for 22 if we used up the budget.
    # But if 21 search was fast (failed early?), maybe?
    # Unlikely.
    
    # Return 20
    edges = []
    for i in range(20):
        for j in range(i + 1, 20):
            if (result_20[i] & (1 << j)):
                edges.append([i, j])
    return {'n': 20, 'edges': edges}
