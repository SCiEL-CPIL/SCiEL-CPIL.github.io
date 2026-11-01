def get_closed_walk_vectors(adj_edges, max_len):
    vectors = set()
    num_edges = 15
    
    # Pre-allocate counts? 
    # We can't easily reuse array in DFS stack without copying.
    # But path is small (<=15).
    
    for start in range(10):
        stack = [(start, [])]
        while stack:
            curr, path = stack.pop()
            L = len(path)
            
            if L > 0 and curr == start:
                if L <= max_len:
                    # Optimization: Only add if length is >= 5? 
                    # Girth of Petersen is 5, so no walks < 5.
                    if L >= 5:
                        counts = [0] * num_edges
                        for e in path:
                            counts[e] += 1
                        vectors.add(tuple(counts))
            
            if L < max_len:
                for next_v, e_idx in adj_edges[curr]:
                    stack.append((next_v, path + [e_idx]))
                    
    return vectors
