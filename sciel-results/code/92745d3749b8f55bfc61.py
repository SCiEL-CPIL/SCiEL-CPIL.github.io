    def construct():
        N = 40
        deadline = time.time() + 105
        
        best_edges = []
        best_count = 0
        
        # ... (definitions of would_create_c6, greedy_construct, local_search) ...
        
        # Phase 1: Multiple greedy restarts + deterministic polish
        seed = 0
        while time.time() < deadline:
            edge_set, adj_bits = greedy_construct(seed)
            # Polish
            new_edges, new_adj, new_count = local_search(edge_set, adj_bits)
            
            if new_count > best_count:
                best_count = new_count
                best_edges = list(new_edges)
            
            seed += 1
        
        return [[u, v] for u, v in best_edges]
    