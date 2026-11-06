    def evict_vertex(current_idx, exclude_v):
        candidates = [u for u in in_memory if u != exclude_v]
        if not candidates:
            return None
            
        best_u = None
        best_key = None
        
        for u in candidates:
            fu = get_future_uses(u, current_idx)
            in_stored = u in stored
            od = out_degree[u]
            
            # Key: (not in_stored, fu, od)
            # We want to evict:
            # 1. Vertices already in stored (free evict)
            # 2. Vertices with low future uses
            # 3. Vertices with low out_degree
            
            key = (0 if in_stored else 1, fu, od)
            
            if best_key is None or key < best_key:
                best_key = key
                best_u = u
                
        if best_u is None:
            return None
            
        in_memory.discard(best_u)
        
        if best_u not in stored:
            if best_u in outputs or out_degree[best_u] > 1:
                ops.append(('store', best_u))
                stored.add(best_u)
            else:
                ops.append(('delete', best_u))
                
        return best_u
