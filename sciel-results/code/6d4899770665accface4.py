<<<<<<< SEARCH
        def backtrack_v(idx):
            nonlocal used_local
            if time.time() > deadline:
                return None
            if idx == len(order_v):
                return True
            v = order_v[idx]
            d = get_domain(v)
            if not d:
                return None
            vals = list(d)
            rng.shuffle(vals)
            for h in vals:
                assign[v] = h
                p = parent[v]
                ek = None
                if p != -1:
                    ek = _edge_key(assign[p], h)
                    used_local.add(ek)
                if backtrack_v(idx + 1) is not None:
                    return True
                if ek is not None:
                    used_local.discard(ek)
                assign[v] = -1
            return None
        
        emb = backtrack_v(0)
=======
        def backtrack_v(depth):
            nonlocal used_local
            if time.time() > deadline:
                return None
            if depth == k:
                return True
            
            # MRV: Find unassigned vertex with smallest domain
            unassigned = [v for v in range(k) if assign[v] == -1]
            if not unassigned:
                return True
            
            best_v = -1
            best_d = None
            min_size = k + 1
            
            for v in unassigned:
                d = get_domain(v)
                if len(d) < min_size:
                    min_size = len(d)
                    best_v = v
                    best_d = d
                    if min_size == 0:
                        return None
                    if min_size == 1:
                        break
            
            vals = list(best_d)
            rng.shuffle(vals)
            
            for h in vals:
                assign[best_v] = h
                p = parent[best_v]
                ek = None
                if p != -1:
                    ek = _edge_key(assign[p], h)
                    used_local.add(ek)
                
                if backtrack_v(depth + 1) is not None:
                    return True
                
                if ek is not None:
                    used_local.discard(ek)
                assign[best_v] = -1
            
            return None
        
        emb = backtrack_v(0)
>>>>>>> REPLACE
