def _local_search_efficient(adj, in_adj, n, time_limit):
    bad_sets = set()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if (in_adj[i] & in_adj[j] & in_adj[k] & in_adj[l]) == 0:
                        bad_sets.add((i, j, k, l))
                        
    bad_count = len(bad_sets)
    if bad_count == 0:
        return adj, in_adj, n, True
        
    while time.time() < time_limit:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v: continue
        
        # Generate affected sets
        affected = []
        others_u = [x for x in range(n) if x != u]
        for a in range(len(others_u)):
            for b in range(a + 1, len(others_u)):
                for c in range(b + 1, len(others_u)):
                    affected.append(tuple(sorted((u, others_u[a], others_u[b], others_u[c]))))
                    
        others_v = [x for x in range(n) if x != v]
        for a in range(len(others_v)):
            for b in range(a + 1, len(others_v)):
                for c in range(b + 1, len(others_v)):
                    affected.append(tuple(sorted((v, others_v[a], others_v[b], others_v[c]))))
                    
        affected = list(set(affected))
        
        removed = []
        added = []
        
        for s in affected:
            if s in bad_sets:
                bad_sets.remove(s)
                removed.append(s)
                
        # Flip
        adj, in_adj = _flip_edge(adj, in_adj, n, u, v)
        
        for s in affected:
            if (in_adj[s[0]] & in_adj[s[1]] & in_adj[s[2]] & in_adj[s[3]]) == 0:
                bad_sets.add(s)
                added.append(s)
                
        new_bad_count = len(bad_sets)
        
        if new_bad_count >= bad_count:
            # Revert
            adj, in_adj = _flip_edge(adj, in_adj, n, u, v)
            for s in added:
                bad_sets.remove(s)
            for s in removed:
                bad_sets.add(s)
            # bad_count remains same
        else:
            bad_count = new_bad_count
            if bad_count == 0:
                return adj, in_adj, n, True
                
    return adj, in_adj, n, bad_count == 0
