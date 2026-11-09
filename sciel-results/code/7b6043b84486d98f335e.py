def _has_q3(edge_set, n=16):
    adj = [0] * n
    for u, v in edge_set:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        
    vertices = list(range(n))
    # Precompute 4-subsets
    # We need to check all 8-subsets.
    # Optimize: iterate 8-subsets.
    for combo in itertools.combinations(vertices, 8):
        # Check bipartitions
        u0 = min(combo)
        rest = [v for v in combo if v != u0]
        for u_subset in itertools.combinations(rest, 3):
            U = [u0] + list(u_subset)
            W = [v for v in combo if v not in U]
            
            # Check if missing edges form a matching
            # Missing edge (u,w) if not (adj[u] & (1<<w))
            # We need to ensure that for each u in U, the missing neighbors in W form a set of size <= 1?
            # No, the set of ALL missing edges must be a matching.
            # This means:
            # 1. No two missing edges share a u.
            # 2. No two missing edges share a w.
            
            # Let's collect missing edges
            missing_u = []
            missing_w = []
            valid = True
            for u in U:
                # Find w in W not connected to u
                # We can do this by iterating W
                found_missing_w = -1
                for w in W:
                    if not (adj[u] & (1 << w)):
                        if found_missing_w != -1:
                            valid = False
                            break
                        found_missing_w = w
                if not valid:
                    break
                if found_missing_w != -1:
                    missing_u.append(u)
                    missing_w.append(found_missing_w)
            
            if not valid:
                continue
                
            # Check if missing_w has duplicates
            if len(set(missing_w)) != len(missing_w):
                continue
                
            # If we get here, missing edges form a matching
            return True
            
    return False
