import random

def construct():
    # Try n=47 first, then 46
    for n in [47, 46]:
        min_deg_required = n - 39
        
        # Try multiple random instances
        for trial in range(200):
            # Use a deterministic seed based on trial to ensure reproducibility if needed,
            # but random is fine. Let's use random for variety.
            
            # Initialize G as empty
            adj = [set() for _ in range(n)]
            
            # Add random edges with p=0.3
            p = 0.3
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        adj[u].add(v)
                        adj[v].add(u)
                        
            # Check initial min degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg < min_deg_required:
                continue
                
            # Iterate to remove C4s
            success = True
            # Limit iterations to prevent infinite loops if logic fails
            max_iter = 5000 
            for _ in range(max_iter):
                # Find a C4
                c4_found = False
                for a in range(n):
                    if not c4_found:
                        for c in range(a+1, n):
                            if c in adj[a]:
                                continue # a, c must be non-adjacent
                            # Find common neighbors
                            # Iterate over smaller set for speed
                            if len(adj[a]) < len(adj[c]):
                                common = [x for x in adj[a] if x in adj[c]]
                            else:
                                common = [x for x in adj[c] if x in adj[a]]
                                
                            if len(common) >= 2:
                                # Found C4: a - b - c - d - a
                                b = common[0]
                                d = common[1]
                                
                                # Edges in C4: (a,b), (b,c), (c,d), (d,a)
                                edges_to_try = [(a,b), (b,c), (c,d), (d,a)]
                                removed = False
                                for u, v in edges_to_try:
                                    # Check if we can remove (u,v)
                                    # We need deg(u) > min_req and deg(v) > min_req
                                    if len(adj[u]) > min_deg_required and len(adj[v]) > min_deg_required:
                                        adj[u].remove(v)
                                        adj[v].remove(u)
                                        removed = True
                                        c4_found = True
                                        break
                                if not removed:
                                    # Cannot remove any edge from this C4 without violating degree
                                    # This means we are stuck?
                                    # Not necessarily, maybe another C4 has removable edges.
                                    # But if we loop through all and none are removable, we fail.
                                    # For now, we just don't mark c4_found, so we continue searching.
                                    # However, if we find NO removable edges in ANY C4, we will loop forever.
                                    # To detect this, we can count how many C4s we found but couldn't fix.
                                    # Simpler: if we go through all pairs and find C4s but no removals, fail.
                                    pass
                    if not c4_found:
                        continue
                    else:
                        # Found and removed one, restart search
                        break
                
                if not c4_found:
                    # No C4 found in this pass
                    # Verify no C4 exists
                    # The loop above checks all pairs. If none found, it's C4-free.
                    break
            
            # Final verification
            is_c4_free = True
            for a in range(n):
                for c in range(a+1, n):
                    if c in adj[a]:
                        continue
                    if len(adj[a]) < len(adj[c]):
                        common = [x for x in adj[a] if x in adj[c]]
                    else:
                        common = [x for x in adj[c] if x in adj[a]]
                    if len(common) >= 2:
                        is_c4_free = False
                        break
                if not is_c4_free:
                    break
            
            if is_c4_free:
                min_deg_final = min(len(adj[i]) for i in range(n))
                if min_deg_final >= min_deg_required:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                    
    # Fallback
    return {'n': 39, 'edges': []}
