def find_short_cycle(adj, n, g):
    # Check 4-cycles
    for i in range(n // 2):
        for j in range(i + 1, n // 2):
            if len(set(adj[i]) & set(adj[j])) >= 2:
                return True, (i, j, 4)
    
    # Check 6-cycles
    # L1-R1-L2-R2-L3-R3-L1
    # Iterate L1, L2 with common R1
    # Then L2, L3 with common R2
    # Then L3, L1 with common R3
    half = n // 2
    for l1 in range(half):
        for l2 in range(half):
            if l1 == l2: continue
            common12 = set(adj[l1]) & set(adj[l2])
            if not common12: continue
            for r1 in common12:
                # L1-R1-L2
                # Now from L2, go to R2 (not R1)
                for r2 in adj[l2]:
                    if r2 == r1: continue
                    # L2-R2
                    # Now from R2, go to L3 (not L2)
                    for l3 in adj[r2]:
                        if l3 == l2: continue
                        # L3-R2-L2
                        # Now check if L3 and L1 share a neighbor R3
                        # R3 cannot be R1 or R2
                        common31 = set(adj[l3]) & set(adj[l1])
                        for r3 in common31:
                            if r3 == r1 or r3 == r2: continue
                            # Cycle: L1-R1-L2-R2-L3-R3-L1
                            return True, (l1, r1, l2, r2, l3, r3, 6)
                            
    # Check 8-cycles using BFS
    # If no 4 or 6, check for 8
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            if dist[u] >= 4: # 8-cycle means path length 7, so dist up to 4
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif dist[v] == dist[u] + 1 and v != parent[u]:
                    # Found cycle of length 2*dist[u] + 2
                    # If dist[u] = 3, length = 8
                    if 2 * dist[u] + 2 < g:
                        return True, (u, v, 2 * dist[u] + 2)
    return False, None
