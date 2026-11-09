"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Model as MWIS in conflict graph
        # Nodes: (v, c) for v in 0..n-1, c in lists[v]
        # Edges: 
        # 1. (v, c1) - (v, c2) if c1 != c2
        # 2. (u, c) - (v, c) if u-v in edges
        
        # 1. Create candidate list
        candidates = []
        cand_id = {} # (v, c) -> id
        for v in range(n):
            for c in lists[v]:
                cid = len(candidates)
                candidates.append((v, c))
                cand_id[(v, c)] = cid
        
        M = len(candidates)
        
        # 2. Precompute masks
        # verts_mask[v]: bitmask of candidate ids for vertex v
        verts_mask = [0] * n
        for v in range(n):
            for c in lists[v]:
                cid = cand_id[(v, c)]
                verts_mask[v] |= (1 << cid)
                
        # conflict_mask[i]: bitmask of candidate ids that conflict with candidate i
        # Includes i itself
        conflict_mask = [0] * M
        
        for i in range(M):
            v, c = candidates[i]
            conflict_mask[i] |= (1 << i) # Conflict with self
            
            # Conflict with other colors of same vertex
            for c2 in lists[v]:
                if c2 != c:
                    j = cand_id[(v, c2)]
                    conflict_mask[i] |= (1 << j)
            
            # Conflict with same color of neighbors
            for u in adj[v]:
                if c in lists[u]:
                    j = cand_id[(u, c)]
                    conflict_mask[i] |= (1 << j)
        
        # 3. Backtracking with bitsets
        # State: current_set (bitmask of chosen candidates), available (bitmask of candidates compatible with current_set)
        # We want to find an independent set of size n.
        
        solution = [-1] * n
        
        def backtrack(current_set, available):
            # Check if all vertices are assigned
            # A vertex v is assigned if current_set has any bit from verts_mask[v]
            # Since we maintain independence, there is at most one.
            
            # Optimization: Check if available is empty
            if available == 0:
                # If we haven't assigned all vertices, fail
                # Check if all vertices have a chosen candidate
                for v in range(n):
                    if (current_set & verts_mask[v]) == 0:
                        return False
                return True
            
            # Find unassigned vertex with min remaining options (MRV)
            min_count = M + 1
            best_v = -1
            
            for v in range(n):
                # If already assigned, skip
                if (current_set & verts_mask[v]) != 0:
                    continue
                # Count available options for v
                opts = available & verts_mask[v]
                cnt = opts.bit_count()
                if cnt == 0:
                    return False # Dead end
                if cnt < min_count:
                    min_count = cnt
                    best_v = v
            
            if best_v == -1:
                # All assigned
                return True
            
            # Get options for best_v
            opts = available & verts_mask[best_v]
            
            # Iterate through set bits in opts
            # We can extract them one by one
            o = opts
            while o:
                # Get lowest set bit
                lsb = o & (-o)
                o ^= lsb
                cid = lsb.bit_length() - 1
                
                # Try this candidate
                v, c = candidates[cid]
                
                # Update state
                new_current = current_set | lsb
                # Remove cid and all its conflicts from available
                new_available = available & ~conflict_mask[cid]
                
                if backtrack(new_current, new_available):
                    # Record solution
                    solution[v] = c
                    # We need to propagate the rest of the solution back up?
                    # Actually, since we return True, we can just fill the solution array
                    # But we need to ensure the rest of the array is filled correctly.
                    # The recursive call fills the solution for the deeper levels.
                    # The current level just needs to set solution[v] = c.
                    # The deeper levels will set their own.
                    return True
            
            return False
        
        if backtrack(0, (1 << M) - 1):
            return solution
        else:
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        col = solve_instance(n, edges, lists)
        out.append(col)
    
    # Ensure types are correct
    final_out = []
    for col_list in out:
        # Ensure it's a list of Python ints
        final_out.append([int(c) for c in col_list])
    
    return final_out
# EVOLVE-BLOCK-END
