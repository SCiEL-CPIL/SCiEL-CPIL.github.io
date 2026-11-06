    def has_q3_fast(adj):
        # adj is list of lists or numpy array
        # Precomputed subsets
        for s in subsets_8:
            # Quick degree check
            # s is tuple of 8 ints
            # We can build a small 8x8 adjacency matrix for this subset
            # and check properties
            
            # Optimization: Check if any vertex has degree != 3 in the induced subgraph
            # We can do this by counting neighbors in s for each v in s.
            
            # Let's use a bitmask representation for the induced subgraph
            # Map s[i] to i
            # adj_mask[i] will have bit j set if s[i] connected to s[j]
            
            adj_mask = [0]*8
            for i in range(8):
                v = s[i]
                mask = 0
                for j in range(8):
                    if i != j and adj[v][s[j]]:
                        mask |= (1 << j)
                adj_mask[i] = mask
            
            # Check degree 3
            valid = True
            for i in range(8):
                if bin(adj_mask[i]).count('1') != 3:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check bipartite
            # Since it's 3-regular, we just need to check if it's bipartite.
            # We can check if there are any odd cycles.
            # Or just BFS.
            
            color = [-1]*8
            for start in range(8):
                if color[start] != -1:
                    continue
                color[start] = 0
                queue = [start]
                while queue:
                    v = queue.pop(0)
                    neighbors = adj_mask[v]
                    # Iterate bits
                    # Simple loop
                    for u in range(8):
                        if (neighbors >> u) & 1:
                            if color[u] == -1:
                                color[u] = 1 - color[v]
                                queue.append(u)
                            elif color[u] == color[v]:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return True
                
        return False
