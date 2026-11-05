    def find_c6_path(u, v, adj):
        def dfs_path(node, depth, path, visited):
            if depth == 4:
                if (1 << v) & adj[node]:
                    return path + [v]
                return None
            
            # Get neighbors of node
            # We can precompute neighbors list? No, adj changes.
            # Just iterate bits.
            mask = adj[node]
            while mask:
                lsb = mask & -mask
                nxt = lsb.bit_length() - 1
                mask ^= lsb
                
                if nxt in visited or nxt == v:
                    continue
                res = dfs_path(nxt, depth + 1, path + [nxt], visited | {nxt})
                if res:
                    return res
            return None
        
        return dfs_path(u, 0, [u], {u})
