    def find_k6_comp(n, adj):
        mask = (1 << n) - 1
        comp = [0] * n
        for i in range(n):
            comp[i] = (mask ^ adj[i]) & ~(1 << i)
        
        def dfs(candidates_mask, path):
            if len(path) == 6:
                return path
            if candidates_mask.bit_count() < 6 - len(path):
                return None
            
            # Iterate through all candidates
            c = candidates_mask
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                c ^= lsb
                
                # New candidates: neighbors of v in comp, that are in c (remaining candidates)
                # We must exclude v from c? c already had v removed.
                # But we need to ensure we only pick vertices > v? 
                # No, the mask `c` contains vertices that are "available".
                # If we don't enforce order, we might find the same clique multiple times?
                # Yes. To avoid duplicates, we should only consider vertices in `c` that are "after" v in some order.
                # But `c` is a bitset. We can just remove v from `c` and intersect with `comp[v]`.
                # However, if we don't restrict to > v, we might pick u < v later.
                # To prevent this, we should maintain the invariant that all future picks are from a set that doesn't contain previously picked vertices?
                # No, that's not enough.
                # Standard trick: when picking v, the next candidates must be a subset of the original candidates that are "greater" than v?
                # Or simply: remove v from the pool, and intersect with neighbors.
                # This allows picking any neighbor.
                # This might find the same set {1,2,3,4,5,6} in different orders?
                # Yes.
                # To avoid this, we can require that the next vertex is greater than the current one?
                # This requires sorting or a specific ordering.
                # Let's just use the list-based DFS which naturally handles order by index.
                # Or, use bitset but enforce index order.
                
                # Let's stick to the list-based DFS for correctness and simplicity, 
                # but optimize it by using bitsets for the "new_cands" calculation if possible?
                # No, list is fine for n=35.
                pass
            return None
        
        # List based DFS
        def dfs_list(candidates, path):
            if len(path) == 6:
                return path
            if len(candidates) < 6 - len(path):
                return None
            
            for i in range(len(candidates)):
                v = candidates[i]
                # New candidates: neighbors of v in comp, that are in candidates[i+1:]
                new_cands = []
                for j in range(i + 1, len(candidates)):
                    c = candidates[j]
                    if (comp[v] >> c) & 1:
                        new_cands.append(c)
                
                res = dfs_list(new_cands, path + [v])
                if res:
                    return res
            return None
        
        return dfs_list(list(range(n)), [])
