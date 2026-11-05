    def is_3_colorable(adj):
        color_masks = [0, 0, 0]
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = idx
            # Try each color
            for c in range(3):
                if (adj[v] & color_masks[c]) == 0:
                    color_masks[c] |= (1 << v)
                    if backtrack(idx + 1):
                        return True
                    color_masks[c] ^= (1 << v)
            
            return False
        
        return backtrack(0)
