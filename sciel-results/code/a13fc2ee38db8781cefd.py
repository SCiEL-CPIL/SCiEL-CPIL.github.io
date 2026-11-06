<<<<<<< SEARCH
        def dfs_mask(mask_available, count):
            if count == target:
                return True
            if mask_available == 0:
                return False
            # Pruning: popcount(mask_available) + count < target
            if bin(mask_available).count('1') + count < target:
                return False
            
            # Pick the lowest set bit (vertex u)
            # lsb = mask_available & -mask_available
            # u = lsb.bit_length() - 1
            
            # Option 1: Skip u
            mask_without_u = mask_available & ~(1 << u) # wait, need to find u first
            lsb = mask_available & -mask_available
            u = lsb.bit_length() - 1
            
            if dfs_mask(mask_available ^ lsb, count):
                return True
            
            # Option 2: Take u
            # New available mask: remove u and all neighbors of u
            new_mask = mask_available & ~(adj_mask[u] | (1 << u))
            if dfs_mask(new_mask, count + 1):
                return True
            
            return False
=======
        def dfs_mask(mask_available, count):
            if count == target:
                return True
            if mask_available == 0:
                return False
            # Pruning: popcount(mask_available) + count < target
            if bin(mask_available).count('1') + count < target:
                return False
            
            # Pick the lowest set bit (vertex u)
            lsb = mask_available & -mask_available
            u = lsb.bit_length() - 1
            
            # Option 1: Skip u
            if dfs_mask(mask_available ^ lsb, count):
                return True
            
            # Option 2: Take u
            # New available mask: remove u and all neighbors of u
            new_mask = mask_available & ~(adj_mask[u] | (1 << u))
            if dfs_mask(new_mask, count + 1):
                return True
            
            return False
>>>>>>> REPLACE

I will also remove the unused `dfs` function to clean up the code, though it's not strictly necessary for fixing the crash. It saves a tiny bit of time.

