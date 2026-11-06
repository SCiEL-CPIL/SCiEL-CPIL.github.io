    def try_extend(s, k, phi_vals_cache=None):
        # We need phi(s-1) and phi(s+k)
        # We don't have a cache, so we compute them
        # Check left
        new_s = s
        new_k = k
        # We need to check if phi(s-1) is distinct from phi(s)..phi(s+k-1)
        # And phi(s+k) is distinct from phi(s)..phi(s+k-1)
        
        # To do this efficiently, we need the set of values.
        # We can reconstruct the set from the segment if we have it, 
        # or just recompute the window.
        # Recomputing a window of size k is cheap if k is small (<200).
        
        vals = [phi(new_s + i) for i in range(new_k)]
        val_set = set(vals)
        
        # Try extending to the left
        while new_s > 1:
            prev_val = phi(new_s - 1)
            if prev_val in val_set:
                break
            val_set.add(prev_val)
            new_s -= 1
            new_k += 1
            if new_k > 200: # Max k constraint
                break
                
        # Try extending to the right
        while new_s + new_k - 1 < 10**12:
            next_val = phi(new_s + new_k)
            if next_val in val_set:
                break
            val_set.add(next_val)
            new_k += 1
            if new_k > 200:
                break
                
        return new_s, new_k
