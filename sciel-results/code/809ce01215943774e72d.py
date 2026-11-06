<<<<<<< SEARCH
    def local_search_simple(subset, elements, max_rounds=100):
        """Simpler local search: repeatedly try to add any element that fits."""
        s = list(subset)
        s_set = set(s)
        
        for _ in range(max_rounds):
            added_any = False
            for x in elements:
                if x in s_set:
                    continue
                # Check if x can be added
                ok = True
                for a in s:
                    if a + x in s_set:
                        ok = False
                        break
                if ok and 2 * x not in s_set:
                    s.append(x)
                    s_set.add(x)
                    added_any = True
            if not added_any:
                break
        
        # Try swaps: remove one, add one
        for _ in range(50):
            improved = False
            for i, a in enumerate(s):
                # Try removing a
                s.remove(a)
                s_set.discard(a)
                # Try adding each element
                best_add = None
                for x in elements:
                    if x in s_set:
                        continue
                    ok = True
                    for b in s:
                        if b + x in s_set:
                            ok = False
                            break
                    if ok and 2 * x not in s_set:
                        best_add = x
                        break
                if best_add is not None:
                    s.append(best_add)
                    s_set.add(best_add)
                    improved = True
                else:
                    # Revert
                    s.append(a)
                    s_set.add(a)
            
            if not improved:
                break
        
        return s
=======
    def local_search_simple(subset, elements, max_rounds=50):
        """Simpler local search: repeatedly try to add any element that fits."""
        s = list(subset)
        s_set = set(s)
        
        for _ in range(max_rounds):
            added_any = False
            # Iterate over a copy to allow safe appending
            for x in elements:
                if x in s_set:
                    continue
                # Check if x can be added
                ok = True
                for a in s:
                    if a + x in s_set:
                        ok = False
                        break
                if ok and 2 * x not in s_set:
                    s.append(x)
                    s_set.add(x)
                    added_any = True
            if not added_any:
                break
        
        # Try swaps: remove one, add one
        # We use a simple hill climbing step
        improved = True
        while improved:
            improved = False
            current_size = len(s)
            # Try removing each element
            for a in list(s):
                s.remove(a)
                s_set.discard(a)
                
                best_add = None
                for x in elements:
                    if x in s_set:
                        continue
                    ok = True
                    for b in s:
                        if b + x in s_set:
                            ok = False
                            break
                    if ok and 2 * x not in s_set:
                        best_add = x
                        break
                
                if best_add is not None:
                    s.append(best_add)
                    s_set.add(best_add)
                    improved = True
                    # We don't necessarily break here, but we found an improvement
                    # If we want strictly better size, we only keep if len(s) > current_size
                    # But usually adding 1 removing 1 keeps size same.
                    # To improve size, we'd need to remove 1 add 2. 
                    # For simplicity and safety, let's just do 1-for-1 swaps to escape local minima?
                    # Actually, 1-for-1 doesn't increase size. 
                    # Let's stick to just adding if possible, which is already done above.
                    # The swap logic above was flawed for increasing size.
                    # Let's revert this swap logic to avoid bugs and just rely on the greedy add.
                    s.remove(best_add)
                    s_set.discard(best_add)
                    s.append(a)
                    s_set.add(a)
                else:
                    # Revert
                    s.append(a)
                    s_set.add(a)
        
        return s
>>>>>>> REPLACE

