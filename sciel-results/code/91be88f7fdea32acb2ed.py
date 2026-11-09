<<<<<<< SEARCH
    def hill_climb_grow(s, rng):
        """
        Aggressively try to grow the set by finding any valid addition.
        """
        a = list(s)
        sums = sum_set(s)
        
        # Try all possible additions
        for x in range(1, 285):
            if x in a:
                continue
            new_sums = {x + y for y in a} | {2 * x}
            if not (new_sums & sums):
                a.append(x)
                sums |= new_sums
        
        return sorted(a)
=======
    def hill_climb_grow(s, rng):
        """
        Aggressively try to grow the set by finding any valid addition.
        Uses O(N) conflict checking.
        """
        a = list(s)
        a_set = set(a)
        sums = sum_set(s)
        
        # Try all possible additions
        for x in range(1, 285):
            if x in a_set:
                continue
            conflict = False
            for y in a:
                if (x + y) in sums:
                    conflict = True
                    break
            if not conflict and (2 * x) in sums:
                conflict = True
            if not conflict:
                a.append(x)
                a_set.add(x)
                for y in a:
                    sums.add(x + y)
                sums.add(2 * x)
        
        return sorted(a)
    
    def swap_improve(s, rng, time_limit=3.0):
        """
        Targeted swap improvement: try to replace elements that block growth.
        For each element, compute how many other elements it conflicts with.
        Replace high-conflict elements with better alternatives.
        """
        a = list(s)
        a_set = set(a)
        sums = sum_set(s)
        best = len(a)
        best_set = list(a)
        
        start = time.time()
        
        while time.time() - start < time_limit:
            improved = False
            
            # For each non-prescribed element, try to find a better replacement
            for i in range(len(a)):
                if a[i] in (1, 2, 5, 11):
                    continue
                
                old_elem = a[i]
                a_without = [v for j, v in enumerate(a) if j != i]
                a_without_set = set(a_without)
                sums_without = sum_set(a_without)
                
                # Find all elements that can replace old_elem
                valid_replacements = []
                for x in range(1, 285):
                    if x in a_without_set:
                        continue
                    conflict = False
                    for y in a_without:
                        if (x + y) in sums_without:
                            conflict = True
                            break
                    if not conflict and (2 * x) in sums_without:
                        conflict = True
                    if not conflict:
                        valid_replacements.append(x)
                
                # Also try to grow: remove old_elem, add two new ones
                if valid_replacements:
                    # Try each replacement
                    for x in valid_replacements[:5]:  # Limit to keep fast
                        a[i] = x
                        a_set.discard(old_elem)
                        a_set.add(x)
                        sums = sums_without
                        for y in a:
                            sums.add(x + y)
                        sums.add(2 * x)
                        
                        # Try to grow further
                        for z in range(1, 285):
                            if z in a_set:
                                continue
                            conflict = False
                            for y in a:
                                if (z + y) in sums:
                                    conflict = True
                                    break
                            if not conflict and (2 * z) in sums:
                                conflict = True
                            if not conflict:
                                a.append(z)
                                a_set.add(z)
                                for y in a:
                                    sums.add(z + y)
                                sums.add(2 * z)
                                improved = True
                                if len(a) > best:
                                    best = len(a)
                                    best_set = list(a)
                                break
                        
                        # Revert if no improvement from growth
                        if not improved:
                            a.pop() if len(a) > len(best_set) else None
                            # Actually, let's just revert the swap if it didn't help
                            a[i] = old_elem
                            a_set.discard(x)
                            a_set.add(old_elem)
                            sums = sum_set(a)
                            improved = False
                        else:
                            break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return best_set
>>>>>>> REPLACE

