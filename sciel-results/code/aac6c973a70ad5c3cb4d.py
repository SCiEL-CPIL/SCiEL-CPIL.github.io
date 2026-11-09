    import time

    def construct():
        candidates = list(range(2, 129))
        n = len(candidates)
        best_len = 0
        best_values = []
        
        # Greedy to set initial best
        def greedy():
            vals = []
            prods = set()
            suff = [1]
            for x in candidates:
                new_suff = [x * s for s in suff]
                if not prods.intersection(new_suff):
                    vals.append(x)
                    prods.update(new_suff)
                    suff = new_suff
            return vals
        
        init_vals = greedy()
        best_len = len(init_vals)
        best_values = init_vals
        
        start_time = time.time()
        
        def search(idx, values, products, suffixes):
            nonlocal best_len, best_values
            
            # Time check
            if time.time() - start_time > 55:
                return
                
            # Pruning
            if len(values) + (n - idx) <= best_len:
                return
                
            if idx == n:
                if len(values) > best_len:
                    best_len = len(values)
                    best_values = list(values)
                return
                
            x = candidates[idx]
            
            # Calculate new suffixes
            new_suffixes = [x * s for s in suffixes]
            
            # Check validity
            valid = True
            for p in new_suffixes:
                if p in products:
                    valid = False
                    break
            
            # Heuristic: Exclude first?
            # Excluding keeps products set smaller, potentially allowing more future inclusions.
            # Including adds constraints.
            # Let's try Exclude first.
            
            search(idx + 1, values, products, suffixes)
            
            if valid:
                added = set(new_suffixes)
                products.update(added)
                values.append(x)
                
                search(idx + 1, values, products, new_suffixes)
                
                values.pop()
                products.difference_update(added)
                
        search(0, [], set(), [1])
        
        return best_values
    