<<<<<<< SEARCH
def _local_search(selected, sums, rng, time_limit):
    """
    Local search to minimize size while maintaining maximality.
    Tries single removals, double removals, and swaps.
    """
    best_size = len(selected)
    best_sel = list(selected)
    best_sums = set(sums)
    start = time.time()
    
    while time.time() - start < time_limit and time.time() < DEADLINE:
        improved = False
        cur_sel = list(selected)
        cur_sums = set(sums)
        
        # --- Single Removal Pass ---
        for i in range(len(cur_sel)):
            if time.time() > DEADLINE:
                return best_size, best_sel, best_sums
            
            e = cur_sel[i]
            new_sel = cur_sel[:i] + cur_sel[i+1:]
            
            removed_sums = {2 * e}
            for a in new_sel:
                removed_sums.add(e + a)
            
            new_sums = cur_sums - removed_sums
            
            if _is_maximal(new_sel, new_sums):
                selected = new_sel
                sums = new_sums
                improved = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sel = list(selected)
                    best_sums = set(sums)
                break
        
        if not improved:
            # --- Double Removal Pass ---
            for i in range(len(cur_sel)):
                if time.time() > DEADLINE:
                    return best_size, best_sel, best_sums
                for j in range(i + 1, len(cur_sel)):
                    new_sel = [x for k, x in enumerate(cur_sel) if k != i and k != j]
                    
                    new_sums = set()
                    for a in range(len(new_sel)):
                        for b in range(a, len(new_sel)):
                            new_sums.add(new_sel[a] + new_sel[b])
                    
                    if _is_maximal(new_sel, new_sums):
                        selected = new_sel
                        sums = new_sums
                        improved = True
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_sel = list(selected)
                            best_sums = set(sums)
                        break
                if improved:
                    break
        
        if not improved:
            # --- Swap Pass: try replacing one element with another ---
            sset = set(cur_sel)
            for i in range(len(cur_sel)):
                if time.time() > DEADLINE:
                    return best_size, best_sel, best_sums
                e = cur_sel[i]
                
                # Try replacing e with a random missing element
                for _ in range(5):
                    if time.time() > DEADLINE:
                        return best_size, best_sel, best_sums
                    
                    # Pick a random element not in cur_sel
                    candidates = [x for x in range(1, N + 1) if x not in sset]
                    if not candidates:
                        break
                    new_e = rng.choice(candidates)
                    
                    new_sel = [x for x in cur_sel if x != e] + [new_e]
                    new_sel.sort()
                    
                    new_sums = set()
                    for a in range(len(new_sel)):
                        for b in range(a, len(new_sel)):
                            new_sums.add(new_sel[a] + new_sel[b])
                    
                    # Check if valid Sidon
                    valid = len(new_sums) == len(new_sel) * (len(new_sel) + 1) // 2
                    if valid and _is_maximal(new_sel, new_sums):
                        selected = new_sel
                        sums = new_sums
                        improved = True
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_sel = list(selected)
                            best_sums = set(sums)
                        break
                if improved:
                    break
        
        if not improved:
            break

    return best_size, best_sel, best_sums
=======
def _local_search(selected, sums, rng, time_limit):
    """
    Local search to minimize size while maintaining maximality.
    Tries single removals, double removals, and systematic swaps.
    """
    best_size = len(selected)
    best_sel = list(selected)
    best_sums = set(sums)
    start = time.time()
    
    while time.time() - start < time_limit and time.time() < DEADLINE:
        improved = False
        cur_sel = list(selected)
        cur_sums = set(sums)
        
        # --- Single Removal Pass ---
        for i in range(len(cur_sel)):
            if time.time() > DEADLINE:
                return best_size, best_sel, best_sums
            
            e = cur_sel[i]
            new_sel = cur_sel[:i] + cur_sel[i+1:]
            
            removed_sums = {2 * e}
            for a in new_sel:
                removed_sums.add(e + a)
            
            new_sums = cur_sums - removed_sums
            
            if _is_maximal(new_sel, new_sums):
                selected = new_sel
                sums = new_sums
                improved = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sel = list(selected)
                    best_sums = set(sums)
                break
        
        if not improved:
            # --- Double Removal Pass ---
            for i in range(len(cur_sel)):
                if time.time() > DEADLINE:
                    return best_size, best_sel, best_sums
                for j in range(i + 1, len(cur_sel)):
                    new_sel = [x for k, x in enumerate(cur_sel) if k != i and k != j]
                    
                    new_sums = set()
                    for a in range(len(new_sel)):
                        for b in range(a, len(new_sel)):
                            new_sums.add(new_sel[a] + new_sel[b])
                    
                    if _is_maximal(new_sel, new_sums):
                        selected = new_sel
                        sums = new_sums
                        improved = True
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_sel = list(selected)
                            best_sums = set(sums)
                        break
                if improved:
                    break
        
        if not improved:
            # --- Swap Pass: try replacing one element with each missing element ---
            sset = set(cur_sel)
            missing = [x for x in range(1, N + 1) if x not in sset]
            rng.shuffle(missing)
            
            for i in range(len(cur_sel)):
                if time.time() > DEADLINE:
                    return best_size, best_sel, best_sums
                e = cur_sel[i]
                
                # Try replacing e with each missing element
                for new_e in missing:
                    if time.time() > DEADLINE:
                        return best_size, best_sel, best_sums
                    
                    new_sel = [x for x in cur_sel if x != e] + [new_e]
                    new_sel.sort()
                    
                    new_sums = set()
                    for a in range(len(new_sel)):
                        for b in range(a, len(new_sel)):
                            new_sums.add(new_sel[a] + new_sel[b])
                    
                    # Check if valid Sidon
                    valid = len(new_sums) == len(new_sel) * (len(new_sel) + 1) // 2
                    if valid and _is_maximal(new_sel, new_sums):
                        selected = new_sel
                        sums = new_sums
                        improved = True
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_sel = list(selected)
                            best_sums = set(sums)
                        break
                if improved:
                    break
        
        if not improved:
            break

    return best_size, best_sel, best_sums
>>>>>>> REPLACE

