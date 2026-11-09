<<<<<<< SEARCH
    # Phase 3: Local search - try to improve by swapping elements
    for _ in range(30):
        if time.time() > deadline - 2:
            break
        
        s_trial = set(best_set)
        improved = False
        
        # Try removing one element and adding a different one
        for rem in list(s_trial):
            if time.time() > deadline - 1:
                break
            s_trial.remove(rem)
            for add in range(1, 257):
                if add in s_trial or add == rem:
                    continue
                if can_add(add, s_trial):
                    s_trial.add(add)
                    if len(s_trial) > best_size:
                        best_size = len(s_trial)
                        best_set = set(s_trial)
                        improved = True
                    break
            s_trial.add(rem)
            if improved:
                break
    
    # Phase 4: Try to add more elements to the best set found
    for x in range(1, 257):
        if time.time() > deadline - 1:
            break
        if x not in best_set and can_add(x, best_set):
            best_set.add(x)
            best_size = len(best_set)
=======
    # Phase 3: Enhanced local search - try multiple swap strategies
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        
        s_trial = set(best_set)
        improved = False
        
        # Strategy 1: Try removing one element and adding a different one
        for rem in list(s_trial):
            if time.time() > deadline - 1:
                break
            s_trial.remove(rem)
            for add in range(1, 257):
                if add in s_trial or add == rem:
                    continue
                if can_add(add, s_trial):
                    s_trial.add(add)
                    if len(s_trial) > best_size:
                        best_size = len(s_trial)
                        best_set = set(s_trial)
                        improved = True
                    break
            s_trial.add(rem)
            if improved:
                break
        
        if not improved:
            # Strategy 2: Try removing two elements and adding two different ones
            s_trial = set(best_set)
            elems = sorted(s_trial)
            for i in range(len(elems)):
                if time.time() > deadline - 1:
                    break
                for j in range(i+1, len(elems)):
                    if time.time() > deadline - 1:
                        break
                    rem1, rem2 = elems[i], elems[j]
                    s_trial.remove(rem1)
                    s_trial.remove(rem2)
                    found = False
                    for add1 in range(1, 257):
                        if time.time() > deadline - 1:
                            break
                        if add1 in s_trial or add1 in (rem1, rem2):
                            continue
                        if not can_add(add1, s_trial):
                            continue
                        s_trial.add(add1)
                        for add2 in range(1, 257):
                            if time.time() > deadline - 1:
                                break
                            if add2 in s_trial or add2 in (rem1, rem2) or add2 == add1:
                                continue
                            if not can_add(add2, s_trial):
                                continue
                            s_trial.add(add2)
                            if len(s_trial) > best_size:
                                best_size = len(s_trial)
                                best_set = set(s_trial)
                                improved = True
                                found = True
                            break
                        if found:
                            break
                        s_trial.remove(add1)
                    if not found:
                        pass
                    s_trial.add(rem1)
                    s_trial.add(rem2)
                    if improved:
                        break
                if improved:
                    break

    # Phase 4: Try to add more elements to the best set found
    for x in range(1, 257):
        if time.time() > deadline - 1:
            break
        if x not in best_set and can_add(x, best_set):
            best_set.add(x)
            best_size = len(best_set)

    # Phase 5: Try different base thresholds and combine
    for base_start in [165, 160, 155, 150]:
        if time.time() > deadline - 3:
            break
        s_trial = set(range(base_start, 257))
        # First check if this base is valid
        if full_check(s_trial):
            # Try to add elements from 1 to base_start-1
            for x in range(base_start - 1, 0, -1):
                if time.time() > deadline - 2:
                    break
                if can_add(x, s_trial):
                    s_trial.add(x)
            if len(s_trial) > best_size:
                best_size = len(s_trial)
                best_set = set(s_trial)
>>>>>>> REPLACE
