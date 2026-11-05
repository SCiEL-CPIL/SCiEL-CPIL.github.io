<<<<<<< SEARCH
    # Hill climbing: try all single swaps (remove one, add one from pool)
    best_score = current_score
    best_set = set(current_set)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        best_swap_gain = 0
        best_remove = None
        best_add = None
        
        # Precompute neighbor counts for each point in current_set
        # deg_in[p] = number of neighbors of p in current_set
        deg_in = {}
        for p in current_set:
            deg_in[p] = 0
        for p in current_set:
            for q in current_set:
                if p != q and is_adjacent(p, q):
                    deg_in[p] += 1
        
        # For each candidate removal and addition, compute net gain
        for p_remove in current_set:
            # Edges lost when removing p_remove
            edges_lost = deg_in[p_remove]
            
            for p_add in pool:
                if p_add in current_set:
                    continue
                # Edges gained when adding p_add to (current_set - {p_remove})
                edges_gained = 0
                for q in current_set:
                    if q == p_remove:
                        continue
                    if is_adjacent(p_add, q):
                        edges_gained += 1
                
                net_gain = edges_gained - edges_lost
                if net_gain > best_swap_gain:
                    best_swap_gain = net_gain
                    best_remove = p_remove
                    best_add = p_add
        
        if best_remove is not None and best_add is not None:
            current_set.remove(best_remove)
            current_set.add(best_add)
            current_score += best_swap_gain
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
=======
    # Hill climbing: try all single swaps (remove one, add one from pool)
    best_score = current_score
    best_set = set(current_set)
    
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        best_swap_gain = 0
        best_remove = None
        best_add = None
        
        # Precompute neighbor counts for each point in current_set
        # deg_in[p] = number of neighbors of p in current_set
        deg_in = {}
        for p in current_set:
            deg_in[p] = 0
        for p in current_set:
            for q in current_set:
                if p != q and is_adjacent(p, q):
                    deg_in[p] += 1
        
        # For each candidate removal and addition, compute net gain
        for p_remove in current_set:
            # Edges lost when removing p_remove
            edges_lost = deg_in[p_remove]
            
            for p_add in pool:
                if p_add in current_set:
                    continue
                # Edges gained when adding p_add to (current_set - {p_remove})
                edges_gained = 0
                for q in current_set:
                    if q == p_remove:
                        continue
                    if is_adjacent(p_add, q):
                        edges_gained += 1
                
                net_gain = edges_gained - edges_lost
                if net_gain > best_swap_gain:
                    best_swap_gain = net_gain
                    best_remove = p_remove
                    best_add = p_add
        
        if best_remove is not None and best_add is not None:
            current_set.remove(best_remove)
            current_set.add(best_add)
            current_score += best_swap_gain
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
    
    # 2-swap local search: try removing two points and adding two different points
    if time.time() < deadline - 10:
        improved2 = True
        while improved2 and time.time() < deadline - 10:
            improved2 = False
            best_2swap_gain = 0
            best_remove1 = None
            best_remove2 = None
            best_add1 = None
            best_add2 = None
            
            # Precompute degrees
            deg_in = {}
            for p in current_set:
                deg_in[p] = 0
            for p in current_set:
                for q in current_set:
                    if p != q and is_adjacent(p, q):
                        deg_in[p] += 1
            
            # For each pair to remove, compute best pair to add
            remove_list = list(current_set)
            for i in range(len(remove_list)):
                for j in range(i + 1, len(remove_list)):
                    p1 = remove_list[i]
                    p2 = remove_list[j]
                    # Edges lost: deg(p1) + deg(p2) - 1 if p1,p2 adjacent
                    edges_lost = deg_in[p1] + deg_in[p2]
                    if is_adjacent(p1, p2):
                        edges_lost -= 1
                    
                    # Try all pairs from pool
                    pool_list = [p for p in pool if p not in current_set]
                    for ai in range(len(pool_list)):
                        for aj in range(ai + 1, len(pool_list)):
                            a1 = pool_list[ai]
                            a2 = pool_list[aj]
                            # Check a1 and a2 are not adjacent to each other in a way that causes issues
                            # Actually they just need to be valid additions
                            edges_gained = 0
                            for q in current_set:
                                if q != p1 and q != p2:
                                    if is_adjacent(a1, q):
                                        edges_gained += 1
                                    if is_adjacent(a2, q):
                                        edges_gained += 1
                            if is_adjacent(a1, a2):
                                edges_gained += 1
                            
                            net_gain = edges_gained - edges_lost
                            if net_gain > best_2swap_gain:
                                best_2swap_gain = net_gain
                                best_remove1 = p1
                                best_remove2 = p2
                                best_add1 = a1
                                best_add2 = a2
                    
                    if time.time() > deadline - 10:
                        break
                if time.time() > deadline - 10:
                    break
            
            if best_remove1 is not None and best_add1 is not None:
                current_set.remove(best_remove1)
                current_set.remove(best_remove2)
                current_set.add(best_add1)
                current_set.add(best_add2)
                current_score += best_2swap_gain
                improved2 = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
>>>>>>> REPLACE

