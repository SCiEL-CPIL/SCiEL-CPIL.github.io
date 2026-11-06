def construct():
    from math import gcd
    import random
    import time
    
    n = 128
    nums = list(range(1, n + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd(i, j)
            
    # Precompute conflict masks
    # conflict_mask[x][u] is a bitmask of y such that (x, u, y) is a forbidden triple
    conflict_mask = [[0] * (n + 1) for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if a == b:
                continue
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                    # (a, b, c) is forbidden
                    conflict_mask[a][b] |= (1 << c)
                    conflict_mask[b][a] |= (1 << c)
                    conflict_mask[c][a] |= (1 << b)
                    conflict_mask[c][b] |= (1 << a)
                    conflict_mask[a][c] |= (1 << b)
                    conflict_mask[b][c] |= (1 << a)

    def is_valid_addition(selected_mask, x):
        # Check if x can be added to selected_mask
        # For all u, v in selected_mask, (u, v, x) must not be forbidden
        # Equivalent: for all u in selected_mask, x not in conflict_mask[u][v] for any v?
        # No, conflict_mask[u][v] is y's forbidden with u,v.
        # So we need: for all pairs (u,v) in selected, x is not in conflict_mask[u][v].
        # This is O(K^2) where K is size of selected.
        # Faster:
        # For each u in selected, we can check if x is in conflict_mask[u][v] for any v in selected.
        # Actually, we can precompute for each x, the set of pairs (u,v) that are forbidden with x.
        # But we have conflict_mask[u][v] which is y's.
        # So x is forbidden with (u,v) if conflict_mask[u][v] has bit x.
        # So:
        temp = selected_mask
        while temp:
            u_bit = temp & -temp
            u = u_bit.bit_length() - 1
            # Check all v in selected
            temp2 = selected_mask
            while temp2:
                v_bit = temp2 & -temp2
                v = v_bit.bit_length() - 1
                if u != v and (conflict_mask[u][v] & (1 << x)):
                    return False
                temp2 ^= v_bit
            temp ^= u_bit
        return True

    def greedy_random():
        random.shuffle(nums)
        selected = set()
        for x in nums:
            if is_valid_addition(0, x): # Wait, selected is a set, not mask yet
                # Convert to mask for check?
                # Let's just use sets for greedy, masks for B&B
                pass
        # Let's stick to sets for the heuristic part
        selected = set()
        for x in nums:
            # Check compatibility
            ok = True
            for u in selected:
                for v in selected:
                    if u == v: continue
                    if conflict_mask[u][v] & (1 << x):
                        ok = False
                        break
                if not ok: break
            if ok:
                selected.add(x)
        return selected

    def local_search(selected):
        improved = True
        while improved:
            improved = False
            # Add
            for x in range(1, n + 1):
                if x in selected: continue
                ok = True
                for u in selected:
                    for v in selected:
                        if u == v: continue
                        if conflict_mask[u][v] & (1 << x):
                            ok = False
                            break
                    if not ok: break
                if ok:
                    selected.add(x)
                    improved = True
                    break
            if improved: continue
            # Swap?
            # Try removing one and adding one?
            # Or 2-add-1-remove?
            # Let's try 1-remove, 1-add first
            for x in list(selected):
                temp_sel = selected - {x}
                for y in range(1, n + 1):
                    if y in temp_sel: continue
                    ok = True
                    for u in temp_sel:
                        for v in temp_sel:
                            if u == v: continue
                            if conflict_mask[u][v] & (1 << y):
                                ok = False
                                break
                        if not ok: break
                    if ok:
                        selected = temp_sel | {y}
                        improved = True
                        break
                if improved: break
            if improved: continue
            # 2-add, 1-remove
            for x in list(selected):
                temp_sel = selected - {x}
                # Try to add two
                added = False
                for y in range(1, n + 1):
                    if y in temp_sel: continue
                    ok_y = True
                    for u in temp_sel:
                        for v in temp_sel:
                            if u == v: continue
                            if conflict_mask[u][v] & (1 << y):
                                ok_y = False
                                break
                        if not ok_y: break
                    if not ok_y: continue
                    
                    # Check if y can be added
                    # Now try z
                    temp_sel_with_y = temp_sel | {y}
                    for z in range(1, n + 1):
                        if z in temp_sel_with_y: continue
                        ok_z = True
                        for u in temp_sel_with_y:
                            for v in temp_sel_with_y:
                                if u == v: continue
                                if conflict_mask[u][v] & (1 << z):
                                    ok_z = False
                                    break
                            if not ok_z: break
                        if ok_z:
                            selected = temp_sel_with_y | {z}
                            improved = True
                            added = True
                            break
                    if added: break
                if improved: break
        return selected

    # Get initial best
    best_set = set()
    start_time = time.time()
    max_time = 55.0
    
    # Run several random restarts
    for _ in range(20):
        if time.time() - start_time > max_time - 10: break
        s = greedy_random()
        s = local_search(s)
        if len(s) > len(best_set):
            best_set = s
            
    best_size = len(best_set)
    best_mask = 0
    for x in best_set:
        best_mask |= (1 << x)
        
    # B&B
    timeout = False
    
    def solve(selected_mask, avail_mask):
        nonlocal best_size, best_mask, timeout
        if timeout:
            return
        if time.time() - start_time > max_time:
            timeout = True
            return
            
        size = selected_mask.bit_count()
        if size + avail_mask.bit_count() <= best_size:
            return
            
        if avail_mask == 0:
            if size > best_size:
                best_size = size
                best_mask = selected_mask
            return
            
        # Pick x
        # Heuristic: Pick x that minimizes new_avail size?
        # Just pick first
        x = (avail_mask & -avail_mask).bit_length() - 1
        
        # Branch 1: Include x
        new_selected = selected_mask | (1 << x)
        new_avail = avail_mask & ~(1 << x)
        
        temp_selected = selected_mask
        while temp_selected:
            u_bit = temp_selected & -temp_selected
            u = u_bit.bit_length() - 1
            new_avail &= ~conflict_mask[x][u]
            temp_selected ^= u_bit
            
        solve(new_selected, new_avail)
        if timeout: return
        
        # Branch 2: Exclude x
        new_avail = avail_mask & ~(1 << x)
        solve(selected_mask, new_avail)

    solve(0, (1 << (n + 1)) - 2)
    
    result = []
    mask = best_mask
    while mask:
        bit = mask & -mask
        x = bit.bit_length() - 1
        result.append(x)
        mask ^= bit
        
    if not result:
        return [1]
        
    return result
