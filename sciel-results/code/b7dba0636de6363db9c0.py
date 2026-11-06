<<<<<<< SEARCH
    # Local Search: Try to shrink S
    # We try to remove a point p. If S\{p} is not maximal, we greedily add points to fix it.
    # If the resulting size is smaller, we accept.
    improved = True
    while improved:
        improved = False
        # Shuffle to break symmetry
        s_list = list(S)
        random.shuffle(s_list)
        
        for p in s_list:
            if p not in S:
                continue
            
            # Save state
            old_S = set(S)
            old_blocked = blocked_count[:]
            
            # Remove p
            remove_point(p)
            
            # Check if maximal
            unblocked = [q for q in range(NUM_POINTS) if is_unblocked(q)]
            
            if not unblocked:
                # S\{p} is maximal and independent. We reduced size by 1.
                # Accept
                improved = True
                break
            else:
                # S\{p} is not maximal. We must add points.
                # Greedily add points to restore maximality
                temp_S_size = len(S)
                # We want to see if we can restore maximality with < 1 added points?
                # Impossible, we need at least 1 if unblocked is not empty.
                # So size will be at least len(S) + 1.
                # Original size was len(S) + 1.
                # New size will be >= len(S) + 1.
                # So we cannot improve size by removing p if maximality requires adding back >= 1 point.
                # Wait.
                # If we remove p, size decreases by 1.
                # If we add k points, size increases by k.
                # Net change: k - 1.
                # We need k - 1 < 0 => k < 1 => k = 0.
                # So we ONLY accept removal if S\{p} is already maximal.
                # If S\{p} is not maximal, we MUST add points.
                # The minimum number of points to add is 1 (if one point blocks all others).
                # So size becomes (len(S)+1) - 1 + 1 = len(S)+1. No improvement.
                # Unless... adding one point allows us to remove another point later?
                # That's a 2-opt move.
                
                # For now, only accept if k=0.
                # Restore state
                S.clear()
                S.update(old_S)
                for i in range(NUM_POINTS):
                    blocked_count[i] = old_blocked[i]
                # We need to re-add p? No, S was restored to old_S which included p.
                # But blocked_count was restored.
                # We just need to continue.
                continue

    return [POINTS[q] for q in S]
=======
    # Local Search with Deletions and Swaps
    improved = True
    while improved:
        improved = False
        s_list = list(S)
        random.shuffle(s_list)
        
        for p in s_list:
            if p not in S:
                continue
            
            old_S = set(S)
            old_blocked = blocked_count[:]
            
            remove_point(p)
            
            unblocked = [q for q in range(NUM_POINTS) if is_unblocked(q)]
            
            if not unblocked:
                # Deletion accepted
                improved = True
                break
            
            # Try to find a swap: add one point q that blocks all unblocked points
            found_swap = False
            for q in unblocked:
                # Check if adding q makes the set maximal
                # q must block all r in unblocked, r != q
                valid_swap = True
                for r in unblocked:
                    if r == q:
                        continue
                    key = (q, r) if q < r else (r, q)
                    blockers = pair_to_qs.get(key, set())
                    # Check if any blocker is in current S (which is S\{p})
                    if not any(s in S for s in blockers):
                        valid_swap = False
                        break
                
                if valid_swap:
                    # Perform swap
                    add_point(q)
                    found_swap = True
                    improved = True # Restart loop to check for deletions
                    break
            
            if not found_swap:
                # Restore state
                S.clear()
                S.update(old_S)
                for i in range(NUM_POINTS):
                    blocked_count[i] = old_blocked[i]
                continue

    return [POINTS[q] for q in S]
>>>>>>> REPLACE
