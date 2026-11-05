def _check_property_fast(n, orientation):
    in_neigh = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                in_neigh[j] |= (1 << i)
            else:
                in_neigh[i] |= (1 << j)
            idx += 1
            
    # Check all 4-sets
    # We can prune: if a 4-set {a,b,c,d} has no common dominator, fail.
    # Common dominator must be in intersection of in_neigh[a], in_neigh[b], in_neigh[c], in_neigh[d]
    # AND must not be in {a,b,c,d}.
    
    # Optimization: Pre-compute pairwise intersections?
    # No, 4-way intersection is necessary.
    
    # To speed up the loop:
    range_n = range(n)
    for a in range_n:
        ia = in_neigh[a]
        for b in range(a+1, n):
            ib = in_neigh[b]
            iab = ia & ib
            if iab == 0: continue # No common in-neighbor for a,b, so any set containing a,b fails? 
            # Wait, if iab is 0, then no vertex dominates both a and b. 
            # So any 4-set containing a and b is bad. 
            # We can return False immediately? 
            # Yes! If there exist a,b such that no vertex dominates both, then any 4-set containing a,b is a counter-example.
            # But wait, the dominator must be OUTSIDE the set.
            # If iab == 0, it means no vertex in V dominates both a and b.
            # So for any c,d, the set {a,b,c,d} has no common dominator (since no vertex dominates a and b).
            # So we can return False.
            
            # However, we must ensure the dominator is not in the set.
            # If iab has bits, we need to check if any of those bits is NOT in {a,b,c,d}.
            
            # Let's refine the early exit.
            # If (ia & ib) == 0, return False.
            
            for c in range(b+1, n):
                ic = in_neigh[c]
                iabc = iab & ic
                if iabc == 0: continue
                # Check if iabc has a bit outside {a,b,c}
                # Actually, we need to check for d too.
                for d in range(c+1, n):
                    id_ = in_neigh[d]
                    iabcd = iabc & id_
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    if (iaabcd & ~mask) == 0:
                        return False
    return True
