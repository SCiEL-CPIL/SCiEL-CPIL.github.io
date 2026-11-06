import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    # Primes p = 3 mod 4
    # 83, 79, 71, 67, 59, 47, 43, 31, 23, 19, 11, 7
    # We want to minimize n, so we try smaller n first?
    # But we need a VALID tournament.
    # If a smaller one is invalid, we fall back to a larger one.
    # We know 83 is valid.
    # We suspect 67 is valid.
    # Let's try 67, 59, 47, 43, 31, 23, 19, 11, 7 in that order?
    # Or just check 67 first. If it works, return it.
    # If it fails, check 59, etc.
    # If all fail, return 83.
    
    candidates = [67, 59, 47, 43, 31, 23, 19, 11, 7, 83]
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Generate Paley tournament
        squares = {x * x % p for x in range(1, p)}
        # Orientation list
        # i < j, 0 if i->j (j-i in squares), 1 if j->i
        # Note: In Paley, i->j iff (j-i) is a square.
        # So if (j-i) % p in squares, bit is 0.
        
        # Build adjacency matrix for fast checking
        # adj[i][j] = 1 if i->j else 0
        # We need to check if for all 4-sets {a,b,c,d}, there exists v such that v->a, v->b, v->c, v->d.
        # Using the D(a,b) bitmask method.
        
        n = p
        # mask[i][j] for i<j: bitmask of v such that v->i and v->j
        # We can store these in a dictionary or a 2D array of integers (bitmasks)
        # Since n <= 90, we can use Python integers as bitmasks.
        
        # Precompute out-neighbors for each vertex?
        # out[i] = set of j such that i->j
        # But we need in-neighbors for D(a,b)?
        # D(a,b) = {v | v->a and v->b} = In(a) \cap In(b)
        
        # Let's compute In(i) as a bitmask.
        # in_mask[i] has bit j set if j->i.
        
        in_masks = [0] * n
        orientation = [0] * (n * (n - 1) // 2)
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % p in squares:
                    # i -> j
                    orientation[idx] = 0
                    # j is in Out(i), i is in In(j)
                    in_masks[j] |= (1 << i)
                else:
                    # j -> i
                    orientation[idx] = 1
                    # i is in Out(j), j is in In(i)
                    in_masks[i] |= (1 << j)
                idx += 1
        
        # Precompute D(a,b) for all a < b
        # D(a,b) = in_masks[a] & in_masks[b]
        # We need to check if D(a,b) & D(c,d) is non-empty for 4-set {a,b,c,d}
        # Note: The intersection must not contain a, b, c, d.
        # But in_masks[a] does not have bit a set (no self loops).
        # Does in_masks[a] have bit b?
        # If b->a, then bit b is in in_masks[a].
        # So D(a,b) might contain b?
        # If b->a, then b is in In(a).
        # Is b in In(b)? No.
        # So b is NOT in D(a,b).
        # Similarly a is NOT in D(a,b).
        # So D(a,b) only contains vertices v \notin {a,b}.
        # Thus D(a,b) & D(c,d) will not contain a,b,c,d.
        
        # So we just check if (in_masks[a] & in_masks[b]) & (in_masks[c] & in_masks[d]) != 0
        
        # Iterate over all 4-sets
        valid = True
        for a in range(n):
            for b in range(a + 1, n):
                mask_ab = in_masks[a] & in_masks[b]
                if mask_ab == 0:
                    continue # No vertex dominates both a and b, so no vertex dominates any 4-set containing a,b?
                    # Wait. If no vertex dominates a and b, then no vertex can dominate a,b,c,d.
                    # So if mask_ab is 0, any 4-set containing a and b is invalid.
                    # We should check this earlier?
                    # Actually, if mask_ab == 0, then for any c,d, the 4-set {a,b,c,d} is invalid.
                    # So we can just fail immediately if any pair has empty common in-neighborhood?
                    # No, we need a common dominator for the 4-set.
                    # If In(a) \cap In(b) is empty, then no vertex points to both a and b.
                    # Therefore, no vertex points to a,b,c,d.
                    # So the tournament is invalid.
                    valid = False
                    break
            if not valid:
                break
            
            for c in range(b + 1, n):
                mask_bc = in_masks[b] & in_masks[c] # This is not D(b,c) in the sense of the pair for the 4-set partition?
                # For 4-set {a,b,c,d}, we used D(a,b) & D(c,d).
                # We can iterate c,d.
                for d in range(c + 1, n):
                    mask_cd = in_masks[c] & in_masks[d]
                    if (mask_ab & mask_cd) == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            if time.time() > deadline:
                break
        if not valid:
            continue
        
        # If we get here, it's valid
        return {'n': n, 'orientation': orientation}

    # Fallback to 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
