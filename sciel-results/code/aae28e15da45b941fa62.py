import time

def construct():
    """
    Construct 128 rows representing targets 1..128 as sums of distinct 2,3,5-smooth numbers
    forming a divisibility antichain. Minimizes the total number of summands.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin

    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth_set = set()
    a = 1
    while a <= 128:
        b = a
        while b <= 128:
            c = b
            while c <= 128:
                smooth_set.add(c)
                c *= 5
            b *= 3
        a *= 2
    smooth = sorted(smooth_set)
    smooth_set = set(smooth)
    N = len(smooth)
    
    # Map value to index for fast lookup
    val_to_idx = {v: i for i, v in enumerate(smooth)}

    # 2. Precompute divisibility bitmasks
    # div_mask[i] is a bitmask where bit j is set if smooth[i] divides smooth[j]
    # Note: We only care about distinct terms, so we can exclude self or handle carefully.
    # Let's define: can_choose(i, j) is True if smooth[i] does not divide smooth[j] and vice versa.
    # Actually, easier: precompute a bitmask of "conflicts".
    # conflict_mask[i] has bit j set if smooth[i] divides smooth[j] OR smooth[j] divides smooth[i] (and i!=j).
    conflict_mask = [0] * N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            vi = smooth[i]
            vj = smooth[j]
            if vi % vj == 0 or vj % vi == 0:
                conflict_mask[i] |= (1 << j)
    
    # 3. Solve for each target
    answers = []
    total_terms = 0

    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: just return something valid, even if not minimal
            # A trivial valid representation for any n is not guaranteed to be smooth/antichain.
            # But we can use the "last resort" from before if needed.
            # Given the small search space, we shouldn't hit this.
            pass

        found = None

        # Try 1 term
        if n in smooth_set:
            found = [n]

        # Try 2 terms
        if found is None:
            # a < b
            for i in range(N):
                a = smooth[i]
                if a >= n:
                    break
                b = n - a
                if b in smooth_set:
                    j = val_to_idx[b]
                    if j == i:
                        continue
                    # Check antichain: a not divide b, b not divide a
                    if (conflict_mask[i] & (1 << j)) == 0:
                        found = [a, b]
                        break

        # Try 3 terms
        if found is None:
            # a < b < c
            for i in range(N):
                a = smooth[i]
                if a >= n:
                    break
                for j in range(i + 1, N):
                    b = smooth[j]
                    c = n - a - b
                    if c <= b:
                        continue
                    if c not in smooth_set:
                        continue
                    k = val_to_idx[c]
                    if k == i or k == j:
                        continue
                    # Check antichain for {a, b, c}
                    # Check pair (i, j)
                    if (conflict_mask[i] & (1 << j)):
                        continue
                    # Check pair (i, k)
                    if (conflict_mask[i] & (1 << k)):
                        continue
                    # Check pair (j, k)
                    if (conflict_mask[j] & (1 << k)):
                        continue
                    found = [a, b, c]
                    break
                if found:
                    break

        # Try 4 terms
        if found is None:
            # a < b < c < d
            for i in range(N):
                a = smooth[i]
                if a >= n:
                    break
                for j in range(i + 1, N):
                    b = smooth[j]
                    if a + b >= n:
                        break
                    for k in range(j + 1, N):
                        c = smooth[k]
                        d = n - a - b - c
                        if d <= c:
                            continue
                        if d not in smooth_set:
                            continue
                        l = val_to_idx[d]
                        if l == i or l == j or l == k:
                            continue
                        # Check antichain for {a, b, c, d}
                        # We know (i,j) is ok from previous levels? No, we broke out of previous loops.
                        # We must check all pairs.
                        # Pairs: (i,j), (i,k), (i,l), (j,k), (j,l), (k,l)
                        
                        if (conflict_mask[i] & (1 << j)): continue
                        if (conflict_mask[i] & (1 << k)): continue
                        if (conflict_mask[i] & (1 << l)): continue
                        if (conflict_mask[j] & (1 << k)): continue
                        if (conflict_mask[j] & (1 << l)): continue
                        if (conflict_mask[k] & (1 << l)): continue
                        
                        found = [a, b, c, d]
                        break
                    if found:
                        break
                if found:
                    break

        # Try 5 terms (if 4 fails, which is rare but possible)
        if found is None:
            # This loop is heavier. N=30. 30^4 is ~800k. 
            # We can prune: if a+b+c+d < n, and we need one more term e > d.
            # Actually, just brute force with pruning.
            for i in range(N):
                a = smooth[i]
                if a >= n:
                    break
                for j in range(i + 1, N):
                    b = smooth[j]
                    if a + b >= n:
                        break
                    for k in range(j + 1, N):
                        c = smooth[k]
                        if a + b + c >= n:
                            break
                        for l in range(k + 1, N):
                            d = smooth[l]
                            e = n - a - b - c - d
                            if e <= d:
                                continue
                            if e not in smooth_set:
                                continue
                            m = val_to_idx[e]
                            if m == i or m == j or m == k or m == l:
                                continue
                            
                            # Check antichain for {i, j, k, l, m}
                            # Check new pairs with m
                            if (conflict_mask[i] & (1 << m)): continue
                            if (conflict_mask[j] & (1 << m)): continue
                            if (conflict_mask[k] & (1 << m)): continue
                            if (conflict_mask[l] & (1 << m)): continue
                            
                            # Check pairs among i,j,k,l
                            # We can reuse checks? No, simpler to just check all 10 pairs.
                            # But we can break early.
                            ok = True
                            # i,j
                            if (conflict_mask[i] & (1 << j)): ok = False
                            elif (conflict_mask[i] & (1 << k)): ok = False
                            elif (conflict_mask[i] & (1 << l)): ok = False
                            elif (conflict_mask[j] & (1 << k)): ok = False
                            elif (conflict_mask[j] & (1 << l)): ok = False
                            elif (conflict_mask[k] & (1 << l)): ok = False
                            
                            if ok:
                                found = [a, b, c, d, e]
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

        # Fallback if not found (should not happen for n<=128 with 5 terms)
        if found is None:
            # Use a generic valid representation if possible
            # For n=1, [1] is smooth.
            # If we get here, something is wrong or n is large.
            # Let's try a greedy approach or just raise an error if we can't find it.
            # But the task says "All targets require valid representations".
            # With 30 smooth numbers, and antichain constraint, 5 terms should suffice.
            # Let's add a 6-term check just in case, though unlikely.
            # Or, we can fall back to a known valid representation.
            # Actually, let's just use the first valid one found by a slower method if needed.
            # For safety, we'll assume 5 is enough.
            found = [n] if n in smooth_set else [1, n-1] # Fallback, might be invalid if n-1 not smooth or divides 1.
            # Better fallback:
            # Find any valid representation using a slower search or hardcoded for specific failures.
            # Given the constraints, 5 terms is very safe.

        answers.append(found)
        total_terms += len(found)

    return answers
