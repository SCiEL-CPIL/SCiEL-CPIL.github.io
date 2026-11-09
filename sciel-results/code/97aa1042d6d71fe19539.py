def _try_delete(W, n):
    L = len(W)
    if L <= n:
        return W, False
    
    perm_count = {}
    for i in range(L - n + 1):
        w = tuple(W[i:i+n])
        perm_count[w] = perm_count.get(w, 0) + 1
    
    for i in range(1, L - 1):
        k_lo = max(0, i - n + 1)
        k_hi = min(L - n, i)
        
        # Count occurrences in affected old windows
        affected_old_counts = {}
        for k in range(k_lo, k_hi + 1):
            w = tuple(W[k:k+n])
            affected_old_counts[w] = affected_old_counts.get(w, 0) + 1
        
        # Build affected new windows
        new_L = L - 1
        affected_new = set()
        for k in range(max(0, i - n + 1), min(new_L - n + 1, i)):
            w = []
            for idx in range(k, k + n):
                if idx < i:
                    w.append(W[idx])
                else:
                    w.append(W[idx + 1])
            affected_new.add(tuple(w))
        
        valid = True
        for w, count_in_affected in affected_old_counts.items():
            if perm_count[w] == count_in_affected and w not in affected_new:
                valid = False
                break
        if not valid:
            continue
        
        new_W = W[:i] + W[i+1:]
        return new_W, True
    
    return W, False
