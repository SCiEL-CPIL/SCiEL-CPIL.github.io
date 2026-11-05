import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 12
    
    # Greedy construction using difference sets
    # S is the set of all subset sums
    # D is the set of all positive differences between elements in S
    # We pick the smallest positive integer x not in D
    
    S = {0}
    D = set()
    seq = []
    
    for i in range(n):
        if time.time() > deadline:
            # Fallback to powers of 2
            return [1 << j for j in range(n)]
            
        if i == 0:
            x = 1
        else:
            # Find smallest x > 0 such that x not in D
            # And x not in seq (distinctness)
            # Note: if x not in D, it implies x is not a difference.
            # Does it imply x is not in seq?
            # If x in seq, then x is a subset sum.
            # If x in S, and 0 in S, then x - 0 = x is in D.
            # So if x not in D, x cannot be in S.
            # Since seq elements are in S, x cannot be in seq.
            # So we just need x not in D.
            
            x = 1
            while x in D:
                x += 1
                # Optimization: if x is large, we might be stuck.
                # But D is sparse for small x? No, D is dense.
                # For powers of 2, D = {1, ..., 2^k - 1}.
                # So x will be 2^k.
        
        seq.append(x)
        
        # Update S and D
        # New S = S union (S + x)
        new_S = set(S)
        for s in S:
            new_S.add(s + x)
        
        # New D
        # D_new = D_old union {s2 - s1 for s1, s2 in S_new, s1 < s2}
        # We can update D incrementally.
        # D_new = D_old 
        #           union {s + x - s' for s, s' in S, s + x > s'}
        #           union {s - (s' + x) for s, s' in S, s > s' + x}
        #           union {x} (since x - 0 = x, and x > 0)
        # Actually, simpler to recompute D if S is small enough.
        # For i=11, |S| = 2048. |D| ~ 2M.
        # Recomputing D from scratch: O(|S|^2) = O(2048^2) ~ 4M operations.
        # 12 steps * 4M = 48M operations.
        # Python might be slow for this. 50s limit.
        # 48M simple ops in Python might take ~5-10s. Should be fine.
        
        S = new_S
        
        # Recompute D
        D = set()
        ss = list(S)
        for i_s in range(len(ss)):
            for j_s in range(i_s + 1, len(ss)):
                d = ss[j_s] - ss[i_s]
                if d > 0:
                    D.add(d)
                    
    return seq
