import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    S = []
    pair_sums = set()
    triple_sums = set()
    
    # We want to maximize the size.
    # Greedy smallest-first is a strong heuristic.
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        
        k = len(S)
        
        # 1. Check new pair sums
        # New pairs: (x, x) and (x, s) for s in S
        valid = True
        
        # Check (x, x)
        if 2 * x in pair_sums:
            valid = False
        else:
            # Check (x, s) for s in S
            for s in S:
                if x + s in pair_sums:
                    valid = False
                    break
        
        if not valid:
            continue
        
        # 2. Check new triple sums
        # New triples: (x, s_i, s_j) for 0 <= i <= j < k
        # These are x + S[i] + S[j]
        for i in range(k):
            si = S[i]
            for j in range(i, k):
                sj = S[j]
                ts = x + si + sj
                if ts in triple_sums:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # 3. Add x
        S.append(x)
        
        # Update pair_sums
        pair_sums.add(2 * x)
        for s in S[:-1]:
            pair_sums.add(x + s)
            
        # Update triple_sums
        # Add x + S[i] + S[j] for i <= j < k (old size)
        for i in range(k):
            si = S[i]
            for j in range(i, k):
                sj = S[j]
                triple_sums.add(x + si + sj)
                
    return S
