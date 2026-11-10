import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    # Precompute comparability bitsets
    # comp_bits[i] is a bitset where bit j is 1 if i and j are comparable (i != j)
    comp_bits = [0] * total_masks
    
    # Helper to check comparability
    # i and j are comparable if (i & j) == i or (i & j) == j
    # Since we iterate i, j, we can just check this.
    # Optimization: 
    # For each i, the sets comparable to i are:
    # 1. All subsets of i (excluding i itself)
    # 2. All supersets of i (excluding i itself)
    
    # We can generate these efficiently.
    # Subsets of i: iterate sub = i; sub = (sub - 1) & i ...
    # Supersets of i: iterate super = i; super = (super + 1) | i ... (not straightforward)
    # Alternative: Just loop j from 0 to total_masks-1. 4096 * 4096 / 2 is ~8M ops. 
    # In Python, this might take a few seconds. Let's try to optimize.
    
    # Faster precomputation:
    # For each mask i:
    #   comp_bits[i] = 0
    #   # Subsets
    #   sub = i
    #   while sub:
    #       comp_bits[i] |= (1 << sub)
    #       sub = (sub - 1) & i
    #   # Supersets
    #   # A superset s of i satisfies (s & i) == i.
    #   # This is equivalent to s = i | t where t is a subset of (~i) & (total_masks-1)
    #   free_bits = (total_masks - 1) ^ i
    #   t = free_bits
    #   while t:
    #       s = i | t
    #       if s != i: # exclude i itself, though t!=0 ensures s!=i if free_bits!=0? 
    #                  # if i == total_masks-1, free_bits=0, loop doesn't run.
    #       comp_bits[i] |= (1 << s)
    #       t = (t - 1) & free_bits
    #   # Remove i itself if it got added (it shouldn't if we are careful, but subsets loop includes 0? 
    #   # sub loop: sub=i, then (sub-1)&i. If i=0, sub=0, loop doesn't run.
    #   # So i is not in its own subset list (unless we handle 0 specially? No, 0 has no non-empty subsets. 
    #   # Wait, empty set is a subset. 
    #   # If i=0, sub=0. Loop `while sub` doesn't run. So 0 is not added as subset of 0.
    #   # But 0 is comparable to 0? No, strict subset.
    #   # So we don't want bit i set in comp_bits[i].
    #   # My loops:
    #   # Subsets: `while sub` skips 0. So 0 is never added as a subset of itself? 
    #   # Actually, we want ALL subsets. 
    #   # Standard trick: 
    #   # sub = i
    #   # while True:
    #   #   if sub != i: comp_bits[i] |= 1 << sub
    #   #   if sub == 0: break
    #   #   sub = (sub - 1) & i
    
    # Let's use the explicit loop for clarity and correctness first, then optimize if needed.
    # 8 million iterations in Python is ~1-2 seconds. Acceptable.
    
    for i in range(total_masks):
        bits = 0
        # Subsets of i (strict)
        sub = (i - 1) & i
        while sub:
            bits |= (1 << sub)
            sub = (sub - 1) & i
        if i > 0:
            bits |= (1 << 0) # 0 is subset of all non-zero i. 
            # Wait, if i=0, no subsets.
            # If i>0, 0 is a subset.
            # My loop `sub = (i-1)&i` starts from the largest proper subset.
            # It will eventually hit 0? 
            # Example i=1 (01). sub=0. Loop `while sub` stops. So 0 is not added.
            # So we must add 0 explicitly if i!=0.
            
        # Supersets of i (strict)
        # s is a superset of i if (s & i) == i and s != i.
        # Let free = (total_masks - 1) ^ i.
        # s = i | t where t is a non-empty subset of free.
        free = (total_masks - 1) ^ i
        t = free
        while t:
            s = i | t
            bits |= (1 << s)
            t = (t - 1) & free
            
        comp_bits[i] = bits

    # Helper: popcount
    def popcount(x):
        return x.bit_count()

    # Initialize S
    # Start with a greedy chain + random fill?
    # Let's try: Pick 40 sets that are "central" or form a good structure.
    # Simple start: First 40 masks? No, they are small.
    # Better: Random 40 distinct masks.
    # Or: Greedy. Start with empty set. Add mask with max comparability to current set.
    
    S = []
    S_set = set()
    S_bitset = 0
    
    # Greedy Initialization
    # Start with 0
    S.append(0)
    S_set.add(0)
    S_bitset |= (1 << 0)
    
    while len(S) < m:
        if time.time() > deadline - 10:
             # Fill rest randomly if running out of time
             while len(S) < m:
                 x = random.randint(0, total_masks - 1)
                 if x not in S_set:
                     S.append(x)
                     S_set.add(x)
                     S_bitset |= (1 << x)
             break
        
        best_x = -1
        best_score = -1
        # To speed up, we can sample candidates? No, 4096 is small.
        for x in range(total_masks):
            if x in S_set:
                continue
            # Score: number of comparable elements in S
            # count = popcount(comp_bits[x] & S_bitset)
            # This is O(1) with bit_count
            score = popcount(comp_bits[x] & S_bitset)
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x == -1:
             # Should not happen
             break
             
        S.append(best_x)
        S_set.add(best_x)
        S_bitset |= (1 << best_x)
        
    # Calculate initial score
    def calc_score(S_list, S_bits):
        # Sum of comparable pairs
        # Each pair counted twice in sum of degrees?
        # deg[x] = popcount(comp_bits[x] & S_bits)
        # Total pairs = sum(deg[x] for x in S_list) / 2
        s = 0
        for x in S_list:
            s += popcount(comp_bits[x] & S_bits)
        return s // 2

    current_score = calc_score(S, S_bitset)
    best_score = current_score
    best_S = S.copy()
    best_S_set = S_set.copy()
    best_S_bitset = S_bitset
    
    # Simulated Annealing
    T = 10.0
    min_T = 0.01
    cooling_rate = 0.995
    steps = 0
    
    while time.time() < deadline - 2:
        if T < min_T:
            T = min_T
            
        # Pick a random element to remove
        idx = random.randint(0, len(S) - 1)
        A = S[idx]
        
        # Pick a random element to add
        # To ensure B != A and B not in S
        # We can pick random B from [0, total_masks) and check
        # Or maintain a list of candidates? 
        # Just picking random is fine, rejection rate is 40/4096 ~ 1%
        B = random.randint(0, total_masks - 1)
        if B == A or B in S_set:
            continue
            
        # Calculate Delta
        # Remove A: lose pairs (A, y) for y in S, y != A
        # Add B: gain pairs (B, y) for y in S, y != A (note: A is removed, so y != A is naturally handled if we use S \ {A})
        
        # S_without_A_bitset = S_bitset & ~(1 << A)
        S_wo_A = S_bitset & ~(1 << A)
        
        # Pairs lost: comparable(A, y) for y in S_wo_A
        lost = popcount(comp_bits[A] & S_wo_A)
        
        # Pairs gained: comparable(B, y) for y in S_wo_A
        gained = popcount(comp_bits[B] & S_wo_A)
        
        delta = gained - lost
        
        if delta > 0:
            accept = True
        else:
            # Accept with probability exp(delta / T)
            # If delta is very negative, exp is 0
            if delta < -100 * T: # Optimization: if too bad, skip
                accept = False
            else:
                accept = random.random() < (2.718281828 ** (delta / T))
                
        if accept:
            # Perform swap
            S[idx] = B
            S_set.remove(A)
            S_set.add(B)
            S_bitset = S_wo_A | (1 << B)
            
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_S = S.copy()
                best_S_set = S_set.copy()
                best_S_bitset = S_bitset
                
        T *= cooling_rate
        steps += 1
        if steps % 10000 == 0:
            # print(f"Steps: {steps}, Score: {current_score}, Best: {best_score}, T: {T:.4f}")
            pass

    # Convert best_S to lists of integers
    result = []
    for mask in best_S:
        lst = []
        for b in range(n):
            if (mask >> b) & 1:
                lst.append(b)
        result.append(lst)
        
    return result
