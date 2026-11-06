def construct():
    import time
    import numpy as np
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 58.0 # 2s safety margin

    N = 17
    M = 1 << N
    min_dist = 6

    # 1. Generate Linear Code
    # ... (same as before) ...
    
    code = set(initial_code)
    K = len(code)
    
    # 2. Precompute offsets
    offsets_list = []
    for r in range(0, min_dist): # 0 to 5
        for comb in combinations(range(N), r):
            mask = 0
            for pos in comb:
                mask |= (1 << pos)
            offsets_list.append(mask)
    offsets = np.array(offsets_list, dtype=np.uint32)
    
    # 3. Initialize blocked array
    blocked = np.zeros(M, dtype=np.int16)
    for w in code:
        blocked[w ^ offsets] += 1
        
    # Helper to update blocked
    def add_word(w):
        nonlocal blocked
        blocked[w ^ offsets] += 1
        
    def remove_word(w):
        nonlocal blocked
        blocked[w ^ offsets] -= 1
        
    # Main Loop
    no_progress = 0
    while time.time() < deadline:
        # Try to add
        zeros = np.where(blocked == 0)[0]
        if len(zeros) > 0:
            # Pick a random zero to avoid bias
            w = zeros[random.randint(0, len(zeros)-1)]
            code.add(int(w))
            add_word(int(w))
            no_progress = 0
            continue
            
        # No immediate add possible. Try repair.
        # Find words with small blocked count
        # We want to add a word w, so we need to clear its conflicts.
        # Let's pick a random word w to try to add.
        # To be efficient, pick w with min blocked count > 0?
        # Or just random?
        # Let's try to find the minimum blocked count.
        min_blocked_val = np.min(blocked)
        if min_blocked_val > 5: # Too dense, maybe kick
             # Perform a kick: random swap
             # Pick random c in code
             c = random.choice(list(code))
             # Find valid c' for c
             # blocked_temp = blocked - mask(c)
             # We need blocked_temp[c'] == 0
             # This is equivalent to: for all x in code\{c}, d(x, c') >= 6
             # And d(c, c') doesn't matter for validity of c' in C\{c}, but c' must not be in C.
             
             # Efficient way to find c':
             # Create a temporary view?
             # blocked - mask(c) is expensive to compute fully.
             # Instead, iterate through candidates?
             
             # Let's just try a few random swaps.
             swap_done = False
             for _ in range(10):
                 c = random.choice(list(code))
                 # Try to move c to a random c'
                 # We need c' such that d(x, c') >= 6 for all x in C\{c}
                 # This is hard to check quickly without blocked.
                 # Let's use the blocked array.
                 # We want to find c' where (blocked - mask(c))[c'] == 0.
                 # This is equivalent to: blocked[c'] == mask(c)[c']
                 # mask(c) is 1 for neighbors of c, 0 else.
                 # So we need c' such that blocked[c'] == (1 if d(c,c')<6 else 0).
                 
                 # This is a specific pattern.
                 # Let's just pick a random c' and check validity manually?
                 # Validity check is O(K).
                 # K=256.
                 # Let's try 10 random c's and 10 random c' for each.
                 
                 c_new = random.randint(0, M-1)
                 if c_new in code: continue
                 
                 # Check if c_new is valid in C \ {c}
                 valid = True
                 for x in code:
                     if x == c: continue
                     if bin(x ^ c_new).count('1') < 6:
                         valid = False
                         break
                 
                 if valid:
                     # Swap
                     code.remove(c)
                     remove_word(c)
                     code.add(c_new)
                     add_word(c_new)
                     swap_done = True
                     break
             
             if not swap_done:
                 no_progress += 1
                 if no_progress > 1000:
                     break
             continue

        # min_blocked_val is small (1, 2, etc)
        # Find indices with min blocked
        candidates = np.where(blocked == min_blocked_val)[0]
        w = candidates[random.randint(0, len(candidates)-1)]
        w = int(w)
        
        # Find conflicting codewords
        conflicts = []
        for c in code:
            if bin(c ^ w).count('1') < 6:
                conflicts.append(c)
        
        # Try to repair by moving one conflict
        repaired = False
        for c in conflicts:
            # Try to find c' for c
            # We need c' such that:
            # 1. c' not in code
            # 2. d(x, c') >= 6 for all x in code \ {c}
            # 3. d(w, c') >= 6 (so w can be added)
            
            # Condition 2 is: (blocked - mask(c))[c'] == 0
            # Condition 3 is: d(w, c') >= 6
            
            # Let's iterate through possible c'?
            # We can use the fact that c' must be a "hole" in C\{c}.
            # Holes are where blocked[c'] == mask(c)[c'].
            
            # Let's try a few random c' near c? Or random?
            # Let's try random c'.
            for _ in range(5):
                c_new = random.randint(0, M-1)
                if c_new in code: continue
                if c_new == w: continue # w is not in code yet
                
                # Check validity in C \ {c}
                valid_c = True
                for x in code:
                    if x == c: continue
                    if bin(x ^ c_new).count('1') < 6:
                        valid_c = False
                        break
                if not valid_c: continue
                
                # Check distance to w
                if bin(w ^ c_new).count('1') < 6:
                    continue
                
                # Success! Swap c -> c_new, then add w
                code.remove(c)
                remove_word(c)
                code.add(c_new)
                add_word(c_new)
                
                code.add(w)
                add_word(w)
                
                repaired = True
                no_progress = 0
                break
            
            if repaired:
                break
        
        if not repaired:
            no_progress += 1
            if no_progress > 5000:
                break

    return list(code)
