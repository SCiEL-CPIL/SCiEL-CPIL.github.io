import time
import bisect

def is_valid_with_new(S, x):
    """
    Check if adding x to sorted list S results in a valid set.
    """
    idx = bisect.bisect_left(S, x)
    
    # Case 1: x is a
    # b, c in S[idx:]
    for i in range(idx, len(S)):
        b = S[i]
        for j in range(i, len(S)):
            c = S[j]
            if (b + c) % x == 0:
                return False
    
    # Case 2: x is b
    # a in S[:idx], c in S[idx:]
    for i in range(idx):
        a = S[i]
        for j in range(idx, len(S)):
            c = S[j]
            if (x + c) % a == 0:
                return False
    
    # Case 3: x is c
    # a, b in S[:idx]
    for i in range(idx):
        a = S[i]
        for j in range(i + 1, idx):
            b = S[j]
            if (b + x) % a == 0:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # 1. Generate initial candidates
    S1 = []
    for x in range(1, 257):
        if is_valid_with_new(S1, x):
            S1.append(x)
            
    S2 = []
    for x in range(256, 0, -1):
        if is_valid_with_new(S2, x):
            S2.append(x)
    S2.sort()
    
    S3 = list(range(171, 257))
    
    candidates = [S1, S2, S3]
    best_S = max(candidates, key=len)
    
    # 2. Local Search
    current_S = best_S[:]
    current_S.sort()
    current_set = set(current_S)
    all_nums = set(range(1, 257))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Additions
        added = True
        while added and time.time() < deadline:
            added = False
            for y in sorted(all_nums - current_set):
                if is_valid_with_new(current_S, y):
                    current_S.append(y)
                    current_S.sort()
                    current_set.add(y)
                    improved = True
                    added = True
                    # Break to restart addition loop with new S
                    break
        
        if time.time() >= deadline:
            break
            
        # Pass 2: Swaps
        # Try to find a swap that might enable future additions
        # We iterate through all x in S, y not in S
        # To be efficient, we might just do a limited number of swaps per iteration
        swaps_made = False
        # Shuffle to avoid bias?
        # Let's just iterate
        for x in current_S:
            if time.time() >= deadline:
                break
            for y in sorted(all_nums - current_set):
                if x == y:
                    continue
                # Check if S' = (S - {x}) U {y} is valid
                # Construct new_S
                new_S = [v for v in current_S if v != x]
                if is_valid_with_new(new_S, y):
                    # Accept swap
                    new_S.append(y)
                    new_S.sort()
                    current_S = new_S
                    current_set.remove(x)
                    current_set.add(y)
                    improved = True
                    swaps_made = True
                    break
            if swaps_made:
                break
                
    return current_S
