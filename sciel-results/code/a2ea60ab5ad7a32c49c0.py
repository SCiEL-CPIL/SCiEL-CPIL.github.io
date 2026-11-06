import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    limit = 512
    
    # Helper to check if x can be added to S
    def can_add(x, S):
        for y in S:
            if x * y // gcd(x, y) > limit:
                return False
        return True

    # Initial set: 1..23 is valid
    S = list(range(1, 24))
    
    # Verify initial set (should be valid)
    # for i in range(len(S)):
    #   for j in range(i+1, len(S)):
    #     if S[i] * S[j] // gcd(S[i], S[j]) > limit:
    #       print("Initial set invalid")
    #       break

    best_S = list(S)
    
    def greedy_fill(S):
        """Add all possible elements to S in increasing order"""
        S = list(S)
        changed = True
        while changed:
            changed = False
            for x in range(1, limit + 1):
                if x not in S and can_add(x, S):
                    S.append(x)
                    changed = True
        return S

    # Main Loop
    improved = True
    while improved:
        if time.time() > deadline:
            break
            
        improved = False
        
        # 1. Try to add
        new_S = greedy_fill(S)
        if len(new_S) > len(S):
            S = new_S
            improved = True
            if len(S) > len(best_S):
                best_S = list(S)
            continue
        
        # 2. Try to swap
        # For each y in S, remove it and try to fill
        best_swap_S = None
        best_swap_size = len(S)
        
        for y in S:
            if time.time() > deadline:
                break
            S_temp = [z for z in S if z != y]
            S_filled = greedy_fill(S_temp)
            if len(S_filled) > best_swap_size:
                best_swap_size = len(S_filled)
                best_swap_S = S_filled
        
        if best_swap_S is not None:
            S = best_swap_S
            improved = True
            if len(S) > len(best_S):
                best_S = list(S)
            continue
            
    # Final check
    if len(best_S) > len(S):
        return best_S
    return S
