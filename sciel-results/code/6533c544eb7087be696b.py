"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Fast validity check for adding x to S
    # S is a list of unique integers
    def can_add(x, S):
        # 1. Check if x divides sum of any 3 distinct elements in S
        #    If |S| < 3, this is false
        if len(S) >= 3:
            # Iterate combinations of 3
            n = len(S)
            for i in range(n):
                si = S[i]
                for j in range(i+1, n):
                    sj = S[j]
                    for k in range(j+1, n):
                        sk = S[k]
                        if (si + sj + sk) % x == 0:
                            return False
        
        # 2. Check if any a in S divides sum of x + b + c
        #    where b, c are distinct elements in S \ {a}
        for a in S:
            # We need b, c in S such that b != a, c != a, b != c
            # And (x + b + c) % a == 0
            # Optimization: 
            # If a > x + max(S)*2, then x+b+c < a, so sum % a != 0 (unless sum==0, impossible)
            # Max sum approx 192*3 = 576.
            # If a > 576, always safe. But max S is 192.
            
            # Iterate b in S
            # We can precompute S without a? 
            # Just loop and skip if b==a
            for b in S:
                if b == a:
                    continue
                for c in S:
                    if c == a or c == b:
                        continue
                    if (x + b + c) % a == 0:
                        return False
        return True

    # Greedy Construction
    S = []
    candidates = list(range(192, 0, -1))
    
    for x in candidates:
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.append(x)
            
    # Local Search / Repair Phase
    # Try to increase size or swap to allow increases
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any missing number
        for v in range(1, 193):
            if time.time() > deadline:
                break
            if v not in S:
                if can_add(v, S):
                    S.append(v)
                    improved = True
                    break # Restart loop to try adding more
        
        if improved:
            continue
            
        # If no direct add, try swapping
        # Try to remove u and add v such that size remains same, 
        # hoping it opens up a spot for a third number later.
        # Or just try to find a valid swap to explore.
        # To be efficient, we only swap if it seems beneficial?
        # Hard to know. Let's just do random swaps or systematic swaps.
        # Systematic: for each u in S, for each v not in S, check if S\{u}+{v} is valid.
        # This is expensive. 
        # Let's limit the number of swap attempts per iteration.
        
        # Shuffle S to vary removal order?
        # Let's try a limited number of random swaps
        attempts = 0
        while attempts < 100 and time.time() < deadline:
            attempts += 1
            if not S: break
            u_idx = random.randint(0, len(S)-1)
            u = S[u_idx]
            
            # Pick a random v not in S
            # If S is large, most numbers are in S.
            # Let's just pick a random number from 1..192 and check if not in S
            v = random.randint(1, 192)
            if v in S:
                continue
                
            S_temp = S[:]
            S_temp.pop(u_idx)
            
            if can_add(v, S_temp):
                # Valid swap
                S = S_temp
                S.append(v)
                improved = True
                break # Restart main while loop to try adding more
        
    return S

# EVOLVE-BLOCK-END
