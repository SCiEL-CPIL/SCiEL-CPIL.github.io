import numpy as np
from math import isqrt
import time

def construct():
    deadline = time.time() + 55.0
    
    # 1. Sieve of Eratosthenes
    N = 10000
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            is_prime[i*i:N+1:i] = False
    
    primes = np.array([i for i, v in enumerate(is_prime) if v], dtype=np.int32)
    P = len(primes)
    
    # Map prime value to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # DP Tables
    # BestLen[i][g]: max length ending at i with last gap exactly g
    # PrefixMax[i][g]: max length ending at i with last gap <= g
    # ArgMaxGap[i][g]: the gap g' <= g that achieved PrefixMax[i][g]
    # PrevIdx[i][g]: previous index in the sequence for the state (i, g)
    # PrevGap[i][g]: the gap g_prev at the previous index i
    
    # Initialize with 0
    BestLen = np.zeros((P, N + 1), dtype=np.int32)
    PrefixMax = np.zeros((P, N + 1), dtype=np.int32)
    ArgMaxGap = np.full((P, N + 1), -1, dtype=np.int32)
    PrevIdx = np.full((P, N + 1), -1, dtype=np.int32)
    PrevGap = np.full((P, N + 1), -1, dtype=np.int32)
    
    # Base case: sequence of length 1 ending at any prime i
    # We model this as having last gap 0.
    for i in range(P):
        BestLen[i, 0] = 1
        # PrefixMax[i][g] for g >= 0 will be at least 1
        # We will compute PrefixMax in the loop or initialize properly.
        # Let's just set BestLen[i,0]=1 and handle the rest in the main loop logic.
        
    # Main DP Loop
    for j in range(P):
        p_j = primes[j]
        
        # Iterate over all previous primes i
        for i in range(j):
            p_i = primes[i]
            g = p_j - p_i
            
            # g is the new gap
            # We can extend any sequence ending at i with last gap <= g
            # The best such length is PrefixMax[i][g]
            # If PrefixMax[i][g] > 0, we can extend it.
            # Note: PrefixMax[i][g] includes the base case length 1 (gap 0) if g >= 0.
            
            val = PrefixMax[i, g] + 1
            
            if val > BestLen[j, g]:
                BestLen[j, g] = val
                PrevIdx[j, g] = i
                # The previous gap is the one that achieved PrefixMax[i][g]
                PrevGap[j, g] = ArgMaxGap[i, g]
                
        # Compute PrefixMax and ArgMaxGap for row j
        # PrefixMax[j][g] = max(BestLen[j][g], PrefixMax[j][g-1])
        # ArgMaxGap[j][g] = g if BestLen[j][g] >= PrefixMax[j][g-1] else ArgMaxGap[j][g-1]
        
        # Initialize for g=0
        # BestLen[j, 0] might be 0 if j is not the first prime?
        # Wait, for j > 0, BestLen[j, 0] is 0 because no two distinct primes have gap 0.
        # But the base case BestLen[j, 0] = 1 was set?
        # No, I set BestLen[i, 0] = 1 for ALL i in the initialization loop above.
        # This is correct: any single prime is a valid sequence of length 1.
        # So BestLen[j, 0] is 1.
        
        PrefixMax[j, 0] = BestLen[j, 0]
        ArgMaxGap[j, 0] = 0 if BestLen[j, 0] > 0 else -1
        
        for g in range(1, N + 1):
            if BestLen[j, g] >= PrefixMax[j, g - 1]:
                PrefixMax[j, g] = BestLen[j, g]
                ArgMaxGap[j, g] = g
            else:
                PrefixMax[j, g] = PrefixMax[j, g - 1]
                ArgMaxGap[j, g] = ArgMaxGap[j, g - 1]
                
        # Check deadline
        if time.time() > deadline:
            break

    # Find the global maximum length
    best_len = 0
    best_j = -1
    best_g = -1
    
    for j in range(P):
        # The max length for prime j is PrefixMax[j][N]
        if PrefixMax[j, N] > best_len:
            best_len = PrefixMax[j, N]
            best_j = j
            best_g = N

    # Backtrack
    path = []
    curr_j = best_j
    curr_g = best_g
    
    while curr_j != -1:
        path.append(primes[curr_j])
        
        # If the gap is 0, it's the start (base case)
        if curr_g == 0:
            break
            
        prev_j = PrevIdx[curr_j, curr_g]
        prev_g = PrevGap[curr_j, curr_g]
        
        # Safety check to avoid infinite loops or invalid indices
        if prev_j == -1 or prev_g == -1:
            break
            
        curr_j = prev_j
        curr_g = prev_g
        
    path.reverse()
    
    return path.tolist()
