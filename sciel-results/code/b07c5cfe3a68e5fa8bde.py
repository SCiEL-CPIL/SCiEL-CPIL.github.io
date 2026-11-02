import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Try to find the largest Sidon set possible
    # Max size for N=50 is around 10-12
    for k in range(12, 0, -1):
        if time.time() > deadline:
            break
            
        # Try random restarts to find a Sidon set of size k
        found = False
        for _ in range(2000):
            if time.time() > deadline:
                break
            try:
                S = random.sample(range(N), k)
                if is_sidon(S):
                    # Found a valid Sidon set of size k
                    h = [0.0] * N
                    for idx in S:
                        h[idx] = 1.0
                    return h
            except Exception:
                pass
        
        # If not found for this k, try smaller k
        # (The loop continues to k-1)

    # Fallback: Greedy construction if random search failed or timed out
    S = []
    for x in range(N):
        if time.time() > deadline:
            break
        candidate = S + [x]
        if is_sidon(candidate):
            S.append(x)
            
    h = [0.0] * N
    for idx in S:
        h[idx] = 1.0
        
    return h
