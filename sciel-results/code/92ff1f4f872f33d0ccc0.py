"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math

def _sumset_size(A):
    s = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s.add(A[i] + A[j])
    return len(s)

def _prodset_size(A):
    s = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s.add(A[i] * A[j])
    return len(s)

def _objective(A):
    return max(_sumset_size(A), _prodset_size(A))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    random.seed(42)
    
    # Initial candidate: try a few structured sets
    # 1) Consecutive integers
    best_A = list(range(1, 21))
    best_obj = _objective(best_A)
    
    # 2) Powers of 2 (geometric)
    cand2 = [2**i for i in range(20)]
    if max(cand2) <= 10000:
        o2 = _objective(cand2)
        if o2 < best_obj:
            best_obj = o2
            best_A = cand2
    
    # 3) Geometric with ratio 2 starting from 1: 1,2,4,...,2^19
    # 2^19 = 524288 > 10000, so use smaller range
    # Let's try 1, 2, 4, 8, ..., up to 20 terms but cap at 10000
    # 2^13 = 8192, 2^14 = 16384 > 10000
    # So we can have at most 14 powers of 2. Not enough for 20 distinct.
    # Try mixed: some powers of 2, some odd multipliers
    
    # 4) Try 5-smooth numbers: 2^a * 3^b * 5^c
    smooths = set()
    for a in range(14):  # 2^13 = 8192
        for b in range(8):  # 3^7 = 2187
            for c in range(5):  # 5^4 = 625
                v = (2**a) * (3**b) * (5**c)
                if 1 <= v <= 10000:
                    smooths.add(v)
    smooths = sorted(smooths)
    
    # Greedy: pick 20 5-smooth numbers to minimize objective
    # Start with a random subset of 20 from smooths
    if len(smooths) >= 20:
        # Try a few random subsets
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            subset = random.sample(smooths, 20)
            subset.sort()
            obj = _objective(subset)
            if obj < best_obj:
                best_obj = obj
                best_A = subset
    
    # Local search: swap elements
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        A = list(best_A)
        A.sort()
        for i in range(len(A)):
            for j in range(len(A)):
                if i == j:
                    continue
                # Try swapping A[i] with a random candidate from smooths
                if time.time() > deadline - 5:
                    break
                for _ in range(20):
                    new_val = random.choice(smooths)
                    if new_val in A:
                        continue
                    old_val = A[i]
                    A[i] = new_val
                    A.sort()
                    # Check distinctness
                    if len(set(A)) != 20:
                        A[i] = old_val
                        continue
                    obj = _objective(A)
                    if obj < best_obj:
                        best_obj = obj
                        best_A = list(A)
                        improved = True
                    A[i] = old_val
                    A.sort()
    
    # Try a few more structured approaches
    # Geometric progression with non-integer ratio? No, must be integers.
    # Try: 1, 2, 3, 5, 8, 13, 21, ... (Fibonacci-like) but capped
    fibs = [1, 2]
    while len(fibs) < 20:
        nxt = fibs[-1] + fibs[-2]
        if nxt > 10000:
            break
        fibs.append(nxt)
    if len(fibs) == 20:
        obj = _objective(fibs)
        if obj < best_obj:
            best_obj = obj
            best_A = fibs
    
    # Try: 1, 2, 4, 8, ... but with some variations
    # Try: 1, 3, 9, 27, ... (powers of 3)
    cand3 = [3**i for i in range(20)]
    if max(cand3) <= 10000:
        o3 = _objective(cand3)
        if o3 < best_obj:
            best_obj = o3
            best_A = cand3
    
    # 3^13 = 1594323 > 10000, so at most 8 powers of 3.
    # 3^8 = 6561, 3^9 = 19683 > 10000. So 9 terms max.
    
    # Try mixed geometric: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    # That's 14 terms. Need 6 more. Add odd numbers?
    
    # Final: return best found
    return best_A
# EVOLVE-BLOCK-END
