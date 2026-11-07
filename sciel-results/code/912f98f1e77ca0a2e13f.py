import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    all_nums = list(range(2, 41))
    candidates = [n for n in all_nums if n not in primes]
    
    # Precompute exponent vectors for all numbers 2..40
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    p_list = primes
    num_primes = len(p_list)
    
    vecs = {}
    for n in all_nums:
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(p_list):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs[n] = tuple(v)
        
    def is_valid(S):
        # S is a list of ints
        # Generate all subset sums of vectors
        sums = set()
        # Start with 0
        sums.add(tuple([0]*num_primes))
        current_sums = [tuple([0]*num_primes)]
        
        for n in S:
            v = vecs[n]
            new_sums = []
            for s in current_sums:
                # Add v to s
                new_s = tuple(s[i] + v[i] for i in range(num_primes))
                if new_s in sums:
                    return False
                sums.add(new_s)
                new_sums.append(new_s)
            current_sums.extend(new_sums)
        return True

    best_set = primes[:]
    best_len = len(primes)
    
    # Greedy addition
    # Try adding candidates in order of "independence" or just random
    # Let's try sorted by value first, then random restarts
    
    # 1. Deterministic Greedy
    S = primes[:]
    for n in candidates:
        if is_valid(S + [n]):
            S.append(n)
            
    if len(S) > best_len:
        best_set = S
        best_len = len(S)
        
    # 2. Local Search / Randomized Restart
    # Try to improve by swapping
    # If time permits, try random shuffles of candidates
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Randomize candidate order
        random.shuffle(candidates)
        S = primes[:]
        for n in candidates:
            if is_valid(S + [n]):
                S.append(n)
                
        if len(S) > best_len:
            best_set = S
            best_len = len(S)
            if best_len >= 18: # Max possible?
                break
        
        # If we are stuck at a low number, try swapping?
        # For now, just restart with different order.
        
    return [int(x) for x in best_set]
