import math
import random
import time

def construct():
    deadline = time.time() + 115.0
    N = 511
    C = 512
    
    # Sieve for primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Weights
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)
        
    # Precompute prime factors for each number
    # pf[a] = list of distinct prime factors of a
    pf = [[] for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            pf[multiple].append(p)
            
    # Conflict check: a and b conflict if they share a prime factor
    # We can use a bitmask for primes if number of primes is small
    # There are 97 primes. 97 bits is too large for a single int in Python?
    # Python ints are arbitrary precision, so it's fine.
    p_index = {p: i for i, p in enumerate(primes)}
    mask = [0] * (N + 1)
    for a in range(1, N + 1):
        m = 0
        for p in pf[a]:
            m |= (1 << p_index[p])
        mask[a] = m
        
    def is_coprime(a, b):
        return (mask[a] & mask[b]) == 0
        
    # Initial Solution:
    # 1. Include 1
    # 2. Include all primes p > 256
    # 3. Include p^k for all primes p <= 256
    S = {1}
    S_mask = 1 # mask[1] is 0, but we track S_mask as OR of masks of elements
    # Actually, let's maintain S as a set and a combined mask?
    # No, combined mask doesn't work for checking "can I add x?"
    # Because x conflicts with S if mask[x] & (OR of masks in S) != 0?
    # No. x conflicts with S if there EXISTS y in S such that mask[x] & mask[y] != 0.
    # This is NOT equivalent to mask[x] & (OR mask) != 0.
    # Example: S={2, 3}. OR mask = 2|3. x=6. mask[6]=2|3.
    # mask[6] & (2|3) != 0. Conflict. Correct.
    # Example: S={2, 3}. x=5. mask[5]=5. 5 & (2|3) = 0. No conflict. Correct.
    # Wait. If S={2, 3}, and x=6.
    # 6 conflicts with 2 AND 6 conflicts with 3.
    # So 6 cannot be added.
    # The condition "x is coprime to all y in S" is equivalent to
    # "for all y in S, mask[x] & mask[y] == 0".
    # This is NOT equivalent to mask[x] & (OR S) == 0.
    # Counter example: S={2, 3}. OR=3 (binary 11). x=6 (binary 110? No, 2 is bit 0, 3 is bit 1? No, primes are indexed).
    # Let's stick to checking against each element or using a more complex structure.
    # Given N=511, |S| is small (< 100). Checking each element is fast.
    
    S = set()
    
    # Add 1
    S.add(1)
    
    # Add large primes
    for p in primes:
        if p > 256:
            S.add(p)
            
    # Add prime powers for small primes
    for p in primes:
        if p <= 256:
            k = 1
            val = p
            while val * p <= N:
                val *= p
                k += 1
            S.add(val)
            
    # Polish: Add any remaining number that is coprime to all in S
    # Iterate multiple times to ensure maximal packing
    changed = True
    while changed:
        changed = False
        # Sort by weight descending to add best ones first
        candidates = [a for a in range(1, N + 1) if a not in S]
        candidates.sort(key=lambda a: -w[a])
        for a in candidates:
            if all(is_coprime(a, b) for b in S):
                S.add(a)
                changed = True
                
    current_score = sum(w[a] for a in S)
    
    # Local Search
    rng = random.Random(42)
    T = 0.5
    T_min = 1e-6
    cooling = 0.999
    
    # Precompute list of all numbers
    all_nums = list(range(1, N + 1))
    
    def get_score(S_set):
        return sum(w[a] for a in S_set)
        
    best_S = S.copy()
    best_score = current_score
    
    while time.time() < deadline - 5.0:
        if T < T_min:
            break
            
        move_type = rng.random()
        new_S = S.copy()
        delta = 0.0
        
        if move_type < 0.3:
            # Swap: Remove one, Add one
            if S:
                x = rng.choice(list(S))
                new_S.discard(x)
                # Find best candidate to add
                best_add = None
                best_add_w = -1
                # Sample a few candidates or iterate all?
                # Iterate all is 511 ops. Fast.
                for a in all_nums:
                    if a in new_S:
                        continue
                    if all(is_coprime(a, b) for b in new_S):
                        if w[a] > best_add_w:
                            best_add_w = w[a]
                            best_add = a
                if best_add is not None:
                    new_S.add(best_add)
                    delta = w[best_add] - w[x]
                else:
                    delta = -w[x] # Just remove? No, swap implies add. If no add, skip?
                    # Let's treat this as a Remove move if no add found
                    new_S = S.copy() # Revert
                    delta = 0.0
                    # Actually, let's just do a pure remove in a separate branch?
                    # Or accept the removal?
                    # Let's just skip if no valid add.
                    continue 
        elif move_type < 0.5:
            # Merge: Remove two, Add one
            if len(S) >= 2:
                x, y = rng.sample(list(S), 2)
                new_S.discard(x)
                new_S.discard(y)
                # Find best candidate to add that is coprime to new_S
                best_add = None
                best_add_w = -1
                for a in all_nums:
                    if a in new_S:
                        continue
                    if all(is_coprime(a, b) for b in new_S):
                        if w[a] > best_add_w:
                            best_add_w = w[a]
                            best_add = a
                if best_add is not None:
                    new_S.add(best_add)
                    delta = w[best_add] - w[x] - w[y]
                else:
                    continue
        elif move_type < 0.7:
            # Split: Remove one, Add two
            if S:
                x = rng.choice(list(S))
                new_S.discard(x)
                # Find best pair to add
                # This is expensive. Sample pairs?
                # Or just add best single, then best single again?
                # Let's try adding best single, then best single from remaining.
                best_add1 = None
                best_add1_w = -1
                for a in all_nums:
                    if a in new_S:
                        continue
                    if all(is_coprime(a, b) for b in new_S):
                        if w[a] > best_add1_w:
                            best_add1_w = w[a]
                            best_add1 = a
                            
                if best_add1 is not None:
                    temp_S = new_S.copy()
                    temp_S.add(best_add1)
                    best_add2 = None
                    best_add2_w = -1
                    for a in all_nums:
                        if a in temp_S:
                            continue
                        if all(is_coprime(a, b) for b in temp_S):
                            if w[a] > best_add2_w:
                                best_add2_w = w[a]
                                best_add2 = a
                    if best_add2 is not None:
                        new_S.add(best_add1)
                        new_S.add(best_add2)
                        delta = w[best_add1] + w[best_add2] - w[x]
                    else:
                        # Only added one
                        new_S.add(best_add1)
                        delta = w[best_add1] - w[x]
                else:
                    continue
        else:
            # Random Perturbation: Remove random, add random valid
            if S:
                x = rng.choice(list(S))
                new_S.discard(x)
                candidates = [a for a in all_nums if a not in new_S and all(is_coprime(a, b) for b in new_S)]
                if candidates:
                    a = rng.choice(candidates)
                    new_S.add(a)
                    delta = w[a] - w[x]
                else:
                    continue
                    
        # Acceptance
        if delta > 0 or rng.random() < math.exp(delta / T):
            S = new_S
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_S = S.copy()
                
        T *= cooling
        
    return list(best_S)
