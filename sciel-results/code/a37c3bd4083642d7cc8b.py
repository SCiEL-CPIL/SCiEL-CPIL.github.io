import random
import time

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    n = P['n']
    
    # Precompute primes for better heuristics
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(n)
    
    # Helper to check if x can be added to S
    # S is a list of ints
    def can_add(x, S):
        k = len(S)
        if k < 2:
            return True
        
        # We need to check for all distinct a, b in S:
        # 1. x does not divide a*b
        # 2. a does not divide x*b
        # 3. b does not divide x*a
        
        for i in range(k):
            a = S[i]
            for j in range(i + 1, k):
                b = S[j]
                # Check 1: x | a*b
                if (a * b) % x == 0:
                    return False
                # Check 2: a | x*b
                if (x * b) % a == 0:
                    return False
                # Check 3: b | x*a
                if (x * a) % b == 0:
                    return False
        return True

    best_set = []
    
    # Strategy 1: Greedy with various heuristics
    # Heuristic 1: Descending value (prefer large numbers)
    # Heuristic 2: Ascending value (prefer small numbers)
    # Heuristic 3: Primes first, then composites
    # Heuristic 4: Random restarts
    
    def greedy_build(candidates):
        S = []
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.append(x)
        return S
    
    # Run multiple greedy passes with different orderings
    # Pass 1: Descending
    candidates = list(range(n, 1, -1))
    S1 = greedy_build(candidates)
    if len(S1) > len(best_set):
        best_set = S1[:]
    
    # Pass 2: Ascending
    candidates = list(range(2, n + 1))
    S2 = greedy_build(candidates)
    if len(S2) > len(best_set):
        best_set = S2[:]
    
    # Pass 3: Primes first, then composites
    primes = [x for x in range(2, n + 1) if is_prime[x]]
    composites = [x for x in range(2, n + 1) if not is_prime[x]]
    candidates = primes + composites
    S3 = greedy_build(candidates)
    if len(S3) > len(best_set):
        best_set = S3[:]
    
    # Pass 4: Primes descending, composites descending
    candidates = sorted(prices, reverse=True) + sorted(composites, reverse=True)
    S4 = greedy_build(candidates)
    if len(S4) > len(best_set):
        best_set = S4[:]
    
    # Random restarts
    for restart in range(30):
        if time.time() > deadline:
            break
        
        candidates = list(range(2, n + 1))
        random.shuffle(candidates)
        S_r = greedy_build(candidates)
        if len(S_r) > len(best_set):
            best_set = S_r[:]
    
    # Local improvement: try to add more numbers by reordering
    # Try to add numbers that were rejected, in different orders
    if time.time() < deadline - 5:
        # Try to improve by removing one element and adding two others
        S = best_set[:]
        rejected = [x for x in range(2, n + 1) if x not in S]
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try removing each element and see if we can add more
            for i in range(len(S)):
                if time.time() > deadline - 2:
                    break
                S_copy = S[:]
                removed = S_copy.pop(i)
                
                # Try to add rejected elements
                added_count = 0
                for x in rejected:
                    if x == removed:
                        continue
                    if can_add(x, S_copy):
                        S_copy.append(x)
                        added_count += 1
                
                if len(S_copy) > len(S):
                    S = S_copy
                    improved = True
                    break
    
    if len(S) > len(best_set):
        best_set = S[:]
    
    # Ensure return type is list of ints
    return [int(x) for x in best_set]
