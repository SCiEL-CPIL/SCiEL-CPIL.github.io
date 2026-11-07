import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    LIMIT = 256
    candidates = list(range(2, LIMIT + 1))
    
    # Precompute divisibility: div[x][y] = x divides y
    # Actually, I need x | y*z for triples.
    # Let me precompute for each x, a set of (y,z) pairs where x | y*z
    # But that's too large. Instead, I'll compute on the fly with caching.
    
    # Better: precompute for each x, for each y, the set of z such that x | y*z
    # Or just do it on the fly.
    
    # Let me precompute prime factorizations for efficiency
    # Actually, let me just use direct modulo checks.
    
    def is_valid_addition(x, S_set, S_list):
        """Check if adding x to S is valid. S_set is set, S_list is list."""
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                y = S_list[i]
                z = S_list[j]
                # Check x | y*z
                if (y * z) % x == 0:
                    return False
                # Check y | x*z
                if (x * z) % y == 0:
                    return False
                # Check z | x*y
                if (x * y) % z == 0:
                    return False
        return True
    
    def is_valid_set(S_list):
        """Check if S_list is valid."""
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    # Start with all primes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, LIMIT + 1) if is_prime[i]]
    
    best_set = primes[:]
    best_size = len(primes)
    
    # Try to add composites greedily
    composites = [x for x in range(2, LIMIT + 1) if not is_prime[x]]
    
    # Try different orderings
    for trial in range(200):
        if time.time() > deadline:
            break
        
        if trial == 0:
            S = primes[:]
        else:
            # Randomized start: start with primes, then try random additions
            S = primes[:]
            random.shuffle(S)
        
        S_set = set(S)
        
        # Try to add each composite
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            for x in composites:
                if x in S_set:
                    continue
                if is_valid_addition(x, S_set, S):
                    S.append(x)
                    S_set.add(x)
                    added_any = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S[:]
        
        # Hill climbing: try swaps
        if time.time() > deadline:
            break
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_set = set(S)
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                removed = S[i]
                S_temp = S[:i] + S[i+1:]
                S_temp_set = S_set - {removed}
                for x in composites:
                    if x in S_temp_set:
                        continue
                    if is_valid_addition(x, S_temp_set, S_temp):
                        S_new = S_temp + [x]
                        if len(S_new) > len(S):
                            S = S_new
                            S_set = set(S)
                            improved = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_set = S[:]
                            break
                if improved:
                    break
                # Try adding without removing
                for x in composites:
                    if x in S_set:
                        continue
                    if is_valid_addition(x, S_set, S):
                        S.append(x)
                        S_set.add(x)
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = S[:]
                        break
    
    # Final verification
    if not is_valid_set(best_set):
        best_set = primes[:]
    
    return list(best_set)
