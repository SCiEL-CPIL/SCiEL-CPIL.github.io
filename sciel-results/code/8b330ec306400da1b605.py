import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 256
    candidates = list(range(2, n + 1))
    
    # Precompute: for each a, list of pairs (b, c) with b < c, b != a, c != a, a | b*c
    # This is the "conflict" structure.
    # For a given set S, the constraint is: no triple (a, b, c) in S with a | b*c.
    # Equivalently, for each a in S, there is no pair (b, c) in S\{a} such that a | b*c.
    
    # Let's precompute for each a, the set of pairs (b, c) that conflict with a.
    # But that's too many. Instead, for a given S, we can check validity by:
    # For each a in S, check if there exist b, c in S\{a} with a | b*c.
    
    # A faster check: for each a in S, let P_a = {b in S\{a} : b is a multiple of some factor of a?}
    # Actually, a | b*c iff for all prime powers p^k || a, p^k | b*c.
    # This is complex. Let's just do the direct check but optimized.
    
    # For a given a, and a set S, we want to know if there exist b, c in S\{a} with a | b*c.
    # We can precompute for each a, a list of "required" conditions on b and c.
    # But let's just iterate over pairs in S.
    
    # Optimization: sort S. For each a, iterate over b in S, b != a.
    # For a fixed a and b, we need c in S\{a, b} such that a | b*c.
    # a | b*c  <=>  (a / gcd(a, b)) | c.
    # So for each a, b, we compute d = a // gcd(a, b). We need c in S\{a, b} such that d | c.
    # If d == 1, then any c works (but c must be distinct from a, b). So if |S| >= 3, conflict exists.
    # If d > 1, we need a multiple of d in S\{a, b}.
    
    import math
    
    def is_valid(S):
        S_list = sorted(S)
        S_set = set(S_list)
        m = len(S_list)
        if m < 3:
            return True
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if i == j:
                    continue
                d = a // math.gcd(a, b)
                if d == 1:
                    # Any c distinct from a, b works. Since m >= 3, there is such c.
                    return False
                # Need c in S_list, c != a, c != b, such that d | c.
                # d | c means c is a multiple of d.
                # Since c <= 256, and d >= 2, multiples are limited.
                # We can check: is there a multiple of d in S_set, excluding a and b?
                # The multiples of d up to 256 are d, 2d, 3d, ...
                # We need one of them to be in S_set, and not equal to a or b.
                found = False
                for k in range(d, 257, d):
                    if k in S_set and k != a and k != b:
                        found = True
                        break
                if found:
                    return False
        return True
    
    # Precompute gcd for all pairs? Might be useful.
    # But let's start with primes.
    
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    if n >= 1: is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    S = [i for i in range(2, n + 1) if is_prime[i]]
    S_set = set(S)
    best_S = S[:]
    best_size = len(S)
    
    # Local Search with swaps and adds
    # We'll try to add elements if possible, and swap if it leads to improvement.
    
    def try_add(c, S, S_set):
        # Check if S U {c} is valid.
        # Only need to check triples involving c.
        # For each a in S, b in S, a != b, check if a | c*b, c | a*b, b | c*a.
        # But we can optimize: 
        # For a | c*b: d = a // gcd(a, c). Need d | b. So check if there's b in S with d | b.
        # For c | a*b: d = c // gcd(c, a). Need d | b.
        # For b | c*a: d = b // gcd(b, c). Need d | a.
        
        S_list = list(S)
        m = len(S_list)
        if m < 2:
            return True  # Can add c, no triples to violate.
        
        # Check for conflicts with c.
        # For each pair (a, b) in S, a != b:
        #   Check if a | c*b OR b | c*a OR c | a*b.
        # This is O(m^2) per add attempt.
        
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                # Check a | c*b
                d1 = a // math.gcd(a, c)
                if d1 == 1:
                    return False
                if b % d1 == 0:
                    return False
                # Check b | c*a
                d2 = b // math.gcd(b, c)
                if d2 == 1:
                    return False
                if a % d2 == 0:
                    return False
                # Check c | a*b
                d3 = c // math.gcd(c, a)
                if d3 == 1:
                    return False
                if b % d3 == 0:
                    return False
                # Also check c | a*b with b and a swapped? No, a*b is symmetric.
                # But we need to check if c | a*b.
                # d3 = c // gcd(c, a). Need d3 | b.
                # That's what we did.
                # What if gcd(c, a) doesn't capture all?
                # c | a*b <=> (c / gcd(c, a)) | b. Yes.
                # But we also need to consider that b might not be the only factor.
                # Actually, the condition is correct.
                
        return True
    
    # Try to improve
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding each non-member
        for c in candidates:
            if c in S_set:
                continue
            if time.time() > deadline - 5:
                break
            
            if try_add(c, S, S_set):
                S.append(c)
                S_set.add(c)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S[:]
                improved = True
                continue
            
            # Try swaps: remove a, add c
            # Find a in S such that S\{a} U {c} is valid.
            # And ideally, this enables future adds.
            # For now, just check validity.
            
            # Which a to remove?
            # Candidates: a such that a conflicts with c in some way.
            # Or just try a few random a's.
            
            # Let's try to find an a that, when removed, allows c to be added.
            # This is equivalent to checking if S\{a} U {c} is valid.
            # We can try a few a's.
            
            # Heuristic: remove a prime factor of c, or a number that shares factors with c.
            temp_c = c
            factors = []
            for p in range(2, int(temp_c**0.5) + 1):
                if temp_c % p == 0:
                    factors.append(p)
                    while temp_c % p == 0:
                        temp_c //= p
            if temp_c > 1:
                factors.append(temp_c)
            
            # Try removing each factor that is in S
            tried = set()
            for a in factors:
                if a in S_set and a not in tried:
                    tried.add(a)
                    S.remove(a)
                    S_set.remove(a)
                    S.append(c)
                    S_set.add(c)
                    
                    if is_valid(S):
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S[:]
                        improved = True
                        break
                    else:
                        S.remove(c)
                        S_set.remove(c)
                        S.append(a)
                        S_set.add(a)
            
            # Also try removing a random element
            if not improved and random.random() < 0.1:
                a = random.choice(S)
                if a != c:
                    S.remove(a)
                    S_set.remove(a)
                    S.append(c)
                    S_set.add(c)
                    
                    if is_valid(S):
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S[:]
                        improved = True
                    else:
                        S.remove(c)
                        S_set.remove(c)
                        S.append(a)
                        S_set.add(a)
    
    # Final validation
    if is_valid(best_S):
        return list(best_S)
    return list(S)

# EVOLVE-BLOCK-END
