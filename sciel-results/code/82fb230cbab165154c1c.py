import time
import math
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(2, N + 1):
        for j in range(2, N + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    # Precompute for each a, b: d[a][b] = a // gcd(a, b)
    d_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(2, N + 1):
        for b in range(2, N + 1):
            d_table[a][b] = a // gcd_table[a][b]
    
    # Precompute multiples: for each d, list of multiples of d in [2, N]
    multiples = [[] for _ in range(N + 1)]
    for d in range(2, N + 1):
        for m in range(d, N + 1, d):
            multiples[d].append(m)
    
    def is_valid(S):
        """Check if set S is valid: no three distinct a,b,c with a | b*c"""
        S_set = set(S)
        S_list = sorted(S)
        m = len(S_list)
        if m < 3:
            return True
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if i == j:
                    continue
                d = d_table[a][b]
                if d == 1:
                    # Any c works, and m >= 3 so there is a c distinct from a,b
                    return False
                # Need c in S, c != a, c != b, such that d | c
                for c in multiples[d]:
                    if c in S_set and c != a and c != b:
                        return False
        return True
    
    def can_add(c, S, S_set):
        """Check if S U {c} is valid, assuming S is already valid"""
        S_list = list(S)
        m = len(S_list)
        if m < 2:
            return True
        
        # Check all triples involving c
        # For each pair (a, b) in S, check if a|c*b, b|c*a, or c|a*b
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                # Check a | c*b: d = a//gcd(a,c), need d|b
                d1 = d_table[a][c]
                if d1 == 1:
                    return False
                if b % d1 != 0:
                    # a does not divide c*b, good
                    pass
                else:
                    return False
                
                # Check b | c*a: d = b//gcd(b,c), need d|a
                d2 = d_table[b][c]
                if d2 == 1:
                    return False
                if a % d2 != 0:
                    pass
                else:
                    return False
                
                # Check c | a*b: d = c//gcd(c,a), need d|b
                d3 = d_table[c][a]
                if d3 == 1:
                    return False
                if b % d3 != 0:
                    pass
                else:
                    return False
        
        return True
    
    # Start with primes
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    if N >= 1:
        is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    S = [i for i in range(2, N + 1) if is_prime[i]]
    S_set = set(S)
    best_S = S[:]
    best_size = len(S)
    
    # Greedy: try to add each composite in order of increasing "conflict potential"
    # Heuristic: numbers with more prime factors tend to conflict more
    def num_prime_factors(n):
        count = 0
        temp = n
        for p in range(2, int(temp**0.5) + 1):
            while temp % p == 0:
                count += 1
                temp //= p
        if temp > 1:
            count += 1
        return count
    
    composites = [i for i in range(2, N + 1) if not is_prime[i]]
    # Sort by number of prime factors (ascending) - fewer factors = less conflict
    composites.sort(key=lambda x: num_prime_factors(x))
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for c in composites:
            if c in S_set:
                continue
            if time.time() > deadline - 10:
                break
            
            if can_add(c, S, S_set):
                S.append(c)
                S_set.add(c)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S[:]
                improved = True
    
    # Try random restarts with different initial sets
    if time.time() < deadline - 15:
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            # Start with a random subset of primes + some composites
            random.seed(42 + restart)
            S2 = []
            S2_set = set()
            
            # Add primes randomly
            primes = [i for i in range(2, N + 1) if is_prime[i]]
            random.shuffle(primes)
            for p in primes:
                if can_add(p, S2, S2_set):
                    S2.append(p)
                    S2_set.add(p)
            
            # Try to add composites
            composites2 = [i for i in range(2, N + 1) if not is_prime[i]]
            random.shuffle(composites2)
            for c in composites2:
                if c in S2_set:
                    continue
                if can_add(c, S2, S2_set):
                    S2.append(c)
                    S2_set.add(c)
            
            if len(S2) > best_size:
                best_size = len(S2)
                best_S = S2[:]
    
    # Final validation
    if is_valid(best_S):
        return list(best_S)
    # Fallback: return primes
    return [i for i in range(2, N + 1) if is_prime[i]]

# EVOLVE-BLOCK-END
