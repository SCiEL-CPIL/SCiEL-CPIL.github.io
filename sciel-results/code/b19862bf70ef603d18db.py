import time
from itertools import combinations
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(200)
    
    def is_smooth(n, smooth_primes):
        if n <= 1:
            return True
        temp = n
        for p in smooth_primes:
            while temp % p == 0:
                temp //= p
        return temp == 1
    
    # Strategy: choose S, compute P=product(S), find x_i in [1, floor(10000/P)]
    # such that all x_i+x_j are S-smooth. Then a_i = P*x_i.
    
    best_result = None
    best_prime_count = float('inf')
    
    # Try various prime sets
    prime_sets_to_try = [
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13},
        {2, 3, 5, 11, 13},
        {2, 3, 7, 11, 13},
        {2, 5, 7, 11, 13},
        {2, 3, 5, 7, 11, 13, 17},
        {3, 5, 7, 11, 13},
        {2, 3, 5, 11},
        {2, 3, 7, 11},
        {2, 5, 7, 11},
    ]
    
    for S in prime_sets_to_try:
        if time.time() > deadline:
            break
        
        S_list = sorted(S)
        P = 1
        for p in S_list:
            P *= p
        
        max_x = 10000 // P
        if max_x < 12:
            continue
        
        # Find all x in [1, max_x] such that for the set we build,
        # all pair sums are S-smooth
        
        # Precompute: for each x in [1, max_x], which y in [1, max_x] have x+y S-smooth
        compatible = {}
        for x in range(1, max_x + 1):
            compatible[x] = set()
            for y in range(1, max_x + 1):
                if x == y:
                    continue
                if is_smooth(x + y, S_list):
                    compatible[x].add(y)
        
        # Now find a clique of size 12 in the compatibility graph
        # Use backtracking
        found = []
        
        def backtrack(candidates, current):
            if time.time() > deadline:
                return False
            if len(current) == 12:
                return True
            
            # Pick candidate with fewest options (MRV heuristic)
            if not candidates:
                return False
            
            # Try each candidate
            for c in list(candidates):
                # Check if c is compatible with all in current
                valid = True
                for x in current:
                    if c not in compatible[x]:
                        valid = False
                        break
                if not valid:
                    continue
                
                current.append(c)
                # New candidates must be compatible with c
                new_candidates = [x for x in candidates if x > c and x in compatible[c]]
                if backtrack(new_candidates, current):
                    return True
                current.pop()
            
            return False
        
        all_x = list(range(1, max_x + 1))
        if backtrack(all_x, []):
            result = [P * x for x in found]
            if len(set(result)) == 12 and all(1 <= x <= 10000 for x in result):
                prime_count = len(S)
                if prime_count < best_prime_count:
                    best_prime_count = prime_count
                    best_result = result
    
    if best_result is not None:
        return best_result
    
    # Fallback
    return list(range(1, 13))

