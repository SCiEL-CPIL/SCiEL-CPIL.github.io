import time
import random
from math import isqrt, gcd

N_MAX = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute primes
    def get_primes(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = get_primes(N_MAX)
    
    # For each x in 2..256, precompute:
    # divisors_in_range[x] = set of d in [2,256] such that d | x
    # multiples_in_range[x] = set of m in [2,256] such that x | m
    divisors = [set() for _ in range(N_MAX + 1)]
    multiples = [set() for _ in range(N_MAX + 1)]
    for x in range(2, N_MAX + 1):
        for d in range(2, x + 1):
            if x % d == 0:
                divisors[x].add(d)
                multiples[d].add(x)
    
    def can_add(S_list, x):
        """Check if x can be added to S_list."""
        n = len(S_list)
        if n == 0:
            return True
        if n == 1:
            a = S_list[0]
            return (a * x) % a != 0 and (a * x) % x != 0
            # Actually for n=1, no triple exists, so always True.
            # Wait, the condition is for every THREE DISTINCT members.
            # If |S| < 3, there are no triples, so any set of size < 3 is valid.
            # But we want to maximize, so we'll add as many as possible.
            # For n=1, adding x gives |S|=2, still no triples. Always valid.
            # For n=2, adding x gives |S|=3, need to check the one triple.
            return True
        
        # For n >= 2, necessary condition: no a in S divides x
        # Because if a | x, then for any b in S \ {a}, a | x*b
        for a in S_list:
            if a in divisors[x]:  # a | x
                return False
        
        # Now check: for all a,b in S_list with a != b, x | a*b is false
        # And: for all a,b in S_list with a != b, a | x*b is false
        # We already handled a | x case. Now a | x*b means (a/gcd(a,x)) | b
        # And x | a*b means (x/gcd(x,a)) | b
        
        # Check x | a*b for all pairs
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (a * b) % x == 0:
                    return False
        
        # Check a | x*b for all pairs (a, b) with a != b
        for i in range(n):
            a = S_list[i]
            g = gcd(a, x)
            k = a // g  # need k | b
            if k == 1:
                # This would mean a | x, already checked
                continue
            for j in range(n):
                if i == j:
                    continue
                b = S_list[j]
                if b % k == 0:
                    return False
        
        return True
    
    def can_add_fast(S_set, S_list, x):
        """Faster check using sets."""
        n = len(S_list)
        if n < 2:
            return True
        
        # Check: no a in S divides x
        for a in S_list:
            if a in divisors[x]:
                return False
        
        # Check x | a*b
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (a * b) % x == 0:
                    return False
        
        # Check a | x*b
        for i in range(n):
            a = S_list[i]
            g = gcd(a, x)
            k = a // g
            if k > 1:
                for j in range(n):
                    if i == j:
                        continue
                    b = S_list[j]
                    if b % k == 0:
                        return False
        
        return True
    
    # Multi-start greedy with local search
    best_S = set()
    best_size = 0
    
    for trial in range(50):
        if time.time() > deadline - 2.0:
            break
        
        random.seed(trial * 1000 + 42)
        
        # Start with primes, then greedily add
        S = set(primes)
        S_list = list(S)
        
        # Greedy add in random order
        candidates = [x for x in range(2, N_MAX + 1) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 1.0:
                break
            if can_add_fast(S, S_list, x):
                S.add(x)
                S_list.append(x)
        
        # Local search: try to add more
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for x in range(2, N_MAX + 1):
                if time.time() > deadline - 0.5:
                    break
                if x not in S:
                    if can_add_fast(S, S_list, x):
                        S.add(x)
                        S_list.append(x)
                        improved = True
        
        # Try 1-swap
        if time.time() < deadline - 0.5:
            c_list = [x for x in range(2, N_MAX + 1) if x not in S]
            s_list = list(S)
            random.shuffle(s_list)
            random.shuffle(c_list)
            
            for y in s_list:
                if time.time() > deadline - 0.5:
                    break
                S_list.remove(y)
                for x in c_list:
                    if time.time() > deadline - 0.5:
                        break
                    if can_add_fast(S, S_list, x):
                        S_list.append(x)
                        S = set(S_list)
                        improved = True
                        break
                if not improved:
                    S_list.append(y)
                else:
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    return [int(x) for x in best_S]
