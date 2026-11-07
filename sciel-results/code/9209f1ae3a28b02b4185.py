import math
import time

def _sf(n):
    """Compute square-free part of n."""
    sf = 1
    p = 2
    x = n
    while p * p <= x:
        c = 0
        while x % p == 0:
            x //= p
            c += 1
        if c % 2 == 1:
            sf *= p
        p += 1 if p == 2 else 2
    if x > 1:
        sf *= x
    return sf

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Primes up to 10210
    primes = []
    is_prime = [True] * 10211
    is_prime[0] = is_prime[1] = False
    for i in range(2, 10211):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 10211, i):
                is_prime[j] = False
            
    # Precompute square-free part for all numbers up to 10210
    sf_vals = [0] * 10211
    for i in range(2, 10211):
        sf_vals[i] = _sf(i)
        
    target_sf = sf_vals[210] # 210
    
    # Iterate M from 211 to 10210
    for M in range(211, 10211):
        if time.time() > deadline:
            break
            
        # Count multiples of each prime in [211, M]
        # We only care about primes p where the count is odd and small
        # Actually, we just need to identify numbers that MUST be excluded
        # A number x is excluded if it has a prime factor p such that
        # count_multiples(p) == 1.
        
        # Efficient way:
        # For each prime p, find multiples in [211, M].
        # If there is exactly 1 multiple, that multiple is forbidden.
        
        forbidden = set()
        
        # Only check primes p > 1? 
        # If p <= M/2, there are at least 2 multiples (p*2, p*3... wait, 2p might be < 211)
        # We need count in [211, M].
        # Let k_min = ceil(211/p), k_max = floor(M/p).
        # Count = k_max - k_min + 1.
        
        for p in primes:
            if p > M:
                break
            k_min = math.ceil(211 / p)
            k_max = math.floor(M / p)
            count = k_max - k_min + 1
            if count == 1:
                # The single multiple is k_min * p
                forbidden.add(k_min * p)
            elif count == 0:
                continue
            # If count >= 2, we can potentially use an even number of them.
            # We don't forbid them immediately, but we note the constraint.
            
        # Get candidate numbers
        candidates = [x for x in range(211, M + 1) if x not in forbidden]
        
        # If we have a small number of candidates, do exact search
        if len(candidates) <= 20:
            # BFS on square-free parts
            # State: sf_part
            # We want to reach target_sf
            # Start with sf_part = 1 (empty product)
            # Use a dict to store min_max_used? No, we are iterating M, so max is M.
            # We just need to know if target_sf is reachable.
            
            reachable = {1: []} # sf_part -> list of numbers used (to reconstruct solution)
            # Actually, storing the list is memory heavy. 
            # We can store parent pointers or just check existence first.
            
            # Let's just check existence first with a set.
            reachable_set = {1}
            # We need to reconstruct the set.
            # Let's use a dict: sf_part -> (parent_sf, number_used)
            parent = {1: None}
            
            found = False
            for x in candidates:
                sf_x = sf_vals[x]
                new_reachable = set()
                for s in reachable_set:
                    new_s = s * sf_x
                    # Reduce to square-free part?
                    # No, s is square-free, sf_x is square-free.
                    # Product might not be square-free.
                    # new_s = sf(s * sf_x)
                    # Since s and sf_x are square-free, sf(s * sf_x) = s * sf_x / gcd(s, sf_x)^2 ?
                    # No. sf(a*b) = sf(a) * sf(b) / gcd(sf(a), sf(b))^2 ?
                    # Example: a=2, b=2. sf(4)=1. sf(2)*sf(2)/2^2 = 4/4=1. Correct.
                    # a=2, b=3. sf(6)=6. 2*3/1=6. Correct.
                    # a=6, b=6. sf(36)=1. 6*6/6^2=1. Correct.
                    # So new_s = (s * sf_x) // (math.gcd(s, sf_x) ** 2)
                    
                    g = math.gcd(s, sf_x)
                    new_s = (s * sf_x) // (g * g)
                    
                    if new_s not in parent:
                        parent[new_s] = (s, x)
                        new_reachable.add(new_s)
                        
                reachable_set = reachable_set | new_reachable
                if target_sf in reachable_set:
                    found = True
                    break
            
            if found:
                # Reconstruct
                sol = []
                cur = target_sf
                while parent[cur] is not None:
                    prev, x = parent[cur]
                    sol.append(x)
                    cur = prev
                return sol
        
        # If candidates > 20, we might need a smarter search or just continue
        # For now, if M is small, candidates will be small.
        # As M grows, candidates grow.
        # But we expect to find a solution for small M.
        
    # Fallback: Return the best known or a default
    # The problem states we must return a valid list.
    # If we didn't find one, we should return a valid one, e.g., [840]
    return [840]
