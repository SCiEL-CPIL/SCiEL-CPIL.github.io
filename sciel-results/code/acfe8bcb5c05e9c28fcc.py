"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s limit
    N = 256
    numbers = list(range(2, N + 1))
    
    # Sieve for primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N+1) if is_prime[i]]
    composites = [i for i in range(2, N+1) if not is_prime[i]]
    
    # Start with all primes
    S = set(primes)
    S_list = list(primes)
    
    # Precompute: for each number x, which pairs (i,j) in current S would conflict
    # i.e., x | S[i]*S[j] for i != j
    # For greedy: try adding composites in order of fewest conflicts
    
    def count_conflicts(x, current_list):
        """Count how many pairs (a,b) in current_list have x | a*b"""
        cnt = 0
        n = len(current_list)
        for i in range(n):
            a = current_list[i]
            for j in range(i+1, n):
                b = current_list[j]
                if (a * b) % x == 0:
                    cnt += 1
        return cnt
    
    # Try to add composites, sorted by initial conflict count (fewest first)
    # But re-evaluate dynamically
    
    remaining = list(composites)
    
    # Initial sort by number of prime factors (fewer = less likely to divide products)
    def prime_factor_count(x):
        cnt = 0
        temp = x
        p = 2
        while p * p <= temp:
            while temp % p == 0:
                cnt += 1
                temp //= p
            p += 1
        if temp > 1:
            cnt += 1
        return cnt
    
    remaining.sort(key=prime_factor_count)
    
    # Greedy: repeatedly pick the composite with fewest conflicts
    while remaining and time.time() < deadline:
        if not remaining:
            break
        
        # Find the element with fewest conflicts
        best_idx = -1
        best_conf = float('inf')
        
        for idx, x in enumerate(remaining):
            conf = count_conflicts(x, S_list)
            if conf < best_conf:
                best_conf = conf
                best_idx = idx
        
        if best_idx == -1:
            break
        
        x = remaining[best_idx]
        
        # Check if adding x creates any violation with existing pairs
        # x | a*b for some pair (a,b) in S means adding x would violate the constraint
        # Wait - the constraint is: for every three DISTINCT members a,b,c, a does not divide b*c
        # So if we add x, we need: for all a,b in S, x does not divide a*b
        # AND for all a in S, a does not divide x * (other member)
        # Actually the constraint is symmetric: for any triple (a,b,c) all distinct in S,
        # none of them divides the product of the other two.
        
        # When adding x to S, new triples are (x, a, b) for a,b in S, a!=b.
        # We need: x does not divide a*b, AND a does not divide x*b, AND b does not divide x*a
        # for all distinct a,b in S.
        
        # So we need to check ALL three conditions, not just x|a*b
        
        can_add = True
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                # Check x | a*b
                if (a * b) % x == 0:
                    can_add = False
                    break
                # Check a | x*b
                if (x * b) % a == 0:
                    can_add = False
                    break
                # Check b | x*a
                if (x * a) % b == 0:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            S.add(x)
            S_list.append(x)
            remaining.pop(best_idx)
        else:
            # Remove x from remaining (it can't be added)
            remaining.pop(best_idx)
    
    result = list(S)
    result.sort()
    return result
# EVOLVE-BLOCK-END
