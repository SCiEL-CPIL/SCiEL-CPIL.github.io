def construct():
    import time
    from math import gcd
    import random

    N = 512
    start = time.time()
    deadline = start + 140.0

    # Generate primes < 512
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False
    primes = [i for i in range(2, N) if is_prime[i]]
    
    # Initial set: 1 and max powers of primes
    S = [1]
    prime_powers = {}
    for p in primes:
        k = 1
        val = p
        while val * p < N:
            val *= p
            k += 1
        S.append(val)
        prime_powers[val] = p # Map power back to prime for merge logic? 
                               # Actually we just need the set of primes currently "represented"
       
    # To track which primes are "active" (i.e., their power is in S)
    # If we merge, we remove the powers and add the composite.
    # We need to know which primes are in the set to check conflicts.
    
    def get_prime_factors(a):
        factors = []
        d = 2
        n = a
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    def score(S):
        return sum(1.0 / (N - a) for a in S)

    def is_coprime(a, S):
        for b in S:
            if gcd(a, b) != 1:
                return False
        return True

    best_S = S[:]
    best_score = score(best_S)
    
    # Local Search: Try to merge subsets of primes
    # Represent S as a set of "atomic" units?
    # No, S contains numbers. Some are primes powers, some composites.
    # We can only merge if the numbers being merged are "prime powers" of distinct primes.
    # If a number is already composite, we can't merge it further (it's already "merged").
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Identify prime powers in current S
        # A number x in S is a prime power if x = p^k.
        # We can merge any subset of these prime powers.
        
        pp_list = []
        for a in S:
            # Check if a is a prime power
            # Simple check: is a in prime_powers? No, prime_powers maps value->prime.
            # But S might contain composites from previous merges.
            # We should only merge prime powers.
            # Let's identify which elements in S are prime powers.
            pass
            
        # This logic is getting complex to track "which primes are free".
        # Simpler: Just try all pairs of elements in S that are coprime?
        # No, we want to replace $p_1, p_2$ with $p_1 p_2$.
        # So we look for $a, b \in S$ such that $\gcd(a,b)=1$.
        # If $a \cdot b < 512$, try replacing $a, b$ with $a \cdot b$.
        # This works even if $a, b$ are composites, as long as they are coprime.
        # But if $a$ is composite, it already "uses" its factors.
        # Merging $a$ and $b$ creates $ab$.
        # We must ensure $ab$ is coprime to all other elements in $S \setminus \{a,b\}$.
        
        S_list = S[:]
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                if gcd(a, b) != 1:
                    continue
                prod = a * b
                if prod >= N:
                    continue
                
                # Check if prod is coprime to rest
                rest = [x for k, x in enumerate(S_list) if k != i and k != j]
                if all(gcd(prod, x) == 1 for x in rest):
                    new_S = rest + [prod]
                    new_score = score(new_S)
                    if new_score > best_score + 1e-12:
                        best_S = new_S
                        best_score = new_score
                        improved = True
                        S = new_S
                        S_list = S[:]
                        break
            if improved:
                break
        
        # Also try single element replacement?
        # Replace a with b? No, b must be coprime to rest.
        # This is just adding b and removing a.
        # We can iterate all a in 1..511 not in S.
        # If gcd(a, S) == 1, try adding a.
        # But we must remove something? No, we can just add if it fits?
        # No, "distinct pairwise coprime". We can add if it's coprime to all.
        # But we might have "filled" the space.
        # Actually, if we can add a coprime number, we should, because weight is positive.
        # So we should always try to add any available coprime number.
        
        # Add loop
        for a in range(N-1, 0, -1):
            if a in S:
                continue
            if all(gcd(a, x) == 1 for x in S):
                new_S = S + [a]
                new_score = score(new_S)
                if new_score > best_score + 1e-12:
                    best_S = new_S
                    best_score = new_score
                    improved = True
                    S = new_S
                    break

    return best_S
