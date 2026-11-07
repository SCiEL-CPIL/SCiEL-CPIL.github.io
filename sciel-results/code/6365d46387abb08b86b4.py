"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _get_prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _is_valid_set(S_list):
    """Check if sorted list S_list satisfies: for all distinct a,b,c, a does not divide b*c."""
    n = len(S_list)
    if n < 3:
        return True
    
    # Optimization: sort by value? The loop order doesn't matter for correctness.
    # We iterate i < j < k.
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            for k in range(j + 1, n):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit (actually 90s wall clock, but keep tight)
    n_max = 256
    
    # 1. Start with all primes
    primes = [x for x in range(2, n_max + 1) if _is_prime(x)]
    best = primes[:]
    best_size = len(best)
    
    # Precompute prime factors for all numbers up to n_max
    prime_factors_cache = {}
    for x in range(2, n_max + 1):
        prime_factors_cache[x] = _get_prime_factors(x)
    
    # Strategy 1: Greedy addition to primes
    # We try to add composites.
    # Condition for adding x to S:
    # 1. x does not divide b*c for any distinct b,c in S.
    # 2. b does not divide x*c for any distinct b,c in S.
    # 3. c does not divide x*b for any distinct b,c in S.
    
    def _can_add(S_set, x):
        L = list(S_set)
        n = len(L)
        if n < 2:
            return True
        
        # Check x !| b*c
        for i in range(n):
            b = L[i]
            for j in range(i+1, n):
                c = L[j]
                if (b * c) % x == 0:
                    return False
        
        # Check b !| x*c and c !| b*x
        for i in range(n):
            b = L[i]
            for j in range(i+1, n):
                c = L[j]
                if (x * c) % b == 0:
                    return False
                if (b * x) % c == 0:
                    return False
        return True

    composites = [x for x in range(2, n_max + 1) if not _is_prime(x)]
    
    # Try adding in descending order (larger numbers are "safer" divisors? No, they are less likely to divide products, but more likely to be divided by? 
    # Actually, if x is large, x | b*c is rare. But b | x*c is more likely if x shares factors with c.
    # Let's try both orders.)
    
    for order in [sorted(composites, reverse=True), sorted(composites)]:
        if time.time() > deadline:
            break
            
        current_S = best[:]
        current_S_set = set(current_S)
        
        for x in order:
            if time.time() > deadline:
                break
            if x in current_S_set:
                continue
            if _can_add(current_S_set, x):
                current_S.append(x)
                current_S_set.add(x)
        
        if len(current_S) > best_size:
            best = current_S[:]
            best_size = len(current_S)
            best.sort()

    # Strategy 2: Inspired by Inspiration 3 - Structural search
    # Remove a prefix of small primes, and add composites whose prime factors are a subset of the removed primes.
    # This ensures that for any a in the remaining set (either a remaining prime or a composite of removed primes),
    # and b,c in the set, a | b*c implies... let's analyze.
    
    # If S consists of:
    # 1. Primes > P_k (where P_k is the k-th prime)
    # 2. Composites x such that PrimeFactors(x) subset {P_0, ..., P_{k-1}}
    
    # Let a, b, c be distinct elements in S.
    # Case 1: a is a remaining prime p > P_k.
    #   b and c are either remaining primes or composites of small primes.
    #   If b or c is a remaining prime q > P_k, then b*c has prime factors > P_k (unless b=c, but they are distinct).
    #   Wait, if b is a remaining prime q, q > P_k. c is any other element.
    #   a | b*c. Since a is prime, a | b or a | c.
    #   a != b and a != c (distinct).
    #   So a must divide c.
    #   If c is a remaining prime r > P_k, a | r implies a=r, contradiction.
    #   If c is a composite of small primes, its prime factors are <= P_k. Since a > P_k, a cannot divide c.
    #   So Case 1 is safe.
    
    # Case 2: a is a composite of small primes.
    #   a's prime factors are in R = {P_0, ..., P_{k-1}}.
    #   b, c are distinct elements in S.
    #   a | b*c.
    #   This is the tricky case.
    #   If b and c are both composites of small primes, their prime factors are in R.
    #   It is possible that a | b*c. Example: R={2,3}. a=6, b=2, c=3. 6 | 2*3.
    #   So this strategy does NOT automatically guarantee validity for composites vs composites.
    #   However, it guarantees validity for:
    #   - Prime vs anything (as shown above)
    #   - Composite vs Prime?
    #     Let a be composite (factors in R), b be prime p > P_k.
    #     a | b*c => a | c (since gcd(a,b)=1).
    #     c is in S.
    #     If c is prime > P_k, a | c impossible (a > 1, c prime, a != c).
    #     If c is composite (factors in R), a | c is possible.
    #     Example: R={2,3}. a=6, b=5, c=12. 6 | 5*12. Yes.
    #   So we must still check the set for validity.
    
    # But this structure allows us to generate candidate sets efficiently.
    
    # Let's refine the candidate generation.
    # For k in range(len(primes)+1):
    #   removed = set(primes[:k])
    #   remaining_primes = [p for p in primes if p not in removed]
    #   composites_to_add = [x for x in range(2, n_max+1) if not _is_prime(x) and prime_factors_cache[x].issubset(removed)]
    #   S_test = remaining_primes + composites_to_add
    #   if _is_valid_set(S_test): update best
    
    # We can also try to add MORE composites to this base set using greedy addition.
    
    for k in range(len(primes) + 1):
        if time.time() > deadline:
            break
            
        removed = set(primes[:k])
        remaining_primes = [p for p in primes if p not in removed]
        
        # Composites that only use removed primes
        composites_to_add = []
        for x in range(2, n_max + 1):
            if _is_prime(x):
                continue
            if prime_factors_cache[x].issubset(removed):
                composites_to_add.append(x)
        
        S_test = remaining_primes + composites_to_add
        S_test.sort()
        
        if _is_valid_set(S_test):
            if len(S_test) > best_size:
                best_size = len(S_test)
                best = S_test[:]
        
        # Now try to add remaining composites to this valid set using greedy
        current_S = S_test[:]
        current_S_set = set(current_S)
        
        # Candidates are composites NOT in current_S
        # We should prioritize composites that "fit" well.
        # Let's just try all remaining composites in descending order.
        remaining_composites = [x for x in composites if x not in current_S_set]
        
        for x in sorted(remaining_composites, reverse=True):
            if time.time() > deadline:
                break
            if _can_add(current_S_set, x):
                current_S.append(x)
                current_S_set.add(x)
        
        if len(current_S) > best_size:
            best_size = len(current_S)
            best = current_S[:]
            best.sort()

    # Strategy 3: Local Search (Swaps)
    # Try to swap an element out and another in to enable more additions.
    # This is expensive, so we do it sparingly.
    
    all_nums = list(range(2, n_max + 1))
    current_best = best[:]
    current_best_set = set(current_best)
    
    # Try a few rounds of swaps
    improved = True
    rounds = 0
    while improved and rounds < 5 and time.time() < deadline:
        improved = False
        rounds += 1
        for x in all_nums:
            if time.time() > deadline:
                break
            if x in current_best_set:
                continue
            for y in current_best:
                if time.time() > deadline:
                    break
                # Try swapping y for x
                S_test = [z for z in current_best if z != y] + [x]
                S_test.sort()
                if _is_valid_set(S_test):
                    # Check if we can add more to this new set
                    S_new = S_test[:]
                    S_new_set = set(S_new)
                    for z in all_nums:
                        if time.time() > deadline:
                            break
                        if z in S_new_set:
                            continue
                        if _can_add(S_new_set, z):
                            S_new.append(z)
                            S_new_set.add(z)
                    
                    if len(S_new) > len(current_best):
                        current_best = S_new[:]
                        current_best_set = S_new_set
                        best = S_new[:]
                        best_size = len(S_new)
                        improved = True
                        break
            if improved:
                break

    # Final verification
    if not _is_valid_set(best):
        # Fallback to primes
        return [int(x) for x in primes]
        
    return [int(x) for x in sorted(best)]

# EVOLVE-BLOCK-END
