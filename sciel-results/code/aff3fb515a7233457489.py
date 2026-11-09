import time
from math import isqrt

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True

def _can_add(x, S_list):
    """Check if x can be added to S_list maintaining validity."""
    n = len(S_list)
    # x as 'a': x ∤ b*c for all distinct b,c in S
    for i in range(n):
        for j in range(i+1, n):
            if (S_list[i] * S_list[j]) % x == 0:
                return False
    # x as 'b' or 'c': a ∤ x*c and a ∤ c*x for all distinct a,c in S
    for i in range(n):
        for j in range(i+1, n):
            if (x * S_list[j]) % S_list[i] == 0:
                return False
            if (x * S_list[i]) % S_list[j] == 0:
                return False
    return True

def _count_conflicts(x, S_list):
    """Count how many triple violations x would create with S_list."""
    n = len(S_list)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if (S_list[i] * S_list[j]) % x == 0:
                count += 1
            if (x * S_list[j]) % S_list[i] == 0:
                count += 1
            if (x * S_list[i]) % S_list[j] == 0:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    all_nums = list(range(2, 257))
    primes = [x for x in all_nums if _is_prime(x)]
    composites = [x for x in all_nums if not _is_prime(x)]
    
    best_set = list(primes)  # Start with all primes as baseline
    
    # Strategy: Start with all primes, then greedily add composites with fewest conflicts
    # Also try: start with primes, add composites, then try removing primes to add more composites
    
    # Phase 1: Greedy add composites to primes
    S = list(primes)
    # Sort composites by number of prime factors (fewer = less likely to divide products)
    # Actually, sort by "difficulty" - composites that are harder to divide products
    # A composite x conflicts if x|b*c for some b,c in S, or a|x*c for some a,c in S
    # For primes in S, x|b*c means x is a divisor of b*c where b,c are distinct primes
    # This only happens if x = b*c (semiprime with both factors in S) or x is prime (not composite)
    # So semiprimes whose both prime factors are in S cannot be added
    # Other composites might be addable
    
    # Sort composites: prefer those that are NOT semiprimes with both factors in primes
    def _composite_key(x):
        # Count how many prime factors x has (with multiplicity)
        # Fewer distinct prime factors = less likely to divide a product of two primes
        n = x
        distinct_pf = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                distinct_pf.add(d)
                n //= d
            d += 1
        if n > 1:
            distinct_pf.add(n)
        # Prefer composites with more distinct prime factors? No...
        # Actually, we want composites that DON'T divide any b*c where b,c are primes in S
        # b*c is semiprime. x divides b*c iff x is 1, p, q, or p*q for primes p,q
        # For composite x, x divides b*c iff x = b*c (and b,c distinct primes in S)
        # OR if x has more than 2 prime factors, x cannot divide a semiprime b*c
        # So composites with >= 3 prime factors (counting multiplicity, i.e., not semiprimes) 
        # can never divide b*c where b,c are distinct primes!
        # They can only conflict via the other condition: a | x*c
        # where a is a prime in S. a | x*c means a divides x*c.
        # If a is prime and a | x*c, then a | x or a | c.
        # Since c is prime and c != a (distinct), a | c is false. So a | x.
        # So the conflict is: for prime a in S, a | x. i.e., a is a prime factor of x.
        # And c must be a prime in S, c != a, and a | x*c which is true iff a | x.
        # Wait, we need a | x*c. If a | x, then a | x*c for any c. 
        # So if a is a prime factor of x and a is in S (primes), then for any c in S with c != a,
        # we have a | x*c. This is a violation!
        # So x can only be added if NO prime factor of x is in S (the prime set).
        # But all primes <= 256 are in S... so no composite can be added if S contains all primes!
        # 
        # Wait, that means with all 54 primes, we cannot add ANY composite.
        # Let me verify: x = 4 = 2^2. a=2 is in S. For any c in S, c != 2: 2 | 4*c? Yes! 4*c/2 = 2c, integer. Violation!
        # x = 6 = 2*3. a=2 in S. For c=5 (prime in S, c!=2): 2 | 6*5=30? Yes. Violation!
        # So indeed, no composite can be added to the full prime set.
        
        # This means we need to REMOVE some primes to add composites.
        # Trade-off: remove k primes, add m composites. Net gain = m - k.
        
        # For a composite x with prime factorization x = p1^a1 * p2^a2 * ... * pk^ak:
        # x can be in S iff for every prime p in S that is a factor of x, there is NO other prime c in S with c != p.
        # Wait, let me re-derive. The constraint is: for distinct a,b,c in S, a ∤ b*c.
        # If a is prime and a | x (x composite in S), then for any c in S, c != a: a | x*c. Violation.
        # So if x is in S and a is a prime factor of x with a in S, then for ALL c in S with c != a, a | x*c.
        # This is a violation. So we need: for every prime factor a of x, a is NOT in S.
        # But a is a prime factor of x, and a <= x <= 256, so a is in the domain.
        # If a is not in S, that's fine.
        
        # So: x can coexist with the prime set S_primes iff no prime factor of x is in S_primes.
        # But all primes <= 256 are potential members. If we keep all primes, no composite works.
        # If we remove prime p, then composites with p as a factor might work (if their other prime factors are also removed or not in S).
        
        # More precisely: x can be in S iff for every prime p that divides x, p is NOT in S.
        # (Because if p | x and p in S, then for any c in S, c != p: p | x*c, violation.)
        
        # Additionally, we need to check: for any a in S (not just primes), a ∤ x*c and a ∤ c*x for all c in S, c != a, c != x.
        # And x ∤ b*c for all b,c in S, b != c, b != x, c != x.
        
        # This is complex. Let me just do a greedy approach with local search.
        return len(distinct_pf)
    
    # Given the analysis above, the key insight is:
    # A composite x can be in S only if none of its prime factors are in S.
    # So we need to remove prime factors of composites we want to add.
    
    # Strategy: For each composite x, determine which primes must be removed (its prime factors).
    # Then find a good set of composites to add vs primes to remove.
    
    # Let's try: group composites by their set of prime factors.
    # For each prime p, composites that have p as a factor need p removed.
    
    # Better approach: local search. Start with all primes. Try removing one prime and adding composites that become valid.
    
    # For now, let's do a simpler greedy: try all single-prime removals
    current_best = list(primes)
    
    for p_to_remove in primes:
        if time.time() > deadline - 10:
            break
        S_test = [x for x in primes if x != p_to_remove]
        # Now try to add composites
        # A composite x can be added if:
        # 1. No prime factor of x is in S_test
        # 2. x ∤ b*c for all distinct b,c in S_test
        # 3. For all a in S_test, a ∤ x*c and a ∤ c*x for all c in S_test, c != a, c != x
        
        # Condition 1: prime factors of x not in S_test
        # Since S_test = primes - {p_to_remove}, condition 1 means:
        # all prime factors of x must be p_to_remove (i.e., x is a power of p_to_remove)
        # OR x has no prime factors (impossible) 
        # Wait: condition is "no prime factor of x is in S_test"
        # S_test contains all primes except p_to_remove
        # So prime factors of x must all equal p_to_remove
        # i.e., x must be a power of p_to_remove: p^k
        
        # Let me verify: x = p^k. Prime factors of x: {p}. Is p in S_test? No (we removed it). Good.
        # x = p*q where q != p. Prime factors: {p, q}. Is q in S_test? Yes (q is prime, q != p). Bad.
        # So only powers of p_to_remove can be added.
        
        # Powers of p in 2..256:
        powers = []
        pk = p_to_remove
        while pk <= 256:
            if pk != p_to_remove:  # exclude the prime itself (already removed)
                powers.append(pk)
            pk *= p_to_remove
        
        added = 0
        for x in powers:
            if time.time() > deadline - 10:
                break
            if _can_add(x, S_test):
                S_test.append(x)
                added += 1
        
        if len(S_test) > len(current_best):
            current_best = S_test
    
    # Also try removing 2 primes
    if time.time() < deadline - 20:
        n_primes = len(primes)
        for i in range(n_primes):
            if time.time() > deadline - 15:
                break
            p1 = primes[i]
            for j in range(i+1, n_primes):
                if time.time() > deadline - 15:
                    break
                p2 = primes[j]
                S_test = [x for x in primes if x != p1 and x != p2]
                
                # Composites whose prime factors are subset of {p1, p2}
                # These are: p1^a * p2^b for a>=1, b>=0, a*b not both 0, value in 2..256
                # But we need prime factors NOT in S_test, i.e., all prime factors in {p1, p2}
                candidates = []
                for a in range(1, 20):
                    for b in range(0, 20):
                        val = (p1 ** a) * (p2 ** b)
                        if val > 256:
                            break
                        if val >= 2 and val != p1 and val != p2:
                            candidates.append(val)
                # Also powers of p1 and p2 individually
                pk = p1 * p1
                while pk <= 256:
                    if pk not in candidates:
                        candidates.append(pk)
                    pk *= p1
                pk = p2 * p2
                while pk <= 256:
                    if pk not in candidates:
                        candidates.append(pk)
                    pk *= p2
                
                candidates = list(set(candidates))
                # Sort by value
                candidates.sort()
                
                for x in candidates:
                    if time.time() > deadline - 15:
                        break
                    if _can_add(x, S_test):
                        S_test.append(x)
                
                if len(S_test) > len(current_best):
                    current_best = S_test
    
    if len(current_best) > len(best_set):
        best_set = current_best
    
    # Ensure return type is list of Python ints
    result = [int(x) for x in best_set]
    return result
