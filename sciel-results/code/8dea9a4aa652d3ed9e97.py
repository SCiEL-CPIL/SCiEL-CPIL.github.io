"""Rewritten program: maximize set size under no-three-distinct-product-divisibility."""
import time
from math import isqrt, gcd
from itertools import combinations

P = {'n': 256}

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    N = P['n']
    elems = list(range(2, N + 1))  # 2..256
    M = len(elems)  # 255 elements
    
    # Precompute: for each pair (i,j) with i != j, whether elems[i] divides elems[j]*elems[k] for any k != i,j
    # Actually, we need: for triple (a,b,c) distinct, a !| b*c
    # Precompute conflict[a][b][c] but that's 255^3 = too much
    # Instead: for each a, and each pair (b,c), check a | b*c
    # Better: for each a, precompute which pairs (b,c) have a | b*c
    # Even better: for each a, and each b, compute the set of c such that a | b*c
    # i.e., c must be a multiple of a/gcd(a,b)
    
    # Precompute gcd table
    gcd_table = [[0] * M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            gcd_table[i][j] = gcd(elems[i], elems[j])
    
    # For each a (index), and each b (index), the condition a | b*c means:
    # a / gcd(a,b) | c
    # So c must be a multiple of (a // gcd(a,b))
    # Precompute: for each a, for each b, the set of c (as bitmask) such that a | b*c
    # But 255^2 bitmasks of 255 bits each = 255*255*32 bytes ~ 2MB, fine
    
    # Actually, let's think about this differently for branch-and-bound.
    # We want to select a subset S maximizing |S| such that for all distinct a,b,c in S: a !| b*c.
    
    # Key observation: if we fix which elements are IN the set, the constraint is:
    # For each a in S, for each pair (b,c) in S\{a}, a !| b*c.
    # Equivalently: for each a in S, no pair (b,c) from S\{a} satisfies a | b*c.
    
    # Let's precompute for each a (value), the set of pairs (b,c) where a | b*c.
    # For branch and bound, when we have a partial set P and candidate set C:
    # - If we add x to P, we need to check:
    #   1. x !| b*c for all pairs (b,c) in P (x as divisor)
    #   2. For each a in P, a !| x*c for all c in P (a as divisor, x and c as the pair)
    #   3. For each a in P, a !| x*b for all b in P (same as 2, symmetric)
    #   Wait, (2) and (3) are the same condition: for each a in P and each c in P\{a}, a !| x*c
    #   But also we need: for each pair (b,c) in P, and a=x: x !| b*c (that's condition 1)
    #   And for each a in P, and each c in P\{a}: a !| x*c (condition 2)
    
    # For efficient checking, precompute for each value a:
    #   divs[a] = set of values v such that a | v  (v is a multiple of a)
    # Then a | b*c iff (b*c) is a multiple of a.
    # b*c is a multiple of a iff c is a multiple of a/gcd(a,b).
    
    # Let's precompute: for each a_idx, b_idx, the threshold t = a // gcd(a,b)
    # Then a | b*c iff c is a multiple of t.
    
    # Precompute multiples sets
    # mult_set[d] = set of indices i where elems[i] is a multiple of d
    max_val = N
    mult_set = [set() for _ in range(max_val + 1)]
    for i, v in enumerate(elems):
        for d in range(1, v + 1):
            if v % d == 0:
                mult_set[d].add(i)
    
    # Precompute threshold: thresh[a_idx][b_idx] = elems[a_idx] // gcd(elems[a_idx], elems[b_idx])
    thresh = [[0] * M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            thresh[i][j] = elems[i] // gcd_table[i][j]
    
    # Now for branch-and-bound:
    # State: selected set (as set of indices), available set (as set of indices)
    # When adding x to selected:
    #   Check 1: for all pairs (b,c) in selected: x !| b*c
    #     x | b*c iff b*c is multiple of x iff c is multiple of x/gcd(x,b)
    #     So: for all b in selected, c in selected, c != b: c not in mult_set[x//gcd(x,b)]
    #   Check 2: for all a in selected, for all c in selected, c != a: a !| x*c
    #     a | x*c iff x*c is multiple of a iff x is multiple of a/gcd(a,c)... 
    #     Wait: a | x*c iff (x*c) % a == 0 iff c * (x % a) ... 
    #     a | x*c iff x*c is divisible by a. Let g = gcd(a, x). Then a/g | c.
    #     So: for all a in selected, for all c in selected, c != a: c not in mult_set[a//gcd(a,x)]
    
    # This is still O(n^2) per addition where n = |selected|. With n up to ~100, that's 10000 ops per add.
    # For branch and bound with depth ~100, this could be slow.
    
    # Alternative: use the pairwise conflict graph approach from the directive.
    # Define: x and y are "conflicting" if there exists z in [2,256] such that x | y*z.
    # This is a NECESSARY condition for (x,y) to be in a valid set (with some z).
    # But it's not sufficient - we need z to also be in S.
    # However, if we build the graph where edge(x,y) means "exists z: x|y*z", then any independent set
    # in this graph is valid for our problem. But it's not the maximum valid set.
    
    # Actually wait - the constraint is: for ALL distinct a,b,c in S, a !| b*c.
    # If we define conflict(x,y) = exists z in [2,256] such that x | y*z, then:
    # If x,y are in S and conflict(x,y), there EXISTS some z with x|y*z.
    # But that z might not be in S. So conflict(x,y) is NOT a valid pairwise constraint.
    
    # The directive's approach is too aggressive. Let me use a different strategy.
    
    # Better approach: Greedy + local search with efficient checking.
    # Precompute for fast can_add checks.
    
    # For each value v, precompute the list of (b_val, c_val) pairs where v | b*c, 
    # but that's too many. Instead, for can_add(x, S):
    #   Part 1: x !| b*c for all b,c in S, b!=c
    #     => for each b in S: check that no c in S (c!=b) has c in mult_set[x//gcd(x,b)]
    #   Part 2: for each a in S: a !| x*c for all c in S, c!=a  
    #     => for each a in S: check that no c in S (c!=a) has c in mult_set[a//gcd(a,x)]
    
    # Let's implement with sets for fast membership.
    
    # Precompute primes for initial set
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(N) + 1):
        if sieve[i]:
            for j in range(i*i, N+1, i):
                sieve[j] = False
    primes = [i for i in range(2, N+1) if sieve[i]]
    composites = [i for i in range(2, N+1) if not sieve[i]]
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(elems)}
    
    # Precompute gcd for value pairs (faster than index pairs since we work with values)
    # Actually let's just use math.gcd directly, it's fast enough.
    
    def can_add(x_val, S_vals):
        """Check if x_val can be added to set S_vals (set of values)."""
        S_list = list(S_vals)
        n = len(S_list)
        
        # Part 1: x !| b*c for all distinct b,c in S
        for i in range(n):
            b = S_list[i]
            t = x_val // gcd(x_val, b)
            if t > N:
                continue  # no c in [2,256] can be a multiple of t if t > 256
            # Check if any c in S (c != b) is a multiple of t
            for j in range(n):
                if i == j:
                    continue
                c = S_list[j]
                if c % t == 0:
                    return False
        
        # Part 2: for each a in S, a !| x*c for all c in S, c != a
        for i in range(n):
            a = S_list[i]
            t = a // gcd(a, x_val)
            if t > N:
                continue
            for j in range(n):
                if i == j:
                    continue
                c = S_list[j]
                if c % t == 0:
                    return False
        
        return True
    
    # Start with all primes
    best_set = set(primes)
    
    # Sort composites by "safety" - fewer prime factors = safer
    def num_prime_factors(x):
        count = 0
        temp = x
        for p in primes:
            if p > temp:
                break
            if temp % p == 0:
                count += 1
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            count += 1
        return count
    
    composites_sorted = sorted(composites, key=lambda x: (num_prime_factors(x), x))
    
    # Greedy add
    for x in composites_sorted:
        if time.time() > deadline:
            break
        if can_add(x, best_set):
            best_set.add(x)
    
    best_size = len(best_set)
    
    # Local search: try removing one element and adding up to 2 others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = sorted(best_set)
        for i in range(len(S_list)):
            if time.time() > deadline:
                break
            candidate = S_list[i]
            new_set = best_set - {candidate}
            added = 0
            for x in composites_sorted:
                if time.time() > deadline:
                    break
                if x in new_set:
                    continue
                if can_add(x, new_set):
                    new_set.add(x)
                    added += 1
                    if added >= 2:
                        break
            if added > 1:
                best_set = new_set
                improved = True
                best_size = len(best_set)
            elif added == 1 and len(new_set) > best_size:
                best_set = new_set
                best_size = len(best_set)
    
    # Second pass: try removing two elements and adding three
    if time.time() < deadline:
        S_list = sorted(best_set)
        n = len(S_list)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                if time.time() > deadline:
                    break
                new_set = best_set - {S_list[i], S_list[j]}
                added = 0
                for x in composites_sorted:
                    if time.time() > deadline:
                        break
                    if x in new_set:
                        continue
                    if can_add(x, new_set):
                        new_set.add(x)
                        added += 1
                        if added >= 3:
                            break
                if len(new_set) > best_size:
                    best_set = new_set
                    best_size = len(new_set)
                    improved = True
    
    result = sorted(best_set)
    return [int(x) for x in result]
