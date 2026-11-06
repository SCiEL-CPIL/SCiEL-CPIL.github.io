import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    limit = 512
    n = limit
    
    # Precompute prime factorizations
    def factorize(x):
        factors = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors[d] = factors.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            factors[x] = factors.get(x, 0) + 1
        return factors
    
    # Precompute lcm compatibility
    comp = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            if i * j // gcd(i, j) <= limit:
                comp[i].add(j)
                if i != j:
                    comp[j].add(i)
    
    # Precompute compatibility as bitmasks for fast intersection
    adj_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in comp[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    # Precompute "rarity" - number of compatible elements
    rarity = [len(comp[i]) for i in range(n + 1)]
    
    # Sort vertices by rarity (most constraining first) for branching
    vertices_by_rarity = sorted(range(1, n + 1), key=lambda x: rarity[x])
    
    # Key insight: elements that are divisors of other elements create structure
    # If a | b, then any element compatible with b is also compatible with a
    # (since lcm(a, x) divides lcm(b, x) when a|b)
    # So selecting b "unlocks" more options than selecting a
    
    # Group by highest prime power
    def highest_prime_power(x):
        """Return (p, e) where p^e is the highest prime power dividing x"""
        best = (0, 0)
        for p, e in factorize(x).items():
            if (p, e) > best:
                best = (p, e)
        return best
    
    # Build poset: a dominates b if a | b and a != b
    # Selecting dominated elements is generally worse
    
    best_clique = set()
    best_size = 0
    
    def is_compatible_with_all(S, x):
        """Check if x is compatible with all elements in S"""
        for y in S:
            if y not in comp[x]:
                return False
        return True
    
    def upper_bound(S, candidates):
        """
        Compute an upper bound on how many more elements can be added.
        Uses prime-exponent capacity argument:
        For each prime p, count how many elements in candidates have p^e 
        for the maximum e seen in S or candidates.
        """
        if not candidates:
            return 0
        
        # Simple bound: count candidates compatible with all of S
        count = 0
        for x in candidates:
            if is_compatible_with_all(S, x):
                count += 1
        return count
    
    def branch_and_bound(S, candidates, depth=0):
        nonlocal best_clique, best_size
        
        # Time check
        if time.time() > deadline - 5:
            return
        
        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_clique = set(S)
        
        # If no candidates, return
        if not candidates:
            return
        
        # Filter candidates to those compatible with all of S
        valid_candidates = [x for x in candidates if is_compatible_with_all(S, x)]
        if not valid_candidates:
            return
        
        # Upper bound: if even adding all valid candidates doesn't beat best, prune
        if len(S) + len(valid_candidates) <= best_size:
            return
        
        # Order candidates by rarity (most constraining first)
        valid_candidates.sort(key=lambda x: rarity[x])
        
        # Branch on the most constraining candidate
        x = valid_candidates[0]
        
        # Option 1: Include x
        new_S = set(S)
        new_S.add(x)
        new_candidates = [y for y in valid_candidates if y != x and y in comp[x]]
        branch_and_bound(new_S, new_candidates, depth + 1)
        
        # Option 2: Exclude x
        remaining = [y for y in valid_candidates if y != x]
        branch_and_bound(S, remaining, depth + 1)
    
    # Initial strategy: start with small numbers (they're compatible with more)
    # and build up
    
    # Strategy 1: Greedy by rarity, then B&B
    S = set()
    for x in vertices_by_rarity:
        if is_compatible_with_all(S, x):
            S.add(x)
    if len(S) > best_size:
        best_size = len(S)
        best_clique = set(S)
    
    # Strategy 2: Start with 1 (compatible with all), then B&B
    S = {1}
    candidates = list(range(2, n + 1))
    branch_and_bound(S, candidates)
    
    # Strategy 3: Try starting with different "seeds"
    # Elements with small values tend to be more compatible
    for seed in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]:
        if time.time() > deadline - 15:
            break
        S = {seed}
        candidates = [x for x in range(1, n + 1) if x != seed and x in comp[seed]]
        branch_and_bound(S, candidates)
    
    # Strategy 4: Local search from current best
    def local_search(S):
        S = set(S)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try removing one element and adding up to 2
            for e in list(S):
                if time.time() > deadline - 10:
                    break
                S_minus = S - {e}
                # Try to add elements
                added = 0
                S2 = set(S_minus)
                for x in range(1, n + 1):
                    if time.time() > deadline - 10:
                        break
                    if x not in S2 and is_compatible_with_all(S2, x):
                        S2.add(x)
                        added += 1
                        if added >= 2:
                            break
                if len(S2) > len(S):
                    S = S2
                    improved = True
        return S
    
    if best_clique:
        S = local_search(best_clique)
        if len(S) > best_size:
            best_size = len(S)
            best_clique = S
    
    if not best_clique:
        return [1]
    
    return sorted(best_clique)
