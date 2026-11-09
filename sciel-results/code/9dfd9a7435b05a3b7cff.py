"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _sieve_prime_factors(max_n):
    """Precompute distinct prime factors for all numbers up to max_n using sieve."""
    factors = [set() for _ in range(max_n + 1)]
    for d in range(2, max_n + 1):
        if not factors[d]:
            for m in range(d, max_n + 1, d):
                factors[m].add(d)
    return factors

def _count_distinct_primes(arr, pf):
    """Count distinct primes dividing all pairwise sums using precomputed factors."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= pf[s]
    return len(primes)

def _find_clique_bitset(candidates, compat_bitsets, target_size, deadline):
    """
    Bitset-based clique search.
    candidates: list of candidate numbers
    compat_bitsets: list of int, compat_bitsets[i] = bitmask of indices j where compatible
    Returns a list of indices forming a clique, or None.
    """
    n = len(candidates)
    if n < target_size:
        return None
    
    def _popcount(x):
        return bin(x).count('1')
    
    def backtrack(current, candidates_mask):
        if time.time() > deadline:
            return None
        if len(current) == target_size:
            return current[:]
        
        # Pruning: not enough candidates left
        if _popcount(candidates_mask) < (target_size - len(current)):
            return None
        
        # Extract candidate indices from mask
        cand_indices = []
        mask = candidates_mask
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            cand_indices.append(idx)
            mask ^= lsb
        
        for i in cand_indices:
            if time.time() > deadline:
                return None
            # Intersect current's compatibility with this node's compatibility
            new_mask = candidates_mask & compat_bitsets[i]
            # Only consider nodes after i to avoid duplicates
            # Create a mask for nodes > i
            higher_mask = 0
            for j in range(i + 1, n):
                higher_mask |= (1 << j)
            new_mask &= higher_mask
            
            current.append(i)
            result = backtrack(current, new_mask)
            if result is not None:
                return result
            current.pop()
        return None
    
    # Greedy initialization using bitsets
    degrees = [_popcount(compat_bitsets[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Try greedy first
    current = []
    current_mask = (1 << n) - 1
    for i in order:
        if time.time() > deadline:
            break
        if not (current_mask & (1 << i)):
            continue
        # Check compatibility with all in current using bitset intersection
        compat_all = (1 << n) - 1
        for c in current:
            compat_all &= compat_bitsets[c]
        if not (compat_all & (1 << i)):
            continue
        current.append(i)
        current_mask &= compat_bitsets[i]
        if len(current) == target_size:
            return current
    
    # Backtracking from greedy start
    if len(current) < target_size:
        # Build mask for nodes not yet considered
        used_mask = 0
        for c in current:
            used_mask |= (1 << c)
        remaining_mask = current_mask & ~used_mask
        result = backtrack(current, remaining_mask)
        if result is not None:
            return result
    
    # Try different starting points
    for start_i in order[:80]:
        if time.time() > deadline:
            break
        current = [start_i]
        initial_mask = compat_bitsets[start_i]
        result = backtrack(current, initial_mask)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Precompute prime factors for all sums (max sum = 20000)
    pf = _sieve_prime_factors(20000)
    
    best_arr = list(range(1, 13))
    best_count = _count_distinct_primes(best_arr, pf)
    
    # Target prime sets to try, ordered by preference (fewer primes first)
    target_prime_sets = [
        {2, 3, 5, 7},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {2, 3, 5, 11},
        {2, 3, 7, 11},
        {2, 5, 7, 11},
        {2, 3, 5, 7, 11},
    ]
    
    for target_primes in target_prime_sets:
        if time.time() > deadline:
            break
        
        # Filter candidates: numbers 1..10000 where every pairwise sum with another candidate
        # has all prime factors in target_primes
        # First, precompute which numbers have sums with only target primes
        # For a number x, we need: for all y in our set, x+y has only primes from target_primes
        
        # Build candidate list: all numbers 1..10000
        candidates = list(range(1, 10001))
        
        # For efficiency, precompute for each number x, which other numbers y make x+y
        # have only target prime factors
        # This is expensive for 10000 candidates, so let's be smarter
        
        # Key insight: if x and y are both ≡ r mod p for each p in target_primes,
        # then x+y ≡ 2r mod p. We need 2r ≡ 0 mod p for all p, OR x+y is a product of
        # only target primes.
        
        # Better approach: directly check compatibility using precomputed factors
        # But 10000 choose 2 is too many. Let's use a smarter filtering.
        
        # Strategy: group numbers by their residue pattern modulo small primes in target
        # For each prime p in target_primes, x+y must be divisible by p OR x+y has no factor p
        # Actually, we just need all prime factors of x+y to be in target_primes
        
        # Let's precompute a "compatible" check function
        def is_compatible(x, y):
            s = x + y
            if s > 20000:
                return False
            return pf[s].issubset(target_primes)
        
        # Greedy construction with backtracking
        # Start with a promising subset: numbers that are multiples of small products
        # Try numbers that are 0 mod M where M = product of some target primes
        
        target_primes_list = sorted(target_primes)
        
        # Try different modulus-based candidate subsets
        moduli_to_try = []
        # Single primes
        for p in target_primes_list:
            moduli_to_try.append(p)
        # Products of 2
        for i in range(len(target_primes_list)):
            for j in range(i+1, len(target_primes_list)):
                m = target_primes_list[i] * target_primes_list[j]
                if m <= 10000:
                    moduli_to_try.append(m)
        # Products of 3
        for i in range(len(target_primes_list)):
            for j in range(i+1, len(target_primes_list)):
                for k in range(j+1, len(target_primes_list)):
                    m = target_primes_list[i] * target_primes_list[j] * target_primes_list[k]
                    if m <= 10000:
                        moduli_to_try.append(m)
        
        found = False
        for mod in moduli_to_try:
            if time.time() > deadline:
                found = True
                break
            if mod > 10000:
                continue
            
            # Numbers ≡ 0 mod mod
            subset = [v for v in range(mod, 10001, mod)]
            if len(subset) < 12:
                continue
            
            # Build compatibility within this subset using bitsets
            n_sub = len(subset)
            # Limit to first 300 candidates for tractability with bitsets
            if n_sub > 300:
                subset = subset[:300]
                n_sub = 300
            
            # Build bitset compatibility
            compat_bitsets = [0] * n_sub
            for i in range(n_sub):
                for j in range(i+1, n_sub):
                    if is_compatible(subset[i], subset[j]):
                        compat_bitsets[i] |= (1 << j)
                        compat_bitsets[j] |= (1 << i)
            
            clique = _find_clique_bitset(subset, compat_bitsets, 12, deadline)
            if clique is not None:
                arr = [subset[i] for i in clique]
                count = _count_distinct_primes(arr, pf)
                if count < best_count:
                    best_count = count
                    best_arr = arr
                    if best_count <= 4:
                        return best_arr
                found = True
                break
        
        if found or time.time() > deadline:
            break
    
    # Fallback: try a broader search if we haven't found a good solution
    if best_count > 4 and time.time() < deadline:
        import random
        random.seed(42)
        for trial in range(50):
            if time.time() > deadline:
                break
            # Random set with structure
            mod = random.choice([2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 30, 42, 60, 70, 105, 210, 420, 840])
            r = random.randint(0, mod - 1)
            start = r % mod
            if start == 0:
                start = mod
            arr = []
            val = start
            while val <= 10000 and len(arr) < 12:
                arr.append(val)
                val += mod
            if len(arr) < 12:
                while len(arr) < 12:
                    x = random.randint(1, 10000)
                    if x not in arr:
                        arr.append(x)
            arr = arr[:12]
            count = _count_distinct_primes(arr, pf)
            if count < best_count:
                best_count = count
                best_arr = arr
                if best_count <= 4:
                    return best_arr
    
    # Additional fallback: try more target prime sets with bitset clique search
    if best_count > 4 and time.time() < deadline:
        additional_prime_sets = [
            {2, 3, 5},
            {2, 3, 7},
            {2, 3, 11},
            {2, 5, 11},
            {3, 5, 7, 11},
        ]
        for target_primes in additional_prime_sets:
            if time.time() > deadline:
                break
            
            def is_compatible2(x, y):
                s = x + y
                if s > 20000:
                    return False
                return pf[s].issubset(target_primes)
            
            target_primes_list = sorted(target_primes)
            moduli_to_try2 = []
            for p in target_primes_list:
                moduli_to_try2.append(p)
            for i in range(len(target_primes_list)):
                for j in range(i+1, len(target_primes_list)):
                    m = target_primes_list[i] * target_primes_list[j]
                    if m <= 10000:
                        moduli_to_try2.append(m)
            
            for mod in moduli_to_try2:
                if time.time() > deadline:
                    break
                if mod > 10000:
                    continue
                
                subset = [v for v in range(mod, 10001, mod)]
                if len(subset) < 12:
                    continue
                
                n_sub = len(subset)
                if n_sub > 300:
                    subset = subset[:300]
                    n_sub = 300
                
                compat_bitsets2 = [0] * n_sub
                for i in range(n_sub):
                    for j in range(i+1, n_sub):
                        if is_compatible2(subset[i], subset[j]):
                            compat_bitsets2[i] |= (1 << j)
                            compat_bitsets2[j] |= (1 << i)
                
                clique = _find_clique_bitset(subset, compat_bitsets2, 12, deadline)
                if clique is not None:
                    arr = [subset[i] for i in clique]
                    count = _count_distinct_primes(arr, pf)
                    if count < best_count:
                        best_count = count
                        best_arr = arr
                        if best_count <= 4:
                            return best_arr
                    break
    
    return best_arr

# EVOLVE-BLOCK-END
