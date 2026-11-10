import time
import random

def _prime_factor_counts(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _check_distinct_products(selected):
    """Check if all 2^k subset products are distinct."""
    if not selected:
        return True
    k = len(selected)
    if k > 18:
        return False
    products = set()
    products.add(1)
    for mask in range(1, 1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= selected[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def _can_add_fast_vector(selected_vecs, cand_vec):
    """
    Fast check: can cand_vec be added to selected_vecs without creating duplicate subset sums?
    
    This is equivalent to checking if the set of all subset sums of selected_vecs 
    is disjoint from the set of all subset sums of selected_vecs + cand_vec (shifted by cand_vec).
    
    Specifically, a collision occurs if there exist two distinct subsets A, B of the new set 
    such that sum(A) == sum(B).
    This is equivalent to checking if there exist disjoint subsets A, B of the old set 
    such that sum(A) - sum(B) == cand_vec (or -cand_vec).
    
    Actually, a simpler check:
    Let S be the set of all subset sums of selected_vecs.
    The new set of subset sums is S union {s + cand_vec for s in S}.
    A collision exists if S intersects {s + cand_vec for s in S}.
    i.e., exists s1, s2 in S such that s1 = s2 + cand_vec => s1 - s2 = cand_vec.
    
    So we need to check if cand_vec is in the set of differences {s1 - s2 : s1, s2 in S}.
    
    For small sets (k <= 18), we can compute S and check differences.
    However, computing all differences is O(|S|^2) which is 2^(2k).
    
    A better way:
    We just need to check if adding cand_vec creates a collision.
    This is true if and only if there exist s1, s2 in S such that s1 - s2 = cand_vec.
    Since S is a set of vectors, we can iterate through all pairs? No, too slow.
    
    Alternative:
    We can just generate the new set of sums S' = S union {s + cand_vec for s in S} and check if len(S') == 2 * len(S).
    If len(S') < 2 * len(S), there was a collision.
    
    This is O(|S|) to generate and O(|S|) to check uniqueness (using a set).
    |S| is at most 2^k. For k=18, 2^18 = 262,144. This is very fast.
    """
    if not selected_vecs:
        return True
    
    # Generate all subset sums of selected_vecs
    sums = {tuple([0]*len(cand_vec))} # Zero vector
    for v in selected_vecs:
        new_sums = set()
        for s in sums:
            new_s = tuple(s[i] + v[i] for i in range(len(v)))
            new_sums.add(new_s)
        sums.update(new_sums)
    
    # Check if adding cand_vec creates a collision
    # New sums would be sums union {s + cand_vec for s in sums}
    # Collision if (s + cand_vec) is already in sums for any s in sums
    # OR if s1 + cand_vec == s2 + cand_vec (impossible since s1 != s2)
    # OR if s + cand_vec == s' (where s, s' in sums)
    
    for s in sums:
        new_s = tuple(s[i] + cand_vec[i] for i in range(len(cand_vec)))
        if new_s in sums:
            return False
            
    return True

def _local_search(current_set, candidates, deadline, precomputed_factors, prime_set):
    """Try to improve current_set via swap operations."""
    k = len(current_set)
    if k >= 18:
        return current_set
    
    current_set = list(current_set)
    
    # Precompute vectors for current set
    # Determine dimension based on primes involved
    # For consistency, let's use the first 10 primes or all primes in 2..40
    # Primes in 2..40: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 (12 primes)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_index = {p: i for i, p in enumerate(primes)}
    d = len(primes)
    
    def to_vec(n):
        factors = precomputed_factors[n]
        v = [0] * d
        for p, e in factors.items():
            v[prime_index[p]] = e
        return v
        
    current_vecs = [to_vec(x) for x in current_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_set)):
            if time.time() > deadline:
                break
            removed = current_set[i]
            remaining = current_set[:i] + current_set[i+1:]
            remaining_vecs = current_vecs[:i] + current_vecs[i+1:]
            
            for cand in candidates:
                if cand in current_set:
                    continue
                cand_vec = to_vec(cand)
                if _can_add_fast_vector(remaining_vecs, cand_vec):
                    new_set = remaining + [cand]
                    return new_set
    return current_set

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 55  # Safety margin
    
    candidates = list(range(2, 41))
    
    # Precompute prime factorizations
    precomputed_factors = {n: _prime_factor_counts(n) for n in candidates}
    
    # Primes in range
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_index = {p: i for i, p in enumerate(primes)}
    d = len(primes)
    
    def to_vec(n):
        factors = precomputed_factors[n]
        v = [0] * d
        for p, e in factors.items():
            v[prime_index[p]] = e
        return v
        
    # Strategy 1: Start with a known valid core (Inspiration 1)
    # Powers of 2: 2, 4, 16 (exponents 1, 2, 4)
    # Powers of 3: 3, 9 (exponents 1, 2)
    # Powers of 5: 5, 25 (exponents 1, 2)
    # Prime 7
    # Other primes: 11, 13, 17, 19, 23, 29, 31, 37
    core = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_set = core[:]
    best_len = len(core)
    
    # Verify core is valid (it should be)
    if not _check_distinct_products(best_set):
        # Fallback to just primes if core is somehow broken
        best_set = primes[:]
        best_len = len(best_set)
        
    # Try to add remaining candidates to the core
    # Remaining: 6, 8, 10, 12, 14, 15, 18, 20, 21, 22, 24, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39, 40
    remaining_candidates = [c for c in candidates if c not in best_set]
    
    # Greedy add
    current_set = best_set[:]
    current_vecs = [to_vec(x) for x in current_set]
    
    for cand in remaining_candidates:
        if time.time() > deadline:
            break
        if best_len >= 18:
            break
        cand_vec = to_vec(cand)
        if _can_add_fast_vector(current_vecs, cand_vec):
            current_set.append(cand)
            current_vecs.append(cand_vec)
            if len(current_set) > best_len:
                best_len = len(current_set)
                best_set = current_set[:]
                
    # If we didn't reach 18, try local search
    if best_len < 18 and time.time() < deadline:
        improved_set = _local_search(current_set, candidates, deadline, precomputed_factors, primes)
        if len(improved_set) > best_len:
            best_len = len(improved_set)
            best_set = improved_set[:]
            
    # If still < 18, try random restarts with the fast vector check
    if best_len < 18 and time.time() < deadline:
        for _ in range(5000):
            if time.time() > deadline:
                break
            if best_len >= 18:
                break
                
            shuffled = candidates[:]
            random.shuffle(shuffled)
            
            current_set = []
            current_vecs = []
            
            for cand in shuffled:
                if time.time() > deadline:
                    break
                cand_vec = to_vec(cand)
                if _can_add_fast_vector(current_vecs, cand_vec):
                    current_set.append(cand)
                    current_vecs.append(cand_vec)
                    
                    if len(current_set) > best_len:
                        best_len = len(current_set)
                        best_set = current_set[:]
                        
                        if best_len >= 18:
                            break
            
            if best_len >= 18:
                break
                
            # Local search on this random set
            if len(current_set) >= best_len - 1: # Only search if promising
                improved_set = _local_search(current_set, candidates, deadline, precomputed_factors, primes)
                if len(improved_set) > best_len:
                    best_len = len(improved_set)
                    best_set = improved_set[:]
                    
    # Final validation
    if best_set and not _check_distinct_products(best_set):
        best_set = primes[:] # Fallback to safe set
        
    if not best_set:
        best_set = primes[:]
        
    return [int(x) for x in best_set]
