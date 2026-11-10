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
    Collision iff exists s1, s2 in S (subset sums) with s1 - s2 == cand_vec.
    Equivalent: (s + cand_vec) in S for some s in S.
    """
    if not selected_vecs:
        return True
    
    # Generate all subset sums of selected_vecs
    d = len(cand_vec)
    sums = {tuple([0]*d)}
    for v in selected_vecs:
        new_sums = set()
        for s in sums:
            new_s = tuple(s[i] + v[i] for i in range(d))
            new_sums.add(new_s)
        sums.update(new_sums)
    
    # Check if adding cand_vec creates a collision
    for s in sums:
        new_s = tuple(s[i] + cand_vec[i] for i in range(d))
        if new_s in sums:
            return False
            
    return True

def _can_add_with_sums(sums, cand_vec):
    """
    Check if cand_vec can be added given precomputed subset sums.
    Returns (ok, new_sums) where new_sums is the updated set if ok.
    """
    d = len(cand_vec)
    # Check collision: exists s in sums with s + cand_vec in sums
    for s in sums:
        new_s = tuple(s[i] + cand_vec[i] for i in range(d))
        if new_s in sums:
            return False, None
    # Build new sums
    new_sums = set(sums)
    for s in sums:
        new_s = tuple(s[i] + cand_vec[i] for i in range(d))
        new_sums.add(new_s)
    return True, new_sums

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
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
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
    
    # Known valid 18-element core:
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 (12)
    # Powers of 2: 4=2^2, 8=2^3, 16=2^4 (3)
    # Powers of 3: 9=3^2, 27=3^3 (2)
    # Powers of 5: 25=5^2 (1)
    # Total: 12 + 3 + 2 + 1 = 18
    core = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 4, 8, 16, 9, 27, 25]
    
    best_set = core[:]
    best_len = len(core)
    
    # Verify core is valid (it should be)
    if not _check_distinct_products(best_set):
        # Fallback to just primes if core is somehow broken
        best_set = primes[:]
        best_len = len(best_set)
    
    # If we already have 18, return immediately (it's the max possible)
    if best_len >= 18:
        return [int(x) for x in best_set]
        
    # Try to add remaining candidates to the core
    remaining_candidates = [c for c in candidates if c not in best_set]
    
    # Greedy add with precomputed sums for speed
    current_set = best_set[:]
    current_vecs = [to_vec(x) for x in current_set]
    
    # Precompute subset sums for current set
    def compute_sums(vecs):
        if not vecs:
            return {tuple([0]*d)}
        sums = {tuple([0]*d)}
        for v in vecs:
            new_sums = set()
            for s in sums:
                new_s = tuple(s[i] + v[i] for i in range(d))
                new_sums.add(new_s)
            sums.update(new_sums)
        return sums
    
    current_sums = compute_sums(current_vecs)
    
    for cand in remaining_candidates:
        if time.time() > deadline:
            break
        if best_len >= 18:
            break
        cand_vec = to_vec(cand)
        ok, new_sums = _can_add_with_sums(current_sums, cand_vec)
        if ok:
            current_set.append(cand)
            current_vecs.append(cand_vec)
            current_sums = new_sums
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
            current_sums = {tuple([0]*d)}
            
            for cand in shuffled:
                if time.time() > deadline:
                    break
                cand_vec = to_vec(cand)
                ok, new_sums = _can_add_with_sums(current_sums, cand_vec)
                if ok:
                    current_set.append(cand)
                    current_vecs.append(cand_vec)
                    current_sums = new_sums
                    
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
