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

def _can_add_fast(selected, cand, cand_factors):
    """
    Fast check: can cand be added to selected without creating duplicate products?
    
    Two subset products collide iff there exist disjoint subsets A, B of the current
    set plus cand such that prod(A) = prod(B * cand) (or similar). This is equivalent
    to: there exists a non-empty subset S of selected such that prod(S) / cand is a
    product of a (possibly empty) subset of selected, OR prod(S) * cand is a product
    of a subset of selected.
    
    More precisely: adding cand creates a collision iff there exist disjoint subsets
    A, B of selected such that prod(A) * cand = prod(B), i.e., prod(A)/prod(B) = 1/cand,
    OR prod(A) = prod(B) * cand, i.e., prod(A)/prod(B) = cand.
    
    So: collision exists iff there's a subset of selected whose product equals cand
    times the product of another disjoint subset of selected.
    
    Equivalently: in the multiplicative group, cand must not be expressible as a
    quotient of two disjoint subset products of selected.
    
    We can check this by: for each subset of selected, compute its product. Then check
    if any product p1 satisfies p1 / cand is also a product of a disjoint subset.
    
    A cleaner formulation: the set of all subset products of selected is P. After adding
    cand, the new set is P ∪ {p * cand : p in P}. Collision iff P ∩ {p*cand : p in P}
    is non-empty, i.e., iff there exist p1, p2 in P with p1 = p2 * cand, i.e., p1/p2 = cand.
    
    So: check if cand is in the set of quotients {p1/p2 : p1, p2 in P, p1 divisible by p2}.
    
    Even better: check if there exist disjoint A, B with prod(A) = prod(B) * cand.
    This means prod(A) / prod(B) = cand. Since A and B are disjoint, we can think of
    assigning each element a coefficient in {-1, 0, 1} where the product of elements
    with coeff +1 divided by product of elements with coeff -1 equals cand.
    """
    n = len(selected)
    if n == 0:
        return True
    
    # Precompute prime factor vectors for selected
    # We need to check if there exist coefficients c_i in {-1, 0, 1} not all zero
    # such that prod(selected[i]^c_i) = cand
    # i.e., sum_i c_i * log(selected[i]) = log(cand) in the exponent space
    # i.e., for each prime p: sum_i c_i * v_p(selected[i]) = v_p(cand)
    
    # Get primes involved
    all_primes = set(cand_factors.keys())
    for s in selected:
        for p in _prime_factor_counts(s):
            all_primes.add(p)
    
    if len(all_primes) > 15:
        # Too many primes for efficient subset-sum; fall back to slower method
        return _can_add_slow(selected, cand)
    
    primes = sorted(all_primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    d = len(primes)
    
    # Build exponent vectors
    def to_vec(factors):
        v = [0] * d
        for p, e in factors.items():
            v[prime_index[p]] = e
        return tuple(v)
    
    cand_vec = to_vec(cand_factors)
    sel_vecs = [to_vec(_prime_factor_counts(s)) for s in selected]
    
    # We need: sum_i c_i * sel_vecs[i] = cand_vec, where c_i in {-1, 0, 1}
    # Split: let P+ = {i : c_i = 1}, P- = {i : c_i = -1}
    # Then sum_{i in P+} sel_vecs[i] - sum_{i in P-} sel_vecs[i] = cand_vec
    # i.e., sum_{i in P+} sel_vecs[i] = cand_vec + sum_{i in P-} sel_vecs[i]
    
    # Meet-in-the-middle: split selected into two halves
    half = n // 2
    left = sel_vecs[:half]
    right = sel_vecs[half:]
    
    # For each half, compute all possible (sum_vec, used_mask) where each element
    # can be used with coeff -1, 0, or +1
    # Actually, we need to track which indices are used so we can ensure disjointness
    # But since each element appears in at most one of P+ or P-, and we're checking
    # if the equation holds, we can just enumerate all coefficient assignments.
    
    # For n elements, 3^n possibilities. For n up to ~15, that's 14M which is too slow.
    # Use meet-in-the-middle: 3^(n/2) for each half.
    
    # For the left half: enumerate all assignments of {-1, 0, 1} to each element
    # Compute the resulting vector sum
    left_sums = {}  # vector -> set of masks (to check disjointness later, but actually
                    # we just need existence, so we can use a set of vectors)
    # Actually for meet-in-the-middle we need: for each possible sum from left,
    # and each possible sum from right, check if left_sum - right_sum = cand_vec
    # i.e., left_sum = cand_vec + right_sum
    
    # But we also need the assignments to be valid (each element used at most once
    # across both halves, which is automatically satisfied since halves are disjoint).
    
    # Enumerate left half
    def enumerate_assignments(vecs):
        """Yield (sum_vector, used_indices_set) for all {-1,0,1} assignments."""
        m = len(vecs)
        results = []
        for mask in range(3 ** m):
            v = [0] * d
            used = set()
            mm = mask
            for i in range(m):
                coeff = mm % 3
                mm //= 3
                if coeff == 1:
                    for j in range(d):
                        v[j] += vecs[i][j]
                    used.add(i)
                elif coeff == 2:  # represents -1
                    for j in range(d):
                        v[j] -= vecs[i][j]
                    used.add(i)
            results.append((tuple(v), frozenset(used)))
        return results
    
    # For efficiency, just store the set of achievable sum vectors
    left_achievable = set()
    for mask in range(3 ** half):
        v = [0] * d
        mm = mask
        for i in range(half):
            coeff = mm % 3
            mm //= 3
            if coeff == 1:
                for j in range(d):
                    v[j] += left[i][j]
            elif coeff == 2:
                for j in range(d):
                    v[j] -= left[i][j]
        left_achievable.add(tuple(v))
    
    # For the right half, we need to check if cand_vec - right_sum is in left_achievable
    # i.e., for each right assignment, compute right_sum, then check if
    # cand_vec - right_sum is achievable by left.
    # Wait: we need left_sum - right_sum = cand_vec, so left_sum = cand_vec + right_sum.
    
    for mask in range(3 ** (n - half)):
        v = [0] * d
        mm = mask
        for i in range(n - half):
            coeff = mm % 3
            mm //= 3
            if coeff == 1:
                for j in range(d):
                    v[j] += right[i][j]
            elif coeff == 2:
                for j in range(d):
                    v[j] -= right[i][j]
        # We need: left_sum = cand_vec + right_sum
        target = tuple(cand_vec[j] + v[j] for j in range(d))
        if target in left_achievable:
            # Check it's not the trivial solution (all zeros)
            # The trivial solution would be all coefficients 0, giving sum = (0,...,0)
            # We need a non-trivial solution, so we need the overall assignment not to be all zeros.
            # If right_sum = (0,...,0) and cand_vec = (0,...,0), that would be trivial,
            # but cand_vec is never all zeros since cand >= 2.
            # Actually cand_vec has at least one non-zero component, so left_sum = cand_vec + right_sum
            # can only equal (0,...,0) if right_sum = -cand_vec, which is fine (non-trivial).
            # The only truly trivial case is when both left and right assignments are all zeros,
            # giving left_sum = (0,...,0) and we'd need cand_vec = (0,...,0), which never happens.
            return False  # Collision found!
    
    return True  # No collision, safe to add

def _can_add_slow(selected, cand):
    """Slower but always-correct check using product enumeration."""
    products = {1}
    for s in selected:
        new_prods = [p * s for p in products]
        products.update(new_prods)
    for p in products:
        if p * cand in products:
            return False
    return True

def _local_search(current_set, candidates, deadline, precomputed_factors):
    """Try to improve current_set via swap operations."""
    k = len(current_set)
    if k >= 18:
        return current_set
    
    current_set = list(current_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set_copy = list(current_set)
        for i in range(len(current_set_copy)):
            if time.time() > deadline:
                break
            removed = current_set_copy[i]
            remaining = current_set_copy[:i] + current_set_copy[i+1:]
            
            for cand in candidates:
                if cand in current_set_copy:
                    continue
                if _can_add_fast(remaining, cand, precomputed_factors[cand]):
                    new_set = remaining + [cand]
                    return new_set
        # No swap found in this pass
    return current_set

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all subset products are distinct.
    """
    deadline = time.time() + 55  # Safety margin
    
    candidates = list(range(2, 41))
    
    # Precompute prime factorizations
    precomputed_factors = {n: _prime_factor_counts(n) for n in candidates}
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Greedy with random restarts + local search
    num_restarts = 50000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        if best_len >= 18:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        
        for cand in shuffled:
            if time.time() > deadline:
                break
            
            if _can_add_fast(current_set, cand, precomputed_factors[cand]):
                current_set.append(cand)
                
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]
                    
                    if best_len >= 18:
                        break
        
        if best_len >= 18:
            break
        
        # Local search: try to improve via swaps
        if len(current_set) >= best_len and len(current_set) < 18:
            improved_set = _local_search(
                current_set, candidates, deadline, precomputed_factors
            )
            if len(improved_set) > best_len:
                best_len = len(improved_set)
                best_set = improved_set[:]
                if best_len >= 18:
                    break
    
    # Strategy 2: Deterministic orderings
    if best_len < 18 and time.time() < deadline:
        def prime_factor_count(n):
            count = 0
            for e in precomputed_factors[n].values():
                count += e
            return count
        
        for order_key in [
            lambda x: (prime_factor_count(x), x),
            lambda x: (-prime_factor_count(x), x),
            lambda x: (x % 2, prime_factor_count(x), x),
            lambda x: (len(precomputed_factors[x]), x),
            lambda x: (x, ),
            lambda x: (-x, ),
        ]:
            if time.time() > deadline or best_len >= 18:
                break
            
            ordered = sorted(candidates, key=order_key)
            current_set = []
            
            for cand in ordered:
                if time.time() > deadline:
                    break
                if _can_add_fast(current_set, cand, precomputed_factors[cand]):
                    current_set.append(cand)
                    
                    if len(current_set) > best_len:
                        best_len = len(current_set)
                        best_set = current_set[:]
                        if best_len >= 18:
                            break
            
            if best_len >= 18:
                break
            
            if len(current_set) >= best_len and len(current_set) < 18:
                improved_set = _local_search(
                    current_set, candidates, deadline, precomputed_factors
                )
                if len(improved_set) > best_len:
                    best_len = len(improved_set)
                    best_set = improved_set[:]
    
    # Final validation
    if best_set and not _check_distinct_products(best_set):
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    if not best_set:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_set]
