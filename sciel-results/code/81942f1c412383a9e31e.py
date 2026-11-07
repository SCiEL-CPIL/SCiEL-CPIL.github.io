"""Combined approach: greedy construction + local search with swap moves for distinct subset products."""

# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

# Primes <= 40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NUM_PRIMES = len(_PRIMES)

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

# Precompute prime-exponent vectors for all numbers 2..40
def _factorize(n):
    """Return list of exponents for primes in _PRIMES."""
    vec = [0] * _NUM_PRIMES
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return vec

# Precompute vectors for 2..40
_VECTORS = {}
for x in range(2, 41):
    _VECTORS[x] = _factorize(x)

def _check_distinct_subset_products(vecs):
    """
    Check if all 2^k subset products are distinct.
    Equivalent: no non-empty subset of {±v_i} sums to zero.
    For small k (<=18), we can check by computing all subset sums of the vectors
    and verifying they're all distinct.
    
    Actually, a cleaner check: compute all 2^k subset-sum vectors and check for duplicates.
    But 2^18 = 262144 which is feasible.
    
    Alternative: for each pair of subsets, check if they have the same sum.
    But that's O(4^k) which is too slow.
    
    Best approach: compute all subset sums (as tuples) and put in a set.
    2^18 = 262144 subset sums, each a tuple of 12 ints. This is fast in Python.
    """
    k = len(vecs)
    if k == 0:
        return True
    if k == 1:
        return True
    
    # Compute all subset sums using iterative doubling
    # Start with {zero vector}, then for each vector v, add v to all existing sums
    sums = {tuple([0] * _NUM_PRIMES)}
    for v in vecs:
        # For each existing sum, create a new sum with v added
        new_sums = set()
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(_NUM_PRIMES))
            new_sums.add(ns)
        sums |= new_sums
    
    return len(sums) == (1 << k)

def _check_distinct_fast(vecs):
    """
    Faster check using the fact that subset products are distinct iff
    the vectors are linearly independent over the integers in a specific sense.
    
    Actually, let's use a different approach: 
    Two subsets A, B have the same product iff sum_{i in A} v_i = sum_{i in B} v_i
    iff sum_{i in A\\B} v_i = sum_{i in B\\A} v_i
    
    A sufficient and necessary condition: the only solution to
    sum_{i=1}^k c_i * v_i = 0 with c_i in {-1, 0, 1} is c_i = 0 for all i.
    
    For k up to 18, we can enumerate all 3^k combinations... no, 3^18 is too large.
    
    Let's stick with the subset-sum approach but optimize it.
    For k <= 18, 2^k <= 262144, which is very fast.
    """
    k = len(vecs)
    if k <= 1:
        return True
    
    # Use the subset-sum enumeration
    # Represent vectors as tuples for hashing
    vec_tuples = [tuple(v) for v in vecs]
    
    # Iterative subset sum computation
    all_sums = {tuple([0] * _NUM_PRIMES)}
    for i in range(k):
        vi = vec_tuples[i]
        # Add vi to each existing sum
        current_size = len(all_sums)
        # We need to iterate over the current set, but we can't modify while iterating
        # Use a list of current sums
        existing = list(all_sums)
        for s in existing:
            ns = tuple(s[j] + vi[j] for j in range(_NUM_PRIMES))
            all_sums.add(ns)
        # Early exit: if we've already found a duplicate
        if len(all_sums) < (1 << (i + 1)):
            return False
    
    return len(all_sums) == (1 << k)

def _greedy_build(candidates, deadline):
    """
    Greedily build a set by adding candidates in order if they maintain distinctness.
    """
    selected = []
    selected_vecs = []
    
    for c in candidates:
        if time.time() > deadline:
            break
        v = _VECTORS[c]
        test_vecs = selected_vecs + [v]
        if _check_distinct_fast(test_vecs):
            selected.append(c)
            selected_vecs.append(v)
    
    return selected

def _local_search(initial_set, deadline, max_iters=5000):
    """
    Local search: try swap moves (remove one, add another) and add moves.
    """
    current = list(initial_set)
    current_vecs = [_VECTORS[x] for x in current]
    best = list(current)
    best_size = len(current)
    
    in_set = set(current)
    all_candidates = set(range(2, 41))
    
    iters = 0
    while time.time() < deadline and iters < max_iters:
        iters += 1
        improved = False
        
        # Try adding an element not in the set
        for c in sorted(all_candidates - in_set):
            if time.time() > deadline:
                break
            v = _VECTORS[c]
            test_vecs = current_vecs + [v]
            if _check_distinct_fast(test_vecs):
                current.append(c)
                current_vecs.append(v)
                in_set.add(c)
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                improved = True
                break  # Restart after one successful add
        
        if improved:
            continue
        
        # Try swap: remove one element, add another
        for i in range(len(current)):
            if time.time() > deadline:
                break
            removed = current[i]
            # Try adding each candidate not in set
            for c in sorted(all_candidates - in_set):
                if c == removed:
                    continue
                # Test: remove current[i], add c
                test_vecs = current_vecs[:i] + current_vecs[i+1:] + [_VECTORS[c]]
                if _check_distinct_fast(test_vecs):
                    # Perform the swap
                    current[i] = c
                    current_vecs[i] = _VECTORS[c]
                    in_set.remove(removed)
                    in_set.add(c)
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
                    improved = True
                    break
            if improved:
                break
        
        if not improved and iters > 100:
            # Stuck, try a random perturbation
            if len(current) > 2:
                # Remove a random element
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                current.pop(idx)
                current_vecs.pop(idx)
                in_set.remove(removed)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin within 90s limit
    
    # Strategy: multiple randomized restarts with greedy + local search
    best_result = []
    best_size = 0
    
    # First: deterministic greedy with primes first, then composites
    primes = [p for p in range(2, 41) if _is_prime(p)]
    composites = [x for x in range(4, 41) if not _is_prime(x)]
    
    # Deterministic order: primes then composites sorted
    order1 = primes + composites
    result1 = _greedy_build(order1, deadline)
    result1 = _local_search(result1, deadline, max_iters=3000)
    if len(result1) > best_size:
        best_result = result1
        best_size = len(result1)
    
    # Reverse composites
    order2 = primes + sorted(composites, reverse=True)
    result2 = _greedy_build(order2, deadline)
    result2 = _local_search(result2, deadline, max_iters=3000)
    if len(result2) > best_size:
        best_result = result2
        best_size = len(result2)
    
    # Randomized restarts
    rng = random.Random(42)
    remaining_time = deadline - time.time()
    num_restarts = min(20, max(1, int(remaining_time / 3.0)))
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random ordering of all candidates
        candidates = list(range(2, 41))
        rng.shuffle(candidates)
        
        # Greedy build
        result = _greedy_build(candidates, deadline)
        
        # Local search with limited iterations
        per_restart_iters = max(500, 5000 // max(1, num_restarts - r))
        result = _local_search(result, deadline, max_iters=per_restart_iters)
        
        if len(result) > best_size:
            best_result = result
            best_size = len(result)
    
    # Final verification
    if best_result:
        vecs = [_VECTORS[x] for x in best_result]
        if _check_distinct_fast(vecs):
            return list(best_result)
    
    # Fallback: just the primes
    return list(primes)

# EVOLVE-BLOCK-END
