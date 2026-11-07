"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

# Precompute prime exponent vectors for all numbers 2..40
_PRIMES_UP_TO_40 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_PRIME_TO_IDX = {p: i for i, p in enumerate(_PRIMES_UP_TO_40)}
_NUM_PRIMES = len(_PRIMES_UP_TO_40)

def _exponent_vector(n):
    """Return the prime exponent vector for n as a tuple."""
    vec = [0] * _NUM_PRIMES
    for p, idx in _PRIME_TO_IDX.items():
        while n % p == 0:
            vec[idx] += 1
            n //= p
    return tuple(vec)

# Precompute vectors for all candidates
_CANDIDATES = list(range(2, 41))
_VECTORS = {x: _exponent_vector(x) for x in _CANDIDATES}

def _can_add_vectors(selected_vecs, new_vec):
    """
    Check if adding new_vec to selected_vecs keeps all subset sum combinations distinct.
    
    The constraint is: for any two distinct subsets S1, S2 of selected_vecs ∪ {new_vec},
    the sum of vectors in S1 must differ from the sum of vectors in S2.
    
    This is equivalent to: no non-empty subset of selected_vecs ∪ {new_vec} has sum = 0.
    But since all vectors have non-negative components and at least one positive component,
    the only way two subsets have the same sum is if they're identical.
    
    Actually, the correct condition is: all 2^(k+1) subset sums of selected_vecs ∪ {new_vec}
    must be distinct. This means:
    - All existing subset sums are distinct (already guaranteed)
    - For each existing subset sum s, s + new_vec must not equal any existing subset sum
    
    So we need: new_vec is not in the difference set {s1 - s2 | s1, s2 in existing_sums}
    where s1 ≠ s2.
    """
    # Get all existing subset sums
    # This could be expensive, so we'll maintain it incrementally in the search
    # For now, this function assumes we have the set of existing sums
    pass

def _subset_sums_set(vecs):
    """Compute the set of all subset sums for a list of vectors."""
    sums = {tuple([0] * _NUM_PRIMES)}
    for v in vecs:
        new_sums = set()
        for s in sums:
            new_sums.add(tuple(s[i] + v[i] for i in range(_NUM_PRIMES)))
        sums = sums | new_sums
    return sums

def _can_add_with_sums(existing_sums, new_vec):
    """
    Check if adding new_vec keeps all subset sums distinct.
    New sums will be {s + new_vec for s in existing_sums}.
    For validity: none of s + new_vec can collide with existing sums.
    """
    for s in existing_sums:
        new_s = tuple(s[i] + new_vec[i] for i in range(_NUM_PRIMES))
        if new_s in existing_sums:
            return False
    return True

def _add_vector_with_sums(existing_sums, new_vec):
    """Return new set of subset sums after adding new_vec."""
    new_sums = set(existing_sums)
    for s in existing_sums:
        new_s = tuple(s[i] + new_vec[i] for i in range(_NUM_PRIMES))
        new_sums.add(new_s)
    return new_sums

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

def construct():
    """
    Find the largest set of distinct integers in 2..40 such that all
    2^k subset products (including empty product = 1) are distinct.
    
    Using prime exponent vectors: subset products are distinct iff
    all 2^k subset sums of the exponent vectors are distinct.
    """
    candidates = _CANDIDATES
    
    # Separate primes and composites
    primes = [x for x in candidates if _is_prime(x)]
    composites = [x for x in candidates if not _is_prime(x)]
    
    # Sort composites by "sparsity" of their exponent vector (fewer non-zero entries first)
    def vector_sparsity(x):
        vec = _VECTORS[x]
        return sum(1 for v in vec if v > 0)
    
    # Greedy initialization with multiple orderings
    best_set = []
    best_size = 0
    
    # Try different orderings
    orderings = [
        primes + sorted(composites, key=lambda x: vector_sparsity(x)),
        primes + sorted(composites, key=lambda x: -vector_sparsity(x)),
        sorted(primes + composites, key=lambda x: vector_sparsity(x)),
        sorted(primes + composites, key=lambda x: -vector_sparsity(x)),
        sorted(primes + composites, key=lambda x: x),
        sorted(primes + composites, key=lambda x: -x),
    ]
    
    for order in orderings:
        if time.time() > _DEADLINE:
            break
        sums = {tuple([0] * _NUM_PRIMES)}
        selected = []
        for x in order:
            if time.time() > _DEADLINE:
                break
            vec = _VECTORS[x]
            if _can_add_with_sums(sums, vec):
                sums = _add_vector_with_sums(sums, vec)
                selected.append(x)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
    
    # Backtracking search to improve
    # Use the best greedy result as lower bound
    
    # Order candidates: primes first (most likely to be in optimal set), then composites by sparsity
    search_candidates = primes + sorted(composites, key=lambda x: vector_sparsity(x))
    
    best = [best_set]
    
    def backtrack(candidates, sums, selected, depth):
        if time.time() > _DEADLINE:
            return
        
        if len(selected) > len(best[0]):
            best[0] = list(selected)
            
        if depth >= len(candidates):
            return
            
        # Pruning: if we can't beat the best size with remaining candidates
        if len(selected) + (len(candidates) - depth) <= len(best[0]):
            return
            
        x = candidates[depth]
        vec = _VECTORS[x]
        
        # Try including x
        if _can_add_with_sums(sums, vec):
            new_sums = _add_vector_with_sums(sums, vec)
            selected.append(x)
            backtrack(candidates, new_sums, selected, depth + 1)
            selected.pop()
            
        # Try excluding x
        backtrack(candidates, sums, selected, depth + 1)
        
    # Run backtracking
    backtrack(search_candidates, {tuple([0] * _NUM_PRIMES)}, [], 0)
    
    # Local search: try to improve by swapping elements
    def local_search(initial_set):
        if time.time() > _DEADLINE:
            return list(initial_set)
        
        current = list(initial_set)
        current_sums = _subset_sums_set([_VECTORS[x] for x in current])
        current_size = len(current)
        
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            for i in range(len(current)):
                if time.time() > _DEADLINE:
                    break
                for x in candidates:
                    if x in current:
                        continue
                    # Try swapping current[i] with x
                    test_set = list(current)
                    test_set[i] = x
                    test_vecs = [_VECTORS[y] for y in test_set]
                    test_sums = _subset_sums_set(test_vecs)
                    if len(test_sums) == (1 << len(test_set)):
                        current = test_set
                        current_sums = test_sums
                        improved = True
                        break
                if improved:
                    break
        
        return current
    
    # Apply local search
    improved_set = local_search(best[0])
    if len(improved_set) > len(best[0]):
        best[0] = improved_set
    
    # Verify the best result
    def verify(nums):
        if not nums:
            return True
        vecs = [_VECTORS[x] for x in nums]
        sums = _subset_sums_set(vecs)
        return len(sums) == (1 << len(nums))
        
    if not verify(best[0]):
        # Fallback to primes
        best[0] = primes
        
    return [int(x) for x in best[0]]

# EVOLVE-BLOCK-END
