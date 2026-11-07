"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import product

P = {'n': 40, 'maximum_size': 18}

# Primes up to 40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}
DIM = len(PRIMES)

def get_vector(n):
    """Get the prime exponent vector for n."""
    vec = [0] * DIM
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

# Precompute vectors for all candidates 2..40
CANDIDATES = list(range(2, P['n'] + 1))
VECTORS = [get_vector(x) for x in CANDIDATES]
# Map vector to candidate index for potential reverse lookup if needed, 
# but we work with indices of CANDIDATES.

def is_independent_mitm(selected_indices, new_idx):
    """
    Check if the vector at new_idx is in the {-1, 0, 1}-span of the vectors at selected_indices.
    If yes, adding it creates a dependency (collision).
    If no, it is independent.
    """
    v_new = VECTORS[new_idx]
    if not selected_indices:
        return True # Always independent if set is empty
        
    k = len(selected_indices)
    # Split selected indices into two halves
    mid = k // 2
    left_indices = selected_indices[:mid]
    right_indices = selected_indices[mid:]
    
    def expand_sums(indices):
        sums = {tuple([0]*DIM)}
        for idx in indices:
            v = VECTORS[idx]
            new_sums = set()
            for s in sums:
                # s
                new_sums.add(s)
                # s + v
                new_sums.add(tuple(s[i] + v[i] for i in range(DIM)))
                # s - v
                new_sums.add(tuple(s[i] - v[i] for i in range(DIM)))
            sums = new_sums
        return sums

    left_sums = expand_sums(left_indices)
    
    # For each sum in right half, check if v_new - sum_right is in left_sums
    # We can iterate over right sums and check membership
    
    for r_sum in expand_sums(right_indices):
        # We need left_sum such that left_sum + r_sum = v_new
        # => left_sum = v_new - r_sum
        target = tuple(v_new[i] - r_sum[i] for i in range(DIM))
        if target in left_sums:
            # Found a dependency
            return False
            
    return True

def verify_distinct_subset_products(values):
    """
    Verify that all 2^k subset products are distinct.
    Returns True if valid, False otherwise.
    """
    k = len(values)
    if k > 20:
        # Too many to enumerate directly, rely on MITM check
        return True
    
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= values[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initial best set: Primes
    primes = [i for i, x in enumerate(CANDIDATES) if all(x % d for d in range(2, int(x**0.5)+1))]
    best_set = set(primes)
    best_size = len(best_set)
    
    # Local Search with MITM check
    # We will use a randomized restart approach with greedy addition and swaps.
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Start from primes
        S = set(primes)
        
        # Greedy Add with multiple orderings
        for ordering in range(3):
            added = True
            while added and time.time() < deadline:
                added = False
                candidates = [c for c in range(len(CANDIDATES)) if c not in S]
                if ordering == 0:
                    random.shuffle(candidates)
                elif ordering == 1:
                    # Sort by number of prime factors (fewer first)
                    candidates.sort(key=lambda c: sum(1 for p in PRIMES if VECTORS[c][PRIME_TO_IDX[p]] > 0))
                else:
                    # Sort by value
                    candidates.sort(key=lambda c: CANDIDATES[c])
                for c in candidates:
                    if is_independent_mitm(list(S), c):
                        S.add(c)
                        added = True
                        break
        
        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)
            if best_size == P['maximum_size']:
                break
                
        # Swap Phase: try all single-element swaps more systematically
        if len(S) < P['maximum_size'] and time.time() < deadline - 5:
            # Try removing each element and adding each candidate
            s_list = list(S)
            for i in range(len(s_list)):
                if time.time() > deadline - 5:
                    break
                to_remove = s_list[i]
                S.remove(to_remove)
                
                candidates = [c for c in range(len(CANDIDATES)) if c not in S]
                if not candidates:
                    S.add(to_remove)
                    continue
                    
                # Try to find a candidate that can be added
                found = False
                for to_add in candidates:
                    if is_independent_mitm(list(S), to_add):
                        S.add(to_add)
                        found = True
                        # Try to add more after the swap
                        added = True
                        while added and time.time() < deadline - 5:
                            added = False
                            candidates2 = [c for c in range(len(CANDIDATES)) if c not in S]
                            random.shuffle(candidates2)
                            for c in candidates2:
                                if is_independent_mitm(list(S), c):
                                    S.add(c)
                                    added = True
                                    break
                        break
                
                if not found:
                    S.add(to_remove) # Restore
                
                if len(S) > best_size:
                    best_set = S.copy()
                    best_size = len(S)
                    if best_size == P['maximum_size']:
                        break
                
        if best_size == P['maximum_size']:
            break

    # Convert indices back to values and verify
    result = [CANDIDATES[i] for i in best_set]
    
    # Final verification using direct product enumeration if feasible
    if len(result) <= 20:
        if not verify_distinct_subset_products(result):
            # Fallback to primes only if verification fails
            return [CANDIDATES[i] for i in primes]
    
    return result

# EVOLVE-BLOCK-END
