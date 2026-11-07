import time
import random
import itertools

# Primes up to 40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NPRIMES = len(_PRIMES)

def _prime_factor_vector(n):
    """Return tuple of exponents for each prime in _PRIMES."""
    vec = [0] * _NPRIMES
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

# Precompute factor vectors for all numbers 2..40
_FACTOR_VECS = {}
for _n in range(2, 41):
    _FACTOR_VECS[_n] = _prime_factor_vector(_n)

def _generate_subset_vectors(vectors, v_new):
    """Generate new subset vectors by adding v_new to all existing vectors."""
    new_vectors = set()
    for sv in vectors:
        nv = tuple(sv[i] + v_new[i] for i in range(_NPRIMES))
        new_vectors.add(nv)
    return new_vectors

def _can_add(vectors, v_new):
    """Check if v_new can be added to the set represented by vectors without collision."""
    # The new vectors are vectors U (vectors + v_new)
    # Collision if any new vector is in vectors, or if any two new vectors are equal (impossible if vectors are unique and v_new fixed)
    # Wait, if v1 + v_new == v2 + v_new, then v1 == v2. Since vectors are a set, v1 != v2 for distinct elements.
    # So we just need to check if any (sv + v_new) is in vectors.
    for sv in vectors:
        nv = tuple(sv[i] + v_new[i] for i in range(_NPRIMES))
        if nv in vectors:
            return False, None
    # If no collision with existing, the new set is vectors U (vectors + v_new)
    new_vectors = set(vectors)
    for sv in vectors:
        nv = tuple(sv[i] + v_new[i] for i in range(_NPRIMES))
        new_vectors.add(nv)
    return True, new_vectors

def _verify(nums):
    """Verify if the list of numbers is valid."""
    if not nums:
        return True
    vectors = {(0,) * _NPRIMES}
    for num in nums:
        v = _FACTOR_VECS[num]
        new_vectors = set()
        for sv in vectors:
            nv = tuple(sv[i] + v[i] for i in range(_NPRIMES))
            new_vectors.add(nv)
        if len(new_vectors) != len(vectors):
            return False
        vectors.update(new_vectors)
    return True

def construct():
    """Return a list of up to 18 distinct integers in 2..40 with distinct subset products."""
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s to be safe
    
    # Initial set: All primes
    best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_vectors = {(0,) * _NPRIMES}
    for num in best_set:
        v = _FACTOR_VECS[num]
        new_vectors = set()
        for sv in best_vectors:
            nv = tuple(sv[i] + v[i] for i in range(_NPRIMES))
            new_vectors.add(nv)
        best_vectors.update(new_vectors)
    
    best_size = len(best_set)
    
    # Candidates to try adding
    candidates = [n for n in range(2, 41) if n not in best_set]
    
    # Greedy add
    for n in candidates:
        if time.time() > deadline - 5.0:
            break
        v = _FACTOR_VECS[n]
        can, new_vecs = _can_add(best_vectors, v)
        if can:
            best_set.append(n)
            best_vectors = new_vecs
            best_size += 1
            # Remove from candidates
            candidates.remove(n)
    
    # Local search / Hill Climbing
    # Try to swap elements to improve size
    # Or just try to add more if greedy got stuck?
    # Greedy might miss combinations.
    
    # Randomized restarts / Shaking
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        # Randomly perturb best_set
        current_set = best_set[:]
        current_vectors = best_vectors.copy()
        
        # Try to remove a random element and add a different one
        if len(current_set) > 0:
            remove_idx = random.randint(0, len(current_set) - 1)
            removed_num = current_set[remove_idx]
            current_set.pop(remove_idx)
            
            # Rebuild vectors for current_set
            current_vectors = {(0,) * _NPRIMES}
            for num in current_set:
                v = _FACTOR_VECS[num]
                new_vectors = set()
                for sv in current_vectors:
                    nv = tuple(sv[i] + v[i] for i in range(_NPRIMES))
                    new_vectors.add(nv)
                current_vectors.update(new_vectors)
            
            # Try to add a random candidate
            available = [n for n in range(2, 41) if n not in current_set]
            if available:
                add_num = random.choice(available)
                v = _FACTOR_VECS[add_num]
                can, new_vecs = _can_add(current_vectors, v)
                if can:
                    current_set.append(add_num)
                    current_vectors = new_vecs
                    if len(current_set) > best_size:
                        best_set = current_set[:]
                        best_vectors = current_vectors
                        best_size = len(current_set)
                        # Restart greedy from here?
                        # Just continue local search

    # Final verification
    if not _verify(best_set):
        # Fallback to known valid set
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    return best_set
