"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

def _exponent_vector(num, primes):
    v = []
    for p in primes:
        cnt = 0
        while num % p == 0:
            num //= p
            cnt += 1
        v.append(cnt)
    return tuple(v)

def construct():
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Map number to exponent vector
    vectors = {}
    for x in range(2, 41):
        vectors[x] = _exponent_vector(x, primes)
        
    # All candidates
    all_candidates = list(range(2, 41))
    
    best_set = []
    best_size = 0
    
    # We will use a backtracking search.
    # To make it efficient, we start with the primes and add composites.
    # Or we can search from scratch, but starting with primes is a good heuristic.
    # Let's try to find the maximum set by backtracking.
    # Since N=39 is small, we can try a few strategies.
    
    # Strategy: 
    # 1. Try to include all primes.
    # 2. Add composites greedily.
    # 3. Try to swap primes for composites if it leads to a larger set.
    
    def get_subset_sums(vecs):
        sums = set()
        sums.add((0,) * num_primes)
        for v in vecs:
            new_sums = set()
            for s in sums:
                # Add v to s
                new_s = tuple(s[i] + v[i] for i in range(num_primes))
                new_sums.add(new_s)
            sums = sums.union(new_sums)
        return sums
    
    # Initial set: all primes
    current_set = primes[:]
    current_vecs = [vectors[x] for x in current_set]
    current_sums = get_subset_sums(current_vecs)
    
    # Candidates to add: composites
    composites = [x for x in range(2, 41) if x not in current_set]
    
    # Greedily add composites
    # We sort composites by "complexity" or just iterate
    # To maximize, we should try to add as many as possible.
    # Order might matter.
    
    # Let's try adding all composites that are valid
    # We need to check validity against the *current* set of sums.
    
    added = []
    for c in composites:
        v = vectors[c]
        # Check if v can be added
        # New sums = current_sums U (current_sums + v)
        # Check if they are disjoint
        # This is equivalent to checking if v is in the difference set of current_sums
        # Or simply:
        new_sums = set()
        for s in current_sums:
            new_s = tuple(s[i] + v[i] for i in range(num_primes))
            new_sums.add(new_s)
        
        if len(new_sums) == 2 * len(current_sums):
            # Valid
            current_set.append(c)
            current_vecs.append(v)
            current_sums = current_sums.union(new_sums)
            added.append(c)
    
    if len(current_set) > best_size:
        best_set = current_set[:]
        best_size = len(current_set)
        
    # Now, try to improve by swapping primes
    # If we remove a prime, we might be able to add more composites.
    # This is a local search / hill climbing.
    
    # Let's do a simple local search
    # Try removing each prime and re-adding composites greedily
    
    for i in range(len(primes)):
        p_to_remove = primes[i]
        # Create a new state without p_to_remove
        temp_set = [x for x in current_set if x != p_to_remove]
        temp_vecs = [vectors[x] for x in temp_set]
        temp_sums = get_subset_sums(temp_vecs)
        
        # Re-add composites (including those already in current_set? No, just the ones not in temp_set)
        # We should re-evaluate all composites not in temp_set
        candidates_to_add = [x for x in composites if x not in temp_set]
        # Also, maybe we can re-add p_to_remove? No, we are testing removal.
        
        # Sort candidates? Maybe by value?
        for c in candidates_to_add:
            v = vectors[c]
            new_sums = set()
            for s in temp_sums:
                new_s = tuple(s[i] + v[i] for i in range(num_primes))
                new_sums.add(new_s)
            
            if len(new_sums) == 2 * len(temp_sums):
                temp_set.append(c)
                temp_vecs.append(v)
                temp_sums = temp_sums.union(new_sums)
                
        if len(temp_set) > best_size:
            best_set = temp_set[:]
            best_size = len(temp_set)
            
    # Ensure all are Python ints
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
