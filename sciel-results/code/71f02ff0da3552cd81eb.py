"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import log
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill

    # Strategy:
    # 1. Precompute prime factorizations for all numbers in 2..40.
    # 2. Represent each number as a vector of prime exponents.
    # 3. A set of numbers has distinct subset products if and only if 
    #    all subset sums of these vectors are distinct.
    # 4. We perform a greedy randomized search to build a large set.
    #    - Start with an empty set.
    #    - At each step, try to add a candidate number.
    #    - Check if adding it causes any collision in subset sums.
    #    - To make it faster, we can maintain the set of all current subset sums.
    #    - If a collision is found, we can backtrack or try a different candidate.
    #    - Given the small search space (39 numbers), a simple greedy with 
    #      randomization and restarts should work well.
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Precompute exponent vectors for 2..40
    # vector[i] is a tuple of length 12, where j-th element is exponent of primes[j]
    vectors = {}
    for n in range(2, 41):
        temp = n
        vec = [0] * 12
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        vectors[n] = tuple(vec)
        
    candidates = list(range(2, 41))
    
    def is_valid_subset_sums(vecs):
        """
        Check if the set of vectors has distinct subset sums.
        Returns True if valid, False otherwise.
        """
        # Start with zero vector
        sums = {tuple([0]*12)}
        for v in vecs:
            new_sums = set()
            for s in sums:
                # Add v to s
                new_s = tuple(s[i] + v[i] for i in range(12))
                # Check collision
                if new_s in sums or new_s in new_sums:
                    return False
                new_sums.add(new_s)
            sums.update(new_sums)
        return True

    def try_greedy(max_steps=1000):
        """
        Greedy construction with randomization.
        """
        current_set = []
        current_vecs = []
        available = candidates[:]
        
        # Shuffle to break symmetry
        random.shuffle(available)
        
        while available:
            added = False
            # Try to add one of the available elements
            # We can try them in the shuffled order
            for i in range(len(available)):
                c = available[i]
                v = vectors[c]
                
                # Check if adding v to current_vecs maintains distinct subset sums
                # We do this by checking if any existing sum + v collides with any existing sum
                # But we don't have the explicit set of sums stored efficiently for fast lookup in this simple version.
                # Let's use the is_valid_subset_sums function on the combined list.
                # This is O(2^k * k) per check, which might be slow if k is large.
                # However, k is at most 18. 2^18 is ~260k. 
                # With 40 candidates, this is manageable.
                
                trial_vecs = current_vecs + [v]
                if is_valid_subset_sums(trial_vecs):
                    current_set.append(c)
                    current_vecs.append(v)
                    available.pop(i)
                    added = True
                    break
            
            if not added:
                break
                
        return current_set

    best_set = []
    best_size = 0
    
    # Run multiple random restarts
    num_restarts = 500
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        s = try_greedy()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
        # If we found the theoretical max or a very high number, we can stop early?
        # Max possible is 18. If we hit 18, we can return immediately.
        if best_size >= 18:
            break

    # Fallback: If greedy didn't find anything good (shouldn't happen), use the known 16-set
    if best_size < 16:
        fallback = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        if is_valid_subset_sums([vectors[x] for x in fallback]):
            return fallback
        else:
            # Should not happen, but return something valid (primes)
            return [p for p in primes]

    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
