import time
import random

# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    all_nums = list(range(2, P['n'] + 1))
    
    # Primes in range
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Precompute exponent vectors for all numbers 2..40
    # This allows us to check for collisions via vector addition (sums)
    # instead of large integer multiplication, which is faster and avoids big int overhead
    # Note: The condition "all subset products are distinct" is equivalent to
    # "all subset sums of prime exponent vectors are distinct".
    # This is because the mapping from integer to exponent vector is injective.
    
    vecs = {}
    for n in all_nums:
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs[n] = tuple(v)
        
    zero_vec = (0,) * num_primes
    
    # Helper to check if adding a vector to a set of sums creates a collision
    # sums is a set of tuples
    # v is the vector to add
    # Returns (is_valid, new_sums_list)
    # We maintain the set of all current subset sums
    def try_add(sums, v):
        new_sums = []
        # We need to check if any (s + v) is in sums
        # And if any (s1 + v) == (s2 + v) which implies s1 == s2, so distinctness is preserved
        # The only collision is if s + v is already in sums
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(num_primes))
            if ns in sums:
                return False, None
            new_sums.append(ns)
        return True, new_sums

    # Initial best set: Primes
    # Primes always form a valid set because their exponent vectors are basis vectors
    best_set = primes[:]
    best_size = len(best_set)
    
    # Verify initial set (should always be valid)
    current_sums = {zero_vec}
    for p in best_set:
        valid, new_sums = try_add(current_sums, vecs[p])
        if not valid:
            # Should not happen
            best_set = []
            best_size = 0
            break
        current_sums.update(new_sums)
    
    # Candidates are non-primes
    candidates = [n for n in all_nums if n not in primes]
    
    # Strategy 1: Greedy addition with multiple orderings
    # We try to add as many candidates as possible to the base set of primes
    # We use random restarts with different orderings of candidates
    
    # Also, we can try starting from scratch with random permutations of all numbers
    # But starting from primes is a strong heuristic.
    
    # Let's also consider that maybe not all primes are needed?
    # No, primes are the "most independent". Removing a prime makes it harder to add composites?
    # Actually, if we remove a prime, we lose a dimension of independence, but might allow a composite
    # that conflicts with that prime?
    # However, the max size is likely 12 (primes) + some composites.
    # Let's stick to extending the prime set first.
    
    # Optimization: 
    # Instead of just extending primes, we can do local search.
    # But first, let's do randomized greedy extension of primes.
    
    # Precompute list of candidate vectors for faster access
    cand_vecs = {n: vecs[n] for n in candidates}
    
    while time.time() < deadline:
        # Randomly shuffle candidates
        shuffled_cands = candidates[:]
        random.shuffle(shuffled_cands)
        
        # Start with primes
        current_set = primes[:]
        current_sums = {zero_vec}
        for p in primes:
            valid, new_sums = try_add(current_sums, vecs[p])
            if not valid:
                break
            current_sums.update(new_sums)
        
        # Try to add candidates in shuffled order
        for n in shuffled_cands:
            v = cand_vecs[n]
            valid, new_sums = try_add(current_sums, v)
            if valid:
                current_sums.update(new_sums)
                current_set.append(n)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size == P['maximum_size']:
                break
        
        # Additional Strategy: Local Search / Swapping
        # If we are stuck, try to swap an element in current_set with one outside
        # This is more expensive, so only do it if we haven't hit max size and have time
        if best_size < P['maximum_size']:
            # Simple local search step
            # Try to swap a random element in current_set with a random element outside
            if current_set:
                # Pick a random index to remove
                idx = random.randint(0, len(current_set) - 1)
                elem_to_remove = current_set[idx]
                
                # Build a set of current elements for fast lookup
                in_set = set(current_set)
                out_set = [n for n in all_nums if n not in in_set]
                
                if out_set:
                    # Try a few random swaps
                    for _ in range(5):
                        if time.time() > deadline:
                            break
                        elem_to_add = random.choice(out_set)
                        
                        # Tentatively remove elem_to_remove
                        # This is complex to do efficiently with the sum set.
                        # Reconstructing the sum set from scratch is O(2^k * k).
                        # k is around 12-18. 2^18 is 262k. k=18. ~4M ops.
                        # This might be too slow for many iterations.
                        # Let's skip complex swaps for now and rely on random restarts.
                        pass

    # Ensure types are correct
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
