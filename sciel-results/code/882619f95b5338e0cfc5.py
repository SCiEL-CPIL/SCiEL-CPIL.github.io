import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s buffer for the 90s kill limit
    
    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Precompute exponent vectors for numbers 2..40
    # vecs[n] is a tuple of exponents
    vecs = {}
    for n in range(2, 41):
        v = [0] * num_primes
        m = n
        for i, p in enumerate(primes):
            while m % p == 0:
                v[i] += 1
                m //= p
        vecs[n] = tuple(v)
        
    numbers = list(range(2, 41))
    best_set = []
    best_size = 0
    
    # Helper to check if a set of vectors is dissociated (distinct subset sums)
    # We maintain the set of all subset sums.
    # Initially {0}.
    # When adding v, new sums are old_sums U (old_sums + v).
    # If intersection is non-empty, it's not dissociated.
    # Since 0 is in old_sums, v must not be in old_sums.
    # More generally, no s1, s2 in old_sums should satisfy s1 - s2 = v.
    # This is equivalent to checking if (old_sums + v) intersects old_sums.
    
    def is_dissociated_add(current_sums, v):
        # Check if any s in current_sums has s+v in current_sums
        # Optimization: if len(current_sums) is small, this is fast.
        # For large sets, we might need a better check, but 2^17 is manageable.
        for s in current_sums:
            sv = tuple(s[i] + v[i] for i in range(num_primes))
            if sv in current_sums:
                return False
        return True

    def add_to_sums(current_sums, v):
        new_sums = set(current_sums)
        for s in current_sums:
            sv = tuple(s[i] + v[i] for i in range(num_primes))
            new_sums.add(sv)
        return new_sums

    # Strategy 1: Randomized Greedy with Local Search
    # We perform many restarts. In each restart, we shuffle numbers and greedily add them.
    # Then we try to improve by swapping elements or adding/removing.
    
    # To speed up local search, we can precompute pairwise incompatibility?
    # No, dissociation is a global property (subset sums), not just pairwise.
    # However, if two vectors are linearly dependent (one is a multiple of the other), 
    # they can never be in a dissociated set together?
    # Actually, if v2 = k*v1, then subset sums can collide?
    # Example: v1=(1,0), v2=(2,0). Sums: 0, 1, 2, 3. Distinct.
    # But v1=(1,0), v2=(1,0). Sums: 0, 1, 1, 2. Collision.
    # So identical vectors are bad.
    # What about v1=(1,0), v2=(2,0)? 
    # Sums of {v1, v2}: 0, 1, 2, 3. Distinct.
    # So linear dependence is not immediately fatal, but it restricts the set.
    
    # Let's stick to the subset sum check.
    
    # Optimization: Use a list of numbers.
    # Random restarts with different seeds.
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        random.shuffle(numbers)
        current_set = []
        current_sums = {tuple([0]*num_primes)}
        
        for n in numbers:
            v = vecs[n]
            if is_dissociated_add(current_sums, v):
                current_set.append(n)
                current_sums = add_to_sums(current_sums, v)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we found 18, it's likely optimal or near-optimal. 
            # The max possible is 39, but dissociated sets in Z^12 with small coordinates are limited.
            # Let's see if we can find 18.
            if best_size == 18:
                # Try to find 19?
                # We can continue searching, but 18 is a strong result.
                pass

    # Strategy 2: Local Search on the best set found
    # Try to swap elements in best_set with elements not in best_set to increase size or maintain size while enabling further additions.
    # Actually, we just want to maximize size.
    
    # Let's try a more targeted local search:
    # 1. Start with best_set.
    # 2. Try to add any number not in best_set.
    # 3. If cannot add, try removing one element and adding two? Or swapping?
    
    # Given the time limit, let's do a few iterations of local search on the best set found.
    
    for _ in range(10):
        if time.time() > deadline:
            break
            
        current_set = best_set[:]
        current_sums = {tuple([0]*num_primes)}
        for n in current_set:
            current_sums = add_to_sums(current_sums, vecs[n])
            
        improved = False
        candidates = [n for n in numbers if n not in current_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            v_c = vecs[c]
            if is_dissociated_add(current_sums, v_c):
                # Can add directly
                current_set.append(c)
                current_sums = add_to_sums(current_sums, v_c)
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                # Restart local search with new set
                break
        
        if not improved:
            # Try swaps: remove one, add two?
            # This is complex. Let's try simple swaps first: remove one, add one (if it helps? No, size stays same).
            # We want to increase size.
            # So we need to find a way to add an element.
            # If we can't add, we must remove some to make room.
            # Try removing each element in current_set and see if we can add 2 elements.
            
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                n_remove = current_set[i]
                # Reconstruct sums without n_remove
                # This is expensive if done naively.
                # Let's just rebuild sums from scratch for the set without n_remove.
                test_set = current_set[:i] + current_set[i+1:]
                test_sums = {tuple([0]*num_primes)}
                for n in test_set:
                    test_sums = add_to_sums(test_sums, vecs[n])
                
                # Now try to add 2 elements from candidates
                # This is O(N^2) checks.
                # Let's just try to add 1 element first.
                for c in candidates:
                    if c == n_remove: continue
                    v_c = vecs[c]
                    if is_dissociated_add(test_sums, v_c):
                        # Can add c. Size is same. Not an improvement in size, but might enable future additions?
                        # Let's ignore size-preserving swaps for now to keep it simple.
                        pass
                
                # Try adding 2 elements?
                # Too slow.
        
    return best_set
