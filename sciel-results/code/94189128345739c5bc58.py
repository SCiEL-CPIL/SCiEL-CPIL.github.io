"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Hardcoded known-good set of 16 elements with distinct subset products
    base_set = [11, 13, 17, 19, 23, 29, 31, 37, 2, 4, 16, 3, 9, 5, 25, 7]
    
    # Deadline: 55 seconds to be safe (limit is 60s)
    deadline = time.time() + 55.0
    
    # We want to find a larger set.
    # The large primes (>= 11) are "safe" because they have unique prime factors not appearing in any other number in [2, 40] (except themselves, which are distinct).
    # Actually, primes 11, 13, 17, 19, 23, 29, 31, 37 appear in numbers like 22, 26, 34, etc.
    # However, if we only use primes >= 11 and numbers composed of primes <= 7, there is no conflict.
    # Any number in [2, 40] that has a prime factor >= 11 is either the prime itself or a multiple of it.
    # Multiples of 11: 11, 22, 33
    # Multiples of 13: 13, 26, 39
    # Multiples of 17: 17, 34
    # Multiples of 19: 19, 38
    # Multiples of 23: 23
    # Multiples of 29: 29
    # Multiples of 31: 31
    # Multiples of 37: 37
    
    # If we include a prime p >= 11, we cannot include any other multiple of p.
    # Including the prime p is always at least as good as including any multiple of p, because p has fewer factors (smaller exponent vector) and thus less likely to cause collisions in the "small" part of the factorization.
    # Wait, if we replace 22 with 11, does that help?
    # If we have 22, we can't have 11.
    # If we have 11, we can't have 22.
    # 11 is "smaller" in terms of interaction with other numbers?
    # Actually, the problem is about distinct subset products.
    # Let S be the set.
    # If we replace 22 with 11, the products involving 11 are different from those involving 22.
    # Generally, using the prime itself is better because it reduces the "weight" of the element in the multiplicative monoid, potentially allowing more combinations.
    
    # So we fix the 8 primes: 11, 13, 17, 19, 23, 29, 31, 37.
    # Now we look for the largest subset of numbers in [2, 40] that are "7-smooth" (only prime factors 2, 3, 5, 7).
    # These numbers are:
    # 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40.
    # Total 25 numbers.
    
    smooth_numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40]
    
    # We know a valid set of 8 smooth numbers: [2, 4, 16, 3, 9, 5, 25, 7]
    # Let's try to find a valid set of 9 smooth numbers.
    # If we find one, total size is 8 + 9 = 17.
    
    def is_valid_smooth_set(smooth_subset):
        # Check if all subset products are distinct
        # Since these numbers only have factors 2, 3, 5, 7, we can represent each number by its exponent vector (e2, e3, e5, e7).
        # The product of a subset is the sum of the exponent vectors.
        # We need all subset sums to be distinct.
        # This is equivalent to saying that the exponent vectors are linearly independent over the integers in a specific way? No, it's that no two subsets have the same sum.
        # This is the "subset sum distinctness" property.
        
        k = len(smooth_subset)
        if k == 0:
            return True
            
        # Map each number to its exponent vector
        def get_vec(n):
            e = [0, 0, 0, 0]
            for i, p in enumerate([2, 3, 5, 7]):
                while n % p == 0:
                    n //= p
                    e[i] += 1
            return tuple(e)
            
        vecs = [get_vec(n) for n in smooth_subset]
        
        # Check all 2^k subsets
        # k is small (<= 9 or 10), so 2^k is manageable.
        # We can use a set of sums.
        sums = set()
        # Empty subset sum is (0,0,0,0)
        sums.add((0,0,0,0))
        
        # Iterate through all non-empty subsets
        # Bitmask iteration
        for mask in range(1, 1 << k):
            s = (0, 0, 0, 0)
            for i in range(k):
                if mask & (1 << i):
                    v = vecs[i]
                    s = (s[0] + v[0], s[1] + v[1], s[2] + v[2], s[3] + v[3])
            if s in sums:
                return False
            sums.add(s)
        return True

    # Greedy/Backtracking search for the largest valid subset of smooth_numbers
    # We can use a simple greedy approach with backtracking for small k.
    # Or just check all combinations of size 9.
    # C(25, 9) is about 2 million. Checking validity takes 2^9 = 512 steps.
    # 2 million * 512 is too slow for Python in 55s?
    # 1e9 operations is too slow.
    
    # Better: Backtracking with pruning.
    # Sort smooth_numbers in some order.
    # Try to add them one by one.
    
    best_smooth_set = [2, 4, 16, 3, 9, 5, 25, 7] # Size 8
    
    # Let's try to improve best_smooth_set.
    # We can use a recursive backtracker.
    
    def backtrack(idx, current_set):
        nonlocal best_smooth_set
        if time.time() > deadline:
            return
        
        # Pruning: if len(current_set) + (len(smooth_numbers) - idx) <= len(best_smooth_set):
        #     return
        # (This is optimistic, but useful)
        
        # Try to add smooth_numbers[idx]
        new_set = current_set + [smooth_numbers[idx]]
        if is_valid_smooth_set(new_set):
            if len(new_set) > len(best_smooth_set):
                best_smooth_set = new_set[:]
                # Don't return, try to find even larger
            backtrack(idx + 1, new_set)
        
        # Try to skip smooth_numbers[idx]
        backtrack(idx + 1, current_set)

    # To speed up, we can sort smooth_numbers.
    # Which order?
    # Numbers with more "unique" factors or smaller magnitudes might be better to pick first?
    # Actually, picking "harder" numbers first (those that conflict more) might prune earlier.
    # Let's just use the natural order or sort by magnitude.
    
    # Let's try sorting by magnitude descending? Or ascending?
    # Ascending is usually better for greedy subset sum problems?
    smooth_numbers_sorted = sorted(smooth_numbers)
    
    # Optimization: is_valid_smooth_set is called many times.
    # We can cache results? The subsets are different.
    # But we can precompute vectors.
    
    vecs_cache = {n: get_vec(n) for n in smooth_numbers}
    
    def is_valid_smooth_set_fast(smooth_subset):
        k = len(smooth_subset)
        if k == 0: return True
        vecs = [vecs_cache[n] for n in smooth_subset]
        sums = set()
        sums.add((0,0,0,0))
        for mask in range(1, 1 << k):
            s = (0, 0, 0, 0)
            for i in range(k):
                if mask & (1 << i):
                    v = vecs[i]
                    s = (s[0] + v[0], s[1] + v[1], s[2] + v[2], s[3] + v[3])
            if s in sums:
                return False
            sums.add(s)
        return True

    def backtrack_fast(idx, current_set, current_vecs, current_sums):
        nonlocal best_smooth_set
        if time.time() > deadline:
            return
            
        # Prune
        if len(current_set) + (len(smooth_numbers_sorted) - idx) <= len(best_smooth_set):
            return

        # Try adding smooth_numbers_sorted[idx]
        n = smooth_numbers_sorted[idx]
        v = vecs_cache[n]
        # Check if adding v causes collision with any existing sum?
        # No, we need to check all NEW subset sums formed by adding v to any existing subset sum.
        # Let existing sums be S. New sums are {s + v | s in S} union S.
        # Collision if any s + v is in S (for s != 0? No, if s+v = s' where s' in S).
        # Actually, if we have set of sums S, and we add vector v, the new set of sums is S U {s+v for s in S}.
        # We need this new set to have size 2 * len(S).
        # i.e., (s+v) not in S for all s in S.
        
        # Optimization:
        # If for any s in current_sums, (s+v) is in current_sums, then collision?
        # Wait.
        # Old sums: S.
        # New sums: S U {s+v | s in S}.
        # If there exists s1, s2 in S such that s1 + v = s2, then s1 + v is in S (it is s2).
        # But is it a collision?
        # The new subset sum s1 + v corresponds to subset (Subset1 + {n}).
        # The existing subset sum s2 corresponds to Subset2.
        # If s1 + v == s2, then Subset1 + {n} has same product as Subset2.
        # So yes, if any s + v is in S, we have a collision?
        # Wait.
        # If s1 + v = s2, and s1, s2 are in S.
        # Then the new sum s2 is already in S.
        # So the size of the new set of sums is less than 2 * len(S).
        # So yes, we just need to check if any (s + v) is in current_sums.
        
        valid = True
        for s in current_sums:
            ns = (s[0] + v[0], s[1] + v[1], s[2] + v[2], s[3] + v[3])
            if ns in current_sums:
                valid = False
                break
        
        if valid:
            new_sums = set(current_sums)
            for s in current_sums:
                ns = (s[0] + v[0], s[1] + v[1], s[2] + v[2], s[3] + v[3])
                new_sums.add(ns)
            
            if len(new_sums) > len(best_smooth_set) or (len(new_sums) == len(best_smooth_set) and len(current_set) + 1 > len(best_smooth_set)):
                # Actually, we just want to maximize size of set.
                # len(new_sums) is 2^(len(current_set)+1).
                # So we just track len(current_set).
                
                backtrack_fast(idx + 1, current_set + [n], current_vecs + [v], new_sums)
        
        # Skip
        backtrack_fast(idx + 1, current_set, current_vecs, current_sums)

    # Initialize backtrack
    # Start with empty set
    backtrack_fast(0, [], [], {(0,0,0,0)})
    
    # Combine with primes
    result = [11, 13, 17, 19, 23, 29, 31, 37] + best_smooth_set
    
    # Ensure distinctness and type
    result = list(set(result))
    result.sort()
    
    # Final validation just in case (should be fast for <= 18 elements)
    # Actually, if best_smooth_set is valid and primes are distinct and coprime to smooth numbers, it is valid.
    # Primes 11..37 are not factors of any smooth number (which are 7-smooth).
    # So any collision would have to be within smooth or within primes.
    # Primes are distinct, so their products are distinct (unique factorization).
    # Smooth products are distinct by construction.
    # Cross products: p * S1 vs q * S2?
    # If p * S1 = q * S2, then p/q = S2/S1.
    # p/q is a ratio of large primes. S2/S1 is a ratio of 7-smooth numbers.
    # A ratio of 7-smooth numbers is 7-smooth.
    # A ratio of distinct large primes is not 7-smooth (unless it is 1, which implies p=q, contradiction).
    # So no cross collisions.
    
    return result

# EVOLVE-BLOCK-END
