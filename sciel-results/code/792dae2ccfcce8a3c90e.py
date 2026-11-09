import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin
    
    # Primes up to 10210
    # Simple sieve
    limit = 10210
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, p in enumerate(is_prime) if p]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    def get_mask(n):
        mask = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    mask ^= (1 << prime_to_idx[p])
        if temp > 1:
            mask ^= (1 << prime_to_idx[temp])
        return mask

    target_mask = get_mask(210)
    
    # Basis: list of (mask, combo_mask)
    # combo_mask is a bitmask of original indices (0-indexed relative to 211)
    # We store the basis in a list. We will maintain it in a way that allows quick reduction.
    # Standard Gaussian elimination:
    # basis[i] is the vector with the highest set bit at position i?
    # Or just a list of vectors, and we reduce new vectors against them.
    
    basis = [] # list of (pivot_bit, vector_mask, combo_mask)
    
    def reduce_vector(v, c):
        """
        Reduces vector v using the current basis.
        Returns (reduced_v, reduced_c).
        Also updates basis if v is independent? No, separate step.
        """
        while v != 0:
            # Find highest set bit
            pivot = v.bit_length() - 1
            # Find basis vector with this pivot
            found = False
            for i, (p, b_mask, b_combo) in enumerate(basis):
                if p == pivot:
                    v ^= b_mask
                    c ^= b_combo
                    found = True
                    break
            if not found:
                # v is independent, add to basis
                basis.append((pivot, v, c))
                # Sort basis by pivot for efficiency? 
                # For small N, linear scan is fine.
                break
        return v, c

    # We iterate M from 211
    # We need to check if target_mask is in the span.
    # We can check this after each addition by trying to reduce target_mask.
    
    best_solution = None
    
    for M in range(211, limit + 1):
        if time.time() > deadline:
            break
        
        idx = M - 211
        v = get_mask(M)
        c = 1 << idx
        
        # Add to basis
        v_red, c_red = reduce_vector(v, c)
        
        # Check if target is in span
        t_red = target_mask
        t_combo = 0
        # Reduce target against basis
        # We need to iterate through basis. 
        # To make this efficient, we can just try to reduce.
        # Since basis is small, this is fast.
        
        # Copy target
        curr_t = target_mask
        curr_c = 0
        # We need to reduce curr_t using basis.
        # The basis is not in RREF, so we might need multiple passes or careful ordering.
        # However, if we maintain basis sorted by pivot descending, one pass works for reduction to 0 or non-zero?
        # No, standard elimination requires the pivot to be unique and we eliminate from bottom up or top down.
        # Let's just do a simple loop.
        
        # To ensure we can reduce, we should keep basis in a state where we can easily reduce.
        # Actually, the `reduce_vector` function above adds to basis if independent.
        # It doesn't update existing basis vectors to be orthogonal.
        # So the basis is not in RREF.
        # To check if target is in span, we can just try to reduce it.
        # But if basis is not RREF, we might need to iterate.
        
        # Alternative: Since M is small, we can just store all vectors and do Gaussian elimination from scratch for each M?
        # No, incremental is better.
        
        # Let's refine the basis maintenance.
        # We will maintain a list of basis vectors.
        # To check if target is in span, we can use the same `reduce_vector` logic on the target, 
        # but WITHOUT adding to basis.
        
        # Let's write a helper to check if a vector is in the span of the current basis.
        def is_in_span(v, c_init=0):
            curr_v = v
            curr_c = c_init
            # We need to reduce curr_v to 0.
            # We iterate through basis. 
            # If we process pivots in descending order, we can eliminate.
            # But our basis list is not sorted by pivot.
            # Let's sort basis by pivot descending before checking? 
            # Or just loop until no change?
            
            # Given small size, we can just do:
            changed = True
            while changed and curr_v != 0:
                changed = False
                pivot = curr_v.bit_length() - 1
                for p, b_mask, b_combo in basis:
                    if p == pivot:
                        curr_v ^= b_mask
                        curr_c ^= b_combo
                        changed = True
                        break
            if curr_v == 0:
                return True, curr_c
            return False, 0

        in_span, combo = is_in_span(target_mask)
        
        if in_span:
            # Reconstruct solution
            # combo is a bitmask of indices 0..M-211
            sol_indices = []
            for i in range(idx + 1):
                if combo & (1 << i):
                    sol_indices.append(i + 211)
            
            # Check size constraint
            if 1 <= len(sol_indices) <= 100:
                # Verify solution
                prod = 210
                for x in sol_indices:
                    prod *= x
                # Check if square
                root = math.isqrt(prod)
                if root * root == prod:
                    best_solution = sol_indices
                    # We want to minimize max, so we return immediately?
                    # Yes, we are iterating M upwards, so first solution is optimal for max.
                    return [int(x) for x in best_solution]
    
    # If no solution found within deadline or limit, return best found or empty?
    # The problem guarantees a solution.
    if best_solution:
        return [int(x) for x in best_solution]
    
    # Fallback: Return a known valid solution if search fails (should not happen)
    # Based on prompt, (224, 240) is valid.
    return [224, 240]
