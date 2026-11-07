import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # 1. Sieve primes up to 10210
    max_val = 10210
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # 2. Compute squarefree kernels as bitmasks
    # kernel[x] is an integer where bit i is 1 if prime[i] divides x an odd number of times
    kernels = [0] * (max_val + 1)
    for x in range(2, max_val + 1):
        if is_prime[x]:
            kernels[x] = 1 << prime_to_idx[x]
        else:
            # Factorize x
            temp = x
            mask = 0
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    count = 0
                    while temp % p == 0:
                        temp //= p
                        count += 1
                    if count % 2 == 1:
                        mask |= (1 << prime_to_idx[p])
            if temp > 1:
                mask |= (1 << prime_to_idx[temp])
            kernels[x] = mask
            
    target_mask = kernels[210]
    
    # 3. Binary search on M
    # We want min M such that exists subset of {211..M} summing to target_mask
    lo, hi = 211, max_val
    best_list = None
    best_M = max_val + 1
    
    # Helper to check if M works and return the subset
    def find_subset(M):
        # basis: dict pivot -> (mask, set_of_indices)
        basis = {}
        
        for i in range(211, M + 1):
            v = kernels[i]
            if v == 0:
                # 1 is in the kernel (empty set), so it doesn't change the product's squareness
                # But we need distinct integers > 210. 1 is not > 210.
                # Actually, kernels[i] == 0 means i is a perfect square.
                # If we include a perfect square, it doesn't affect the parity of prime exponents.
                # So we can ignore them for the linear algebra part, but we might need to include them?
                # No, we want to minimize max. Including extra numbers increases max or keeps it same.
                # We want *a* subset. We don't need to include all possible numbers.
                # So we can skip v=0 in basis construction?
                # Yes, because they add 0 to the sum. They are redundant for satisfying the condition.
                continue
                
            # Reduce v by existing basis
            for pivot in sorted(basis.keys(), reverse=True):
                if (v >> pivot) & 1:
                    v ^= basis[pivot][0]
                    # Update indices: symmetric difference
                    # We need to update the indices for the *current* vector being reduced?
                    # No, the basis vectors store their own indices.
                    # The vector `v` is just being reduced to find its pivot.
                    # We don't track indices for `v` yet, only for basis vectors.
                    pass
            
            if v == 0:
                continue
            
            # Find pivot
            # Find the highest set bit
            pivot = v.bit_length() - 1
            
            if pivot in basis:
                # Should not happen if we reduced properly
                continue
                
            # Add to basis
            # The indices for this basis vector are just {i}
            # Wait, if we reduced v by other basis vectors, the "logical" vector represented by v
            # in terms of original vectors is not just {i}.
            # We need to track the linear combination for each basis vector.
            
            # Let's restart the logic for basis construction to track indices correctly.
            
        # Corrected Basis Construction
        basis = {} # pivot -> (mask, frozenset_of_indices)
        
        for i in range(211, M + 1):
            v = kernels[i]
            if v == 0:
                continue
                
            # Track the combination of original indices that form the current vector v
            # Initially, v corresponds to {i}
            # But as we XOR with basis vectors, we must XOR their index sets too.
            # However, we are modifying v in place.
            # So we need a parallel structure or just carry the set along.
            
            # Let's use a list of (mask, indices_set) for the current vector being processed?
            # No, just one vector at a time.
            
            current_mask = v
            current_indices = {i}
            
            # Reduce current_mask by basis
            # We iterate through pivots in descending order
            # To be efficient, we can just check bits
            for pivot in sorted(basis.keys(), reverse=True):
                if (current_mask >> pivot) & 1:
                    current_mask ^= basis[pivot][0]
                    current_indices = current_indices ^ basis[pivot][1]
                    
            if current_mask == 0:
                continue
                
            pivot = current_mask.bit_length() - 1
            
            if pivot not in basis:
                basis[pivot] = (current_mask, frozenset(current_indices))
                
        # Now try to reduce target_mask
        rem_mask = target_mask
        rem_indices = set()
        
        for pivot in sorted(basis.keys(), reverse=True):
            if (rem_mask >> pivot) & 1:
                rem_mask ^= basis[pivot][0]
                rem_indices = rem_indices ^ set(basis[pivot][1])
                
        if rem_mask == 0:
            return list(rem_indices)
        else:
            return None

    # Binary Search
    # We want the smallest M.
    # Note: The solution set might not be monotonic in a simple way?
    # If a solution exists for M, does it exist for M+1?
    # Yes, because we can just not use M+1. The set of available vectors grows.
    # So existence is monotonic.
    
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        res = find_subset(mid)
        if res is not None:
            hi = mid
            best_list = res
            best_M = mid
        else:
            lo = mid + 1
            
    if best_list is None:
        # Fallback to the old logic or a known valid solution
        # The old code found 840 (k=1) or pairs.
        # Let's just return a valid solution found by a quick check
        # Or use the old code's result if available?
        # Let's try M = max_val
        res = find_subset(max_val)
        if res is not None:
            best_list = res
        else:
            # Should not happen
            return [840]
            
    return best_list
