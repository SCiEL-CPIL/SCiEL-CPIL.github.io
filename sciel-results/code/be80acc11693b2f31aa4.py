"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _prime_sieve(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def construct():
    n_base = 210
    max_size = 100
    max_value = 10210
    
    # 1. Precomputation
    # Get all primes up to max_value
    primes = _prime_sieve(max_value)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute parity vectors (as bitsets) for all numbers in (210, max_value]
    # parity[i] corresponds to number 210 + i + 1 ? No, let's index by value for clarity, or just a list.
    # Let's make a list `parity_list` where parity_list[k] is the bitset for number 211 + k.
    # Range: 211 to 10210. Count: 10210 - 211 + 1 = 10000.
    parity_list = []
    for num in range(211, max_value + 1):
        temp = num
        bits = 0
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                bits |= (1 << prime_to_idx[p])
        if temp > 1:
            # temp is a prime factor > sqrt(original num)
            # It must be in our prime list because we sieved up to max_value
            bits |= (1 << prime_to_idx[temp])
        parity_list.append(bits)
    
    # Target bitset for 210
    target_bits = 0
    temp = 210
    for p in primes:
        if p * p > temp:
            break
        count = 0
        while temp % p == 0:
            count += 1
            temp //= p
        if count % 2 == 1:
            target_bits |= (1 << prime_to_idx[p])
    if temp > 1:
        target_bits |= (1 << prime_to_idx[temp])
        
    # 2. Binary Search on M
    # We want to find the smallest M such that target_bits is in the span of parity_list[0 : M-210]
    # Let k = M - 210. We check span of first k elements.
    # k ranges from 1 to 10000.
    
    def check_span(k):
        """
        Returns (success, subset_indices) 
        subset_indices are indices into parity_list (0-based)
        """
        if k == 0:
            return False, []
        
        # We need to check if target is in span of parity_list[0:k]
        # Use Gaussian elimination with tracking
        
        # To avoid copying large lists repeatedly, we can just work on the slice.
        # But we need to modify rows.
        # We create a list of (vector, mask) tuples.
        # Mask tracks which original indices are involved.
        # Index i in parity_list corresponds to mask (1 << i).
        
        rows = []
        for i in range(k):
            rows.append((parity_list[i], 1 << i))
        
        # Append target
        # The target is what we want to represent.
        # We add it as a row with mask 0.
        # If it reduces to 0, its mask will contain the combination.
        rows.append((target_bits, 0))
        
        n_rows = len(rows)
        
        # Gaussian Elimination
        # We iterate through columns (primes).
        # Since num_primes is ~1260, this loop is manageable.
        
        rank = 0
        for col in range(num_primes):
            # Find pivot
            pivot = -1
            for r in range(rank, n_rows):
                if (rows[r][0] >> col) & 1:
                    pivot = r
                    break
            
            if pivot == -1:
                continue
            
            # Swap pivot to rank
            rows[rank], rows[pivot] = rows[pivot], rows[rank]
            
            # Eliminate this column from all other rows
            pivot_vec, pivot_mask = rows[rank]
            
            for r in range(n_rows):
                if r != rank and ((rows[r][0] >> col) & 1):
                    # rows[r] ^= rows[rank]
                    new_vec = rows[r][0] ^ pivot_vec
                    new_mask = rows[r][1] ^ pivot_mask
                    rows[r] = (new_vec, new_mask)
            
            rank += 1
            
            # Optional: Early exit if rank is high? No, we need to reduce target row.
            # We can stop if the target row (last row) becomes 0?
            # If rows[-1][0] becomes 0, we found a solution.
            # But we must ensure we don't miss a simpler solution?
            # Actually, if it becomes 0, the current mask is a valid solution.
            # We can return immediately.
            if rows[-1][0] == 0:
                # Check size
                mask = rows[-1][1]
                # Count bits in mask
                # If bit_count is available (Python 3.10+), use it.
                # Otherwise, bin(mask).count('1')
                count = bin(mask).count('1')
                if count <= max_size:
                    # Extract indices
                    indices = []
                    m = mask
                    idx = 0
                    while m:
                        if m & 1:
                            indices.append(idx)
                        m >>= 1
                        idx += 1
                    return True, indices
                else:
                    # Solution too large. 
                    # We could continue elimination to see if a smaller one emerges?
                    # Unlikely. Gaussian elimination gives a specific solution.
                    # If it's too big, we might need a different strategy or accept failure for this M.
                    # However, for the purpose of finding *a* solution, if the first one is too big,
                    # it's risky. But usually small M implies small dimension and small solutions.
                    # Let's return False to force a larger M if needed, or just return the indices
                    # and let the caller check.
                    # The task says "Minimize max...". If we return a valid subset, it's fine.
                    # If the subset is > 100, it's invalid.
                    # So we should return False if count > max_size.
                    return False, []
                    
            # Optimization: If rank == num_primes, we can break? No, we still need to reduce the target row.
            # But if rank reaches the number of rows - 1, the last row is the only one left.
            # Actually, we just need to reduce the target row to 0.
            
        # After processing all columns, check if target row is 0
        if rows[-1][0] == 0:
            mask = rows[-1][1]
            count = bin(mask).count('1')
            if count <= max_size:
                indices = []
                m = mask
                idx = 0
                while m:
                    if m & 1:
                        indices.append(idx)
                    m >>= 1
                    idx += 1
                return True, indices
        
        return False, []

    # Binary search for minimal k (number of elements)
    # k corresponds to M = 210 + k
    # We want smallest k such that check_span(k) is True.
    
    lo = 1
    hi = len(parity_list)
    best_k = None
    best_indices = None
    
    # First, check if a solution exists at all in the full range
    # If not, we might have to return a fallback, but the problem implies a solution exists.
    
    while lo <= hi:
        mid = (lo + hi) // 2
        success, indices = check_span(mid)
        if success:
            best_k = mid
            best_indices = indices
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_k is None:
        # Fallback
        return [840]
        
    # Convert indices to values
    # index i corresponds to value 211 + i
    result = [211 + i for i in best_indices]
    
    # Verify
    prod = 1
    for x in result:
        prod *= x
    prod *= 210
    s = math.isqrt(prod)
    if s * s != prod:
        # Should not happen
        return [840]
        
    return result
# EVOLVE-BLOCK-END
