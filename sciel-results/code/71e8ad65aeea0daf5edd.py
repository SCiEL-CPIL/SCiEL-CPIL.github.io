<<<<<<< SEARCH
def _verify_sorts_fast(nets, n_wires, timeout=None):
    """Verify that a sorting network sorts all 2^n Boolean inputs using vectorized operations."""
    n = n_wires
    total = 1 << n
    
    start_time = time.time()
    
    # Create array of all 2^n inputs
    # Use uint32 to be safe for n up to 32
    inputs = np.arange(total, dtype=np.uint32)
    
    # Apply each comparator
    for (i, j) in nets:
        bi = (inputs >> i) & 1
        bj = (inputs >> j) & 1
        # If bi > bj, we need to swap: clear bit i, set bit j
        mask = (bi > bj).astype(np.uint32)
        inputs = (inputs & ~(mask << i)) | (mask << j)
        
        if timeout and time.time() - start_time > timeout:
            return None  # Timeout
    
    # Check if sorted: all 0s come before all 1s
    # x is sorted iff x = 0 or x = (1<<n)-1 or x has form 0...01...1
    # i.e., the complement y = ~x & ((1<<n)-1) should be of form (1<<k)-1
    # which means y & (y+1) == 0
    y = (~inputs) & ((1 << n) - 1)
    # Check y & (y+1) == 0 for all
    result = (y & (y + 1)) == 0
    return bool(np.all(result))
=======
def _verify_sorts_fast(nets, n_wires, timeout=None):
    """Verify that a sorting network sorts all 2^n Boolean inputs using vectorized operations."""
    n = n_wires
    total = 1 << n
    
    start_time = time.time()
    
    # Create array of all 2^n inputs
    # Use uint32 to be safe for n up to 32
    inputs = np.arange(total, dtype=np.uint32)
    
    # Precompute masks for faster comparator application
    for (i, j) in nets:
        # Extract bits i and j
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        # If bit_i > bit_j, swap them
        swap_mask = (bit_i & ~bit_j).astype(np.uint32)
        # Clear bit_i and set bit_j where swap is needed
        inputs = (inputs & ~(swap_mask << i)) | (swap_mask << j)
        
        if timeout and time.time() - start_time > timeout:
            return None  # Timeout
    
    # Check if sorted: all 0s come before all 1s
    # A number is "sorted" (all 0s then all 1s) iff it's of form 0...01...1
    # This is equivalent to: x & (x + 1) == 0 when we look at the complement
    # Actually, let's use a simpler check:
    # x is sorted iff x has the form 000...0111...1
    # This means x & (x + 1) should be 0 for the "transition" bit
    # Better: x is sorted iff (x & (x + 1)) == 0 OR x == 0 OR x == (1<<n)-1
    # Wait, that's not right. Let me think again.
    # For x to be of form 0...01...1, the binary representation has at most one 1->0 transition
    # Actually, it has zero 1->0 transitions (all 0s first, then all 1s)
    # So x should be of the form: some number of 0s, then some number of 1s
    # This is equivalent to: x & (x + 1) == 0 is NOT correct
    # Let's use: x is sorted iff the set of positions where x has 1s is a suffix
    # i.e., if bit k is 1, then all bits k+1, k+2, ... are also 1
    # This is equivalent to: x & (x + 1) == 0 for the "carry" behavior
    # Actually, the correct check is:
    # x is sorted iff (x & (x + 1)) == 0 is false for most cases
    # Let's use a different approach:
    # x is sorted iff x can be written as (1 << k) - 1 for some k, or x = 0
    # Wait no, x can be 0, 1, 3, 7, 15, ... (all-ones numbers) OR any number of form 0...01...1
    # Actually, 0...01...1 means: the 1s are contiguous at the bottom
    # So x & (x + 1) should be 0? Let's test:
    # x = 0b0111 = 7, x+1 = 0b1000 = 8, x & (x+1) = 0 ✓
    # x = 0b0011 = 3, x+1 = 0b0100 = 4, x & (x+1) = 0 ✓
    # x = 0b0101 = 5 (NOT sorted), x+1 = 0b0110 = 6, x & (x+1) = 0b0100 ≠ 0 ✓
    # x = 0b0000 = 0, x+1 = 0b0001 = 1, x & (x+1) = 0 ✓
    # x = 0b1111 = 15, x+1 = 0b10000 = 16, x & (x+1) = 0 ✓
    # So the check is: (x & (x + 1)) == 0
    
    # But wait, we need to mask to n bits
    mask_n = (1 << n) - 1
    y = inputs & mask_n
    result = (y & (y + 1)) == 0
    return bool(np.all(result))
>>>>>>> REPLACE

