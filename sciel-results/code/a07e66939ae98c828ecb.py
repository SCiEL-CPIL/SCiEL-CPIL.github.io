    def build_mask(w, M):
        # M is power of 2
        # pattern_len = 2 ** (w + 1)
        # ones_len = 2 ** w
        # zeros_len = 2 ** w
        
        # Create pattern: 0...0 1...1
        # In Python, bit 0 is LSB.
        # We want bit 0..2^w-1 to be 0.
        # Bit 2^w..2^(w+1)-1 to be 1.
        
        # Mask for the lower 2^w bits being 1? No.
        # We want the upper half of the block to be 1.
        
        # Let's build the pattern integer.
        # pattern = (1 << ones_len) - 1  # This is 2^w ones.
        # We need to shift it by zeros_len (which is ones_len).
        # pattern = pattern << ones_len
        
        # Now we have a block of size 2^(w+1) with the correct bits.
        # We need to repeat it M / (2^(w+1)) times.
        
        # repeats = M >> (w + 1)
        
        # We can use a loop to shift and OR.
        # Or use a trick.
        
        mask = 0
        block = pattern
        # The block is 2^(w+1) bits wide.
        # We need to place it at 0, 2^(w+1), 2*2^(w+1), ...
        
        # This loop runs M / 2^(w+1) times.
        # For w=0, M/2 = 65536 iterations.
        # For w=16, M/2^17 = 1 iteration.
        # Total iterations: sum(M / 2^(w+1)) for w=0..16.
        # M/2 + M/4 + ... + M/2^17 = M - 1.
        # So ~131k iterations total.
        # Much better than 2.2M.
        
        for r in range(M >> (w + 1)):
            mask |= (block << (r * (1 << (w + 1))))
            
        return mask
    