import numpy as np
import time
import random

N = 16
NUM_INPUTS = 1 << N

def _verify_network_numpy(comps):
    """
    Verifies if the network sorts all 2^N binary inputs using NumPy vectorization.
    Returns True if valid, False otherwise.
    """
    if not comps:
        return False # A network with 0 comparators cannot sort N>1 inputs
        
    # Generate all 2^N input patterns
    # We represent each input as a bitset. 
    # To be efficient, we can just create an array of shape (NUM_INPUTS, N) of bits?
    # Or better, use integer arithmetic on the whole array if possible?
    # NumPy bitwise ops on uint32/int32 arrays are fast.
    
    # Create array of all masks from 0 to 2^N - 1
    masks = np.arange(NUM_INPUTS, dtype=np.uint32)
    
    for i, j in comps:
        # Check condition: bit i is 1 AND bit j is 0
        # mask has bit i set: (mask >> i) & 1 == 1
        # mask has bit j clear: (mask >> j) & 1 == 0
        
        # Vectorized check
        # Note: shifting uint32 array is fast
        bit_i = (masks >> i) & 1
        bit_j = (masks >> j) & 1
        
        # We need to swap bits if bit_i == 1 and bit_j == 0
        swap_mask = (bit_i == 1) & (bit_j == 0)
        
        # If swap is needed, toggle bit i and bit j
        # Toggle mask: (1 << i) | (1 << j)
        toggle_val = (1 << i) | (1 << j)
        
        # Apply toggle to masks where swap_mask is True
        # masks[swap_mask] ^= toggle_val
        # In-place operation on subset is efficient
        
        # Optimization: only do work if any swap is needed
        if np.any(swap_mask):
            masks[swap_mask] ^= toggle_val
            
    # Check if sorted
    # A state is sorted if all 0s are followed by all 1s.
    # For a 16-bit number, this means:
    # If k ones are present, the lowest N-k bits must be 0.
    # Equivalent check: The number of transitions from 0 to 1 is at most 1?
    # Or simpler: Check if the sequence of bits is non-decreasing.
    
    # Efficient check for sortedness in bitmasks:
    # A bitmask is sorted if it looks like 000...011...1
    # This is equivalent to: (mask | (mask - 1)) has all lower bits set? No.
    
    # Let's use the property: 
    # If we count the number of 1s (popcount), let it be k.
    # The sorted mask should be (2^k - 1) << (N - k).
    
    # Calculate popcount for all masks.
    # np.bitwise_count is not in standard numpy yet (added in 2.0? or need workaround).
    # Workaround for popcount on uint32 array:
    # x = x - ((x >> 1) & 0x55555555)
    # x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    # x = (x + (x >> 4)) & 0x0F0F0F0F
    # x = x + (x >> 8)
    # x = x + (x >> 16)
    # x & 0xFF
    
    x = masks.copy()
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    popcounts = x & 0xFF
    
    # Construct the expected sorted mask for each popcount
    # expected_mask = (1 << popcount) - 1) << (N - popcount)
    # Note: popcount is an array.
    
    # We can compute this vectorially
    # ones = (1 << popcounts) - 1
    # sorted_masks = ones << (N - popcounts)
    
    # Be careful with 1 << popcounts when popcounts is array.
    # In numpy, 1 << array works element-wise if dtype is integer.
    
    # However, shifting by variable amounts is supported.
    
    # Edge case: popcount = 0 -> expected 0.
    # popcount = 16 -> expected (2^16 - 1) << 0.
    
    # Let's compute:
    # val = (1 << popcounts) - 1
    # But 1 << 16 overflows uint32? No, 1<<16 is 65536.
    # Wait, 1 << popcounts where popcounts is uint8/32.
    
    # Actually, there is a simpler check for sortedness:
    # A bitmask is sorted if (mask & (mask - 1)) == 0 is false? No, that's power of 2.
    
    # Let's stick to the popcount comparison.
    # sorted_mask = ((1 << popcounts) - 1) << (N - popcounts)
    
    # To avoid overflow issues in numpy shifting with arrays:
    # We can compute this in Python if the array is small, but it's 65536 elements.
    # Vectorized is better.
    
    # One trick: 
    # sorted_mask = (1 << popcounts) - 1
    # sorted_mask = sorted_mask << (N - popcounts)
    
    # Let's try this.
    # Note: 1 << popcounts might be tricky if popcounts is 32. But max is 16.
    
    ones = (1 << popcounts) - 1
    # If popcounts is 0, 1<<0 - 1 = 0.
    # If popcounts is 16, 1<<16 - 1 = 65535.
    
    # Shift left by N - popcounts
    shift_amt = N - popcounts
    expected = ones << shift_amt
    
    # Check if all masks match expected
    return np.all(masks == expected)

def construct():
    # Start with Odd-Even Transposition Network (120 comparators)
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # Ensure initial network is valid (it should be)
    # _verify_network_numpy(comps) # Skip to save time, we know it's valid
    
    best_comps = comps.copy()
    
    # Heuristic: Random Deletion
    # To be more effective, we can try to delete comparators that are "less important".
    # But random deletion with fast verification is surprisingly effective.
    
    # We will also add a simple "restart" mechanism if we get stuck for a long time?
    # Or just keep going.
    
    # To improve the search, we can shuffle the order of deletion attempts.
    # We'll just pick random indices.
    
    # Optimization: If we fail to delete for a while, maybe try a different strategy?
    # For now, pure random deletion is robust.
    
    # To speed up convergence, we can try to delete from the end first?
    # Later comparators in a sorting network are often redundant.
    # Let's bias the random selection towards the end of the list.
    
    while time.time() < deadline:
        if not comps:
            break
            
        # Bias towards later comparators (higher index in list)
        # Simple uniform random is okay, but let's try a slight bias or just uniform.
        # Uniform is safer to avoid local minima.
        
        idx = random.randint(0, len(comps) - 1)
        candidate = comps[idx]
        
        # Remove it
        comps.pop(idx)
        
        # Verify
        if _verify_network_numpy(comps):
            # Keep it removed
            best_comps = comps.copy()
        else:
            # Restore it
            comps.insert(idx, candidate)
            
    return best_comps

if __name__ == "__main__":
    # Quick test
    pass
