import random
import time

def construct():
    # Max possible sum for 30 elements of max 256
    # We target size up to 20-25, so max sum is 256 * 25 = 6400. 
    # Let's set MAX_SUM to 256 * 30 = 7680 to be safe.
    MAX_SUM = 7680
    
    # Precompute MultMask[v]: bitmask of all multiples of v up to MAX_SUM
    # MultMask[v] has bit i set if i % v == 0 and i > 0
    # We need to check if S contains a multiple of v other than v itself.
    # So MultMask[v] should include v itself, and we check (S & MultMask[v]) == (1 << v).
    
    MultMask = [0] * (MAX_SUM + 1)
    for v in range(1, MAX_SUM + 1):
        mask = 0
        for m in range(v, MAX_SUM + 1, v):
            mask |= (1 << m)
        MultMask[v] = mask

    def is_valid(S):
        # Check if bitset S is valid (no two sums divide each other)
        # Iterate through set bits in S
        temp = S
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            # Check if S has any other multiple of v
            # (S & MultMask[v]) should be exactly lsb
            if (S & MultMask[v]) != lsb:
                return False
            temp ^= lsb
        return True

    # Greedy construction with randomization
    candidates = list(range(1, 257))
    best_set = []
    best_size = 0
    
    start_time = time.time()
    timeout = 55  # Leave 5s buffer
    
    attempts = 0
    while time.time() - start_time < timeout:
        attempts += 1
        # Randomize candidates
        # Bias towards larger numbers? Smaller numbers create smaller sums which have more multiples.
        # So larger numbers are "safer".
        # Let's shuffle but maybe sort descending with some randomness?
        # Or just random shuffle.
        
        cand = candidates[:]
        random.shuffle(cand)
        
        # Try to build a set
        current_set = []
        bits = 0  # Bitset of sums. Bit 0 is 0 (empty sum), but we only care about non-empty.
        # We start with empty set, sum 0.
        # But the condition is on NONEMPTY subset sums.
        # So bits should not include 0?
        # If bits includes 0, then 0 divides everything? No, 0 is not a positive integer.
        # The problem says "1..30 distinct integers". Sums are positive.
        # So we only track sums >= 1.
        
        # Initialize bits = 0 (no sums)
        
        for x in cand:
            # New sums = old_sums + x, and {x}
            # new_bits = bits | (bits << x) | (1 << x)
            new_bits = bits | (bits << x) | (1 << x)
            
            # Check validity
            # Optimization: We only need to check if the NEW bits cause issues.
            # But is_valid checks everything. It's fast enough.
            
            if is_valid(new_bits):
                bits = new_bits
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we find a very large set, we might want to try to extend it further?
            # The loop restarts from scratch. 
            # To improve, we could try to "fix" the current best by trying to add more?
            # But random restarts are usually robust.
            
            # Early exit if we hit a theoretical upper bound? 
            # Max size is 30.
            if best_size >= 25:
                # Unlikely, but stop searching for better if we are close to max?
                # Actually, let's keep running to see if we can hit 26, 27...
                pass

    # Ensure return type is list of ints
    return [int(x) for x in best_set]
