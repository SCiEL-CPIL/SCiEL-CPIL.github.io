import random
import time

def construct():
    MAX_SUM = 256 * 30 + 10  # ~7690
    
    # Precompute forbidden masks
    # forbidden_masks[s] has bits set at 2s, 3s, ...
    forbidden_masks = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM // 2 + 1):
        mask = 0
        # Optimization: use bit shifting loop or sum of powers?
        # Loop is fine for precomputation
        for m in range(2 * s, MAX_SUM + 1, s):
            mask |= (1 << m)
        forbidden_masks[s] = mask

    def is_valid(bits):
        # Check if any bit s in bits has a multiple also in bits
        # Iterate through set bits
        b = bits
        while b:
            lsb = b & -b
            s = lsb.bit_length() - 1
            # Check if any multiple of s is in bits
            # forbidden_masks[s] contains 2s, 3s, ...
            if bits & forbidden_masks[s]:
                return False
            b ^= lsb
        return True

    candidates = list(range(1, 257))
    best_set = []
    best_size = 0
    
    start_time = time.time()
    timeout = 55
    
    # Strategy: 
    # 1. Try descending order (large numbers first, fewer multiples)
    # 2. Try random shuffles
    
    # We can also try to force a specific structure?
    # No, just search.
    
    attempts = 0
    while time.time() - start_time < timeout:
        attempts += 1
        if attempts % 3 == 0:
            cand = sorted(candidates, reverse=True)
        else:
            cand = candidates[:]
            # Bias towards larger numbers in random shuffle?
            # We can sort by -random.random() * x? No, just shuffle.
            random.shuffle(cand)
            
        current_set = []
        bits = 0
        
        for x in cand:
            new_bits = bits | (bits << x) | (1 << x)
            if is_valid(new_bits):
                bits = new_bits
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # Early stop if we hit a very high size?
            # Max possible is 30, but likely much lower.
            # If we get 15, we can keep going for a bit to see if 16 is possible.
            if best_size >= 15:
                # We can increase timeout or break? 
                # Let's keep trying a bit more.
                pass

    return [int(x) for x in best_set]
