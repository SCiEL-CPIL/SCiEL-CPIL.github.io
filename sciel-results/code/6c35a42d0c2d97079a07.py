import random
import time

def construct():
    # Max possible sum for 30 elements of max 256
    MAX_SUM = 256 * 30 + 10
    
    # Precompute multiples and divisors masks
    # multiples_mask[s] has bits set at 2s, 3s, ...
    # divisors_mask[s] has bits set at d such that s % d == 0 (d < s)
    
    multiples_mask = [0] * (MAX_SUM + 1)
    divisors_mask = [0] * (MAX_SUM + 1)
    
    for s in range(1, MAX_SUM + 1):
        # Multiples
        m = s * 2
        mask = 0
        while m <= MAX_SUM:
            mask |= (1 << m)
            m += s
        multiples_mask[s] = mask
        
        # Divisors
        d = 1
        mask = 0
        while d < s:
            if s % d == 0:
                mask |= (1 << d)
            d += 1
        divisors_mask[s] = mask

    def is_valid_incremental(old_bits, new_bits, added_bits):
        # Check if any s in added_bits conflicts with any bit in new_bits
        # Conflict: s divides t or t divides s for distinct s, t.
        # Since added_bits are new, they didn't exist in old_bits.
        # We check against ALL bits in new_bits (old + new).
        
        bits_to_check = added_bits
        while bits_to_check:
            # Get lowest set bit
            lsb = bits_to_check & -bits_to_check
            s = lsb.bit_length() - 1
            
            # Check if s divides any other bit in new_bits
            # multiples_mask[s] contains 2s, 3s...
            if new_bits & multiples_mask[s]:
                return False
            
            # Check if any other bit in new_bits divides s
            # divisors_mask[s] contains d < s where s % d == 0
            if new_bits & divisors_mask[s]:
                return False
            
            bits_to_check ^= lsb
        
        return True

    best_set = []
    best_size = 0
    
    candidates = list(range(1, 257))
    
    start_time = time.time()
    timeout = 55  # Leave 5s buffer
    
    attempts = 0
    while time.time() - start_time < timeout:
        attempts += 1
        
        # Strategy: Mostly descending, some random
        if attempts % 5 == 0:
            cand = candidates[:]
            random.shuffle(cand)
        else:
            cand = sorted(candidates, reverse=True)
            
        current_set = []
        bits = 0
        
        for x in cand:
            # New sums: shift old bits by x, add x itself
            # Note: bits represents the set of subset sums.
            # If we add x, the new subset sums are {s + x | s in old_sums} U {x}
            # In bitset: (bits << x) | (1 << x)
            # Union with old bits: bits | (bits << x) | (1 << x)
            
            shifted = bits << x
            new_single = 1 << x
            new_bits = bits | shifted | new_single
            
            # Identify which bits were added
            added_bits = new_bits & ~bits
            
            # Optimization: If added_bits is 0, it means x created no new sums?
            # Impossible, 1 << x is always new if x > 0 and not in bits.
            # Unless x is 0, but x >= 1.
            
            if is_valid_incremental(bits, new_bits, added_bits):
                bits = new_bits
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we find a very good set, we can tighten the search or stop
            if best_size >= 20:
                # Break early if we hit a high bar? 
                # Let's keep trying for a bit more to see if 21+ is possible
                pass

    # Ensure return type is list of ints
    return [int(x) for x in best_set]
