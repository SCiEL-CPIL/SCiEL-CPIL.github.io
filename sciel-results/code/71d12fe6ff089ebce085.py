import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    max_val = 256
    max_size = 60
    
    current_set = []
    # bits[k] is a bitmask of possible sums for subsets of cardinality k
    # bits[0] = 1 (bit 0 set, sum 0)
    bits = [0] * (max_size + 1)
    bits[0] = 1
    
    for x in range(max_val, 0, -1):
        if time.time() > deadline:
            break
            
        # Tentatively add x
        new_bits = [0] * (max_size + 1)
        new_bits[0] = 1
        
        current_len = len(current_set)
        
        # Update DP
        # We need to consider subsets of the new set of size k.
        # They either don't use x (from old bits[k]) or use x (from old bits[k-1] + x).
        for k in range(1, current_len + 2):
            # old_bits[k] is 0 if k > current_len
            val_from_old = bits[k] if k <= current_len else 0
            val_from_new = (bits[k-1] << x) if k-1 <= current_len else 0
            new_bits[k] = val_from_old | val_from_new
            
        # Check for collisions
        valid = True
        new_len = current_len + 1
        for i in range(new_len + 1):
            if new_bits[i] == 0: continue # No sums possible? (Shouldn't happen for i<=n with distinct positive ints)
            for j in range(i + 1, new_len + 1):
                if new_bits[j] == 0: continue
                if (new_bits[i] & new_bits[j]) != 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            current_set.append(x)
            bits = new_bits
            if len(current_set) == max_size:
                break
                
    return current_set
