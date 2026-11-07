import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, leave 5s margin
    
    best_n = 0
    best_set = []
    
    def check_validity(bits, n):
        """Check if the bitsets represent disjoint sums for different cardinalities."""
        seen = 0
        for k in range(n + 1):
            if bits[k] & seen:
                return False
            seen |= bits[k]
        return True

    def greedy_build(order, current_bits, current_set, current_n, mask_all):
        """Greedily add elements from order to current_set."""
        bits = current_bits[:]
        used = set(current_set)
        n = current_n
        m = mask_all
        
        for x in order:
            if time.time() > deadline:
                return bits, list(current_set), n, m
            if x in used:
                continue
            
            valid = True
            # Optimized check:
            # new_sums for cardinality k+1 is bits[k] << x
            # It must not intersect with any bits[j] where j != k+1
            # Since all bits[j] are disjoint, the union of all bits[j] for j != k+1 is (mask_all ^ bits[k+1])
            # So we check (bits[k] << x) & (mask_all ^ bits[k+1])
            for k in range(n, -1, -1):
                if bits[k]:
                    new_sums = bits[k] << x
                    forbidden = m ^ bits[k+1]
                    if new_sums & forbidden:
                        valid = False
                        break
            
            if valid:
                # Update bits and mask_all incrementally
                new_contrib = 0
                for k in range(n, -1, -1):
                    if bits[k]:
                        new_sums = bits[k] << x
                        new_contrib |= new_sums
                        bits[k + 1] |= new_sums
                m |= new_contrib
                current_set.append(x)
                used.add(x)
                n += 1
                
        return bits, list(current_set), n, m

    # Strategy: Try multiple orders
    orders = []
    # 1. Ascending
    orders.append(list(range(1, 257)))
    # 2. Descending
    orders.append(list(range(256, 0, -1)))
    # 3. Evens then Odds
    orders.append(list(range(2, 257, 2)) + list(range(1, 257, 2)))
    # 4. Odds then Evens
    orders.append(list(range(1, 257, 2)) + list(range(2, 257, 2)))
    # 5. Small-Large Alternating
    alt = []
    l, r = 1, 256
    while l <= r:
        alt.append(l)
        if l != r:
            alt.append(r)
        l += 1
        r -= 1
    orders.append(alt)
    
    # Random shuffles
    for _ in range(50):
        o = list(range(1, 257))
        random.shuffle(o)
        orders.append(o)

    for fam in orders:
        if time.time() > deadline:
            break
        
        current_set = []
        bits = [0] * 61
        bits[0] = 1
        mask_all = 1
        n = 0
        
        bits, current_set, n, mask_all = greedy_build(fam, bits, current_set, n, mask_all)
        
        if n > best_n:
            best_n = n
            best_set = current_set[:]
            
        if best_n >= 60:
            break

    # Local Search / Improvement
    # Try to extend the best set by trying all remaining numbers in random orders
    if best_n < 60 and time.time() < deadline - 5:
        # Re-initialize bits for best_set
        bits = [0] * 61
        bits[0] = 1
        mask_all = 1
        cur_n = 0
        for x in best_set:
            for k in range(cur_n, -1, -1):
                if bits[k]:
                    new_sums = bits[k] << x
                    bits[k + 1] |= new_sums
                    mask_all |= new_sums
            cur_n += 1
        n = cur_n
        
        remaining = [x for x in range(1, 257) if x not in best_set]
        
        # Try to add more
        for _ in range(100):
            if time.time() > deadline:
                break
            order = remaining[:]
            random.shuffle(order)
            
            # Copy state
            b_copy = bits[:]
            s_copy = best_set[:]
            m_copy = mask_all
            n_copy = n
            
            b_copy, s_copy, n_copy, m_copy = greedy_build(order, b_copy, s_copy, n_copy, m_copy)
            
            if n_copy > best_n:
                best_n = n_copy
                best_set = s_copy[:]
                bits = b_copy[:]
                mask_all = m_copy
                n = n_copy
                # Update remaining
                remaining = [x for x in range(1, 257) if x not in best_set]
                
                if best_n >= 60:
                    break

    # Ensure we return a valid list of ints
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 60}
# EVOLVE-BLOCK-END
