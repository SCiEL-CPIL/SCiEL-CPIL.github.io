"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50
    n = 256
    
    # Precompute conflicts using bitmasks
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    # Helper to count bits
    def count_bits(mask):
        return bin(mask).count('1')

    # Base case: All odds are always valid (sum of two odds is even, product is odd, odd % even != 0)
    odds_mask = 0
    for x in range(1, n + 1, 2):
        odds_mask |= (1 << x)
    
    best_mask = odds_mask
    best_size = count_bits(best_mask)
    
    full_mask = 0
    for x in range(1, n + 1):
        full_mask |= (1 << x)

    # Strategy 1: Greedy Addition to Odds
    evens = [x for x in range(2, n + 1, 2)]
    
    def greedy_add_to_odds():
        current_mask = odds_mask
        # Randomize order of evens
        order = evens[:]
        random.shuffle(order)
        for e in order:
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
        return current_mask

    # Strategy 2: Greedy Removal from Full Set
    def greedy_remove_from_full():
        current_mask = full_mask
        # To speed up, we can just iterate. 256 nodes is small.
        while current_mask:
            # Find vertex with max conflicts in current_mask
            max_conf = -1
            candidates = []
            # Iterate through bits in current_mask
            m = current_mask
            while m:
                # Get lowest set bit
                lsb = m & -m
                v = lsb.bit_length() - 1
                # Count conflicts of v within current_mask
                conf_count = count_bits(conflict_mask[v] & current_mask)
                if conf_count > max_conf:
                    max_conf = conf_count
                    candidates = [v]
                elif conf_count == max_conf:
                    candidates.append(v)
                m ^= lsb
                
            if max_conf == 0:
                break # No conflicts left
                
            # Pick random candidate among max conflict vertices
            target = random.choice(candidates)
            current_mask &= ~(1 << target)
            
        return current_mask

    # Local Search: Try to add any compatible vertex
    def local_search_add(mask):
        improved = True
        while improved:
            improved = False
            for x in range(1, n + 1):
                if not (mask >> x) & 1:
                    if (conflict_mask[x] & mask) == 0:
                        mask |= (1 << x)
                        improved = True
                        break
        return mask

    # Main Loop
    while time.time() < deadline - 5:
        # Try Strategy 1
        if time.time() < deadline - 5:
            res_mask = greedy_add_to_odds()
            res_mask = local_search_add(res_mask)
            size = count_bits(res_mask)
            if size > best_size:
                best_size = size
                best_mask = res_mask

        # Try Strategy 2
        if time.time() < deadline - 5:
            res_mask = greedy_remove_from_full()
            # Greedy removal produces a maximal independent set, so local_search_add shouldn't find anything, 
            # but it's safe to call.
            res_mask = local_search_add(res_mask)
            size = count_bits(res_mask)
            if size > best_size:
                best_size = size
                best_mask = res_mask

    # Convert best_mask back to list
    best = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    
    return best
# EVOLVE-BLOCK-END
