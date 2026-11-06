import time
import random

# EVOLVE-BLOCK-START

def _score_bitset(bits):
    """
    Compute the number of distinct d > 0 such that a, a+d, a+2d all in set,
    where bits is a Python int with bit i set iff element i is in the set.
    Elements are in 1..512, so we use bits 1..512 (bit 0 unused).
    """
    count = 0
    b = bits
    # d ranges from 1 to 255 (since max element is 512, a+2d <= 512 implies d <= 255)
    for d in range(1, 256):
        if (b & (b >> d) & (b >> (2 * d))) != 0:
            count += 1
    return count

def _hill_climb(bits, deadline, rng, max_iter=10000):
    """
    Best-Improvement Hill Climbing.
    Returns best bits, best score, and the set.
    """
    current_score = _score_bitset(bits)
    best_bits = bits
    best_score = current_score
    
    # Extract set for easier manipulation
    # We can work with bits directly, but we need to know which bits are set to remove.
    # Let's maintain a list of set elements for the removal loop.
    
    # Helper to get list of set elements from bits
    def get_elements(b):
        elems = []
        tmp = b
        while tmp:
            lb = tmp & (-tmp)
            idx = lb.bit_length() - 1
            if idx > 0: # Skip bit 0 if any, though we don't set it
                elems.append(idx)
            tmp ^= lb
        return elems
    
    current_elems = get_elements(bits)
    
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        if time.time() > deadline:
            break
            
        improved = False
        best_swap = None
        best_new_score = current_score
        best_new_bits = bits
        best_new_elems = current_elems
        
        # Try all swaps: remove x, add y
        # Optimization: Only consider y that are not in current_elems
        # We can precompute a bitmask of available elements?
        # Or just check `if y in current_elems`
        
        for x in current_elems:
            x_bit = 1 << x
            # Remove x
            bits_without_x = bits & ~x_bit
            
            # Try adding y
            # To speed up, we can skip y if it's in current_elems
            # We can iterate 1..512 and check
            for y in range(1, 513):
                if y == x:
                    continue
                if bits_without_x & (1 << y):
                    continue # y is already in the set (since x was removed, if y != x, y is in original set)
                
                new_bits = bits_without_x | (1 << y)
                new_score = _score_bitset(new_bits)
                
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_bits = new_bits
                    best_swap = (x, y)
                    # Update best_new_elems later or just recompute if accepted
                    # For now, just track bits and score
                    
        if best_swap:
            # Apply the best swap
            x, y = best_swap
            current_elems.remove(x)
            current_elems.append(y)
            bits = best_new_bits
            current_score = best_new_score
            improved = True
            
            if current_score > best_score:
                best_score = current_score
                best_bits = bits
        else:
            break
            
    return best_bits, best_score, get_elements(best_bits)

def construct():
    start_time = time.time()
    # Safety margin: 10 seconds
    deadline = start_time + 140.0
    
    rng = random.Random(42)
    
    best_overall_bits = 0
    best_overall_score = -1
    best_overall_set = set()
    
    # Number of restarts
    # Each restart involves a random initialization and hill climbing.
    # Hill climbing takes a few seconds.
    # Let's do as many restarts as possible.
    
    # Phase 1: Greedy Initialization for the first run to get a good start
    # Greedy: add element that maximizes score
    greedy_bits = 0
    greedy_elems = []
    for _ in range(32):
        best_elem = -1
        best_score = -1
        for candidate in range(1, 513):
            if greedy_bits & (1 << candidate):
                continue
            new_bits = greedy_bits | (1 << candidate)
            sc = _score_bitset(new_bits)
            if sc > best_score:
                best_score = sc
                best_elem = candidate
        greedy_elems.append(best_elem)
        greedy_bits |= (1 << best_elem)
        
    # Run Hill Climb on Greedy
    hc_bits, hc_score, hc_set = _hill_climb(greedy_bits, deadline, rng)
    
    if hc_score > best_overall_score:
        best_overall_score = hc_score
        best_overall_bits = hc_bits
        best_overall_set = hc_set
        
    # Phase 2: Iterated Local Search with Random Restarts
    # We will run several restarts.
    # Each restart:
    # 1. Randomly generate a set of 32 elements.
    # 2. Run Hill Climb.
    # 3. Update best.
    
    # To diversify, we can also perturb the current best solution.
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Generate initial state
        # Mix of random and perturbed best
        if restart_count % 3 == 0 and best_overall_bits != 0:
            # Perturb the best solution
            init_bits = best_overall_bits
            init_elems = get_elements(init_bits) # Using the helper defined inside _hill_climb? No, let's define a global one or duplicate.
            # Let's define a global helper for cleanliness
            def get_elements_global(b):
                elems = []
                tmp = b
                while tmp:
                    lb = tmp & (-tmp)
                    idx = lb.bit_length() - 1
                    if idx > 0:
                        elems.append(idx)
                    tmp ^= lb
                return elems
            
            init_elems = get_elements_global(init_bits)
            
            # Perform 2-4 random swaps
            num_swaps = rng.randint(2, 4)
            for _ in range(num_swaps):
                idx = rng.randrange(32)
                x = init_elems[idx]
                y = rng.randint(1, 512)
                if y not in init_elems:
                    init_elems[idx] = y
                    init_bits = init_bits & ~(1 << x) | (1 << y)
        else:
            # Random initialization
            init_elems = set()
            while len(init_elems) < 32:
                init_elems.add(rng.randint(1, 512))
            init_elems = list(init_elems)
            init_bits = 0
            for x in init_elems:
                init_bits |= (1 << x)
                
        # Run Hill Climb
        hc_bits, hc_score, hc_set = _hill_climb(init_bits, deadline, rng)
        
        if hc_score > best_overall_score:
            best_overall_score = hc_score
            best_overall_bits = hc_bits
            best_overall_set = hc_set
            
    # Return the best set found
    return sorted(best_overall_set)

# EVOLVE-BLOCK-END
