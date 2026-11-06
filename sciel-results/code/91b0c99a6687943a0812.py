def construct():
    import random
    import math
    import time
    import sys

    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCMs
    def get_lcm(a, b):
        return a * b // math.gcd(a, b)

    # bad_masks[a][b] is a bitmask of c such that (a,b,c) is a bad triple
    # a, b, c are 1-indexed in logic, 0-indexed in bit position (c-1)
    bad_masks = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Precomputation
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L = get_lcm(a, b)
            mask = 0
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if get_lcm(a, c) == L and get_lcm(b, c) == L:
                    mask |= (1 << (c - 1))
            bad_masks[a][b] = mask
            bad_masks[b][a] = mask

    def popcount(x):
        return bin(x).count('1')

    def is_valid(S_mask):
        """Check if set represented by S_mask has no bad triples."""
        for a in range(1, n + 1):
            if not (S_mask & (1 << (a - 1))):
                continue
            for b in range(a + 1, n + 1):
                if not (S_mask & (1 << (b - 1))):
                    continue
                if S_mask & bad_masks[a][b]:
                    return False
        return True

    def get_conflict_counts(S_mask):
        """
        Returns a dictionary {element: count} for elements involved in bad triples.
        Also returns a list of bad triples if needed, but counts are sufficient for heuristic.
        """
        counts = {}
        for a in range(1, n + 1):
            if not (S_mask & (1 << (a - 1))):
                continue
            for b in range(a + 1, n + 1):
                if not (S_mask & (1 << (b - 1))):
                    continue
                mask = S_mask & bad_masks[a][b]
                if mask:
                    # For each c in mask, the triple (a,b,c) is bad.
                    # a, b, and c are all part of this conflict.
                    # Increment counts for a, b, and each c.
                    if a not in counts: counts[a] = 0
                    if b not in counts: counts[b] = 0
                    counts[a] += 1
                    counts[b] += 1
                    # Iterate through set bits in mask
                    m = mask
                    while m:
                        lsb = m & -m
                        c_idx = lsb.bit_length() - 1
                        c_val = c_idx + 1
                        if c_val not in counts: counts[c_val] = 0
                        counts[c_val] += 1
                        m ^= lsb
        return counts

    def greedy_build(order, rng):
        S_mask = 0
        for x in order:
            new_mask = S_mask | (1 << (x - 1))
            if is_valid(new_mask):
                S_mask = new_mask
        return S_mask

    def repair_and_maximize(S_mask, rng):
        # Phase 1: Remove elements to resolve all conflicts
        # We remove the element involved in the most conflicts.
        while True:
            counts = get_conflict_counts(S_mask)
            if not counts:
                break
            
            # Find element with max conflict count
            # Tie-breaking: random among max, or smallest index?
            # Let's pick the one with highest count, tie-break by random.
            max_count = max(counts.values())
            candidates = [e for e, c in counts.items() if c == max_count]
            victim = rng.choice(candidates)
            
            S_mask &= ~(1 << (victim - 1))
        
        # Phase 2: Try to add back elements that were removed or weren't in the set
        # We iterate through all elements not in S_mask and try to add them.
        # To be more effective, we can try multiple times or sort by "potential".
        # Simple approach: try all elements 1..128 in random order, add if valid.
        missing = [x for x in elements if not (S_mask & (1 << (x - 1)))]
        rng.shuffle(missing)
        
        for x in missing:
            new_mask = S_mask | (1 << (x - 1))
            if is_valid(new_mask):
                S_mask = new_mask
        
        # Phase 3: Check if we can improve by swapping?
        # If we removed some elements, maybe a different choice would have been better.
        # This simple repair is local. To escape local optima, we rely on multiple restarts.
        
        return S_mask

    best_mask = 0
    best_size = 0
    
    start_time = time.time()
    max_time = 55.0
    
    rng = random.Random(42)
    
    # Run multiple iterations of Greedy + Repair
    while time.time() - start_time < max_time:
        # Generate a random order for greedy build
        order = elements[:]
        rng.shuffle(order)
        
        # Greedy build
        S_mask = greedy_build(order, rng)
        
        # Repair and maximize
        S_mask = repair_and_maximize(S_mask, rng)
        
        curr_size = popcount(S_mask)
        if curr_size > best_size:
            best_size = curr_size
            best_mask = S_mask
            
    result = []
    for i in range(1, n + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
            
    if not result:
        result = [1]
        
    return result
