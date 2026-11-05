import time
import random

def construct():
    N = 256
    # Use a tight deadline to ensure we return well before 60s
    deadline = time.time() + 55.0 

    # Precompute 4-AP conflicts
    # conflicts[x] is a list of masks. A mask has bits set for the other 3 elements in a 4-AP containing x.
    # If (current_mask & conflict_mask) == conflict_mask, then adding x creates a 4-AP.
    conflicts = [[] for _ in range(N)]
    
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # 4-AP: (a, b, c, e)
            conflicts[a].append((1 << b) | (1 << c) | (1 << e))
            conflicts[b].append((1 << a) | (1 << c) | (1 << e))
            conflicts[c].append((1 << a) | (1 << b) | (1 << e))
            conflicts[e].append((1 << a) | (1 << b) | (1 << c))

    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for cmask in conflicts[x]:
            if (mask & cmask) == cmask:
                return False
        return True

    def count_3aps(mask):
        """Count 3-term APs in the set represented by mask."""
        count = 0
        for i in range(N):
            if not ((mask >> i) & 1):
                continue
            max_d = (N - 1 - i) // 2
            for d in range(1, max_d + 1):
                j = i + d
                k = i + 2 * d
                if ((mask >> j) & 1) and ((mask >> k) & 1):
                    count += 1
        return count

    def delta_3aps_add(mask, x):
        """Calculate increase in 3-AP count if x is added to mask."""
        delta = 0
        # Case 1: x is the start: x, x+d, x+2d
        max_d = (N - 1 - x) // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x + d)) & 1) and ((mask >> (x + 2 * d)) & 1):
                delta += 1
        # Case 2: x is the middle: x-d, x, x+d
        max_d = min(x, N - 1 - x)
        for d in range(1, max_d + 1):
            if ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1):
                delta += 1
        # Case 3: x is the end: x-2d, x-d, x
        max_d = x // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x - 2 * d)) & 1) and ((mask >> (x - d)) & 1):
                delta += 1
        return delta

    def delta_3aps_remove(mask, x):
        """Calculate decrease in 3-AP count if x is removed from mask.
        This is equivalent to delta_3aps_add on the mask WITHOUT x, but we need to be careful.
        Actually, it's just the number of 3-APs containing x.
        """
        delta = 0
        # Case 1: x is the start
        max_d = (N - 1 - x) // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x + d)) & 1) and ((mask >> (x + 2 * d)) & 1):
                delta += 1
        # Case 2: x is the middle
        max_d = min(x, N - 1 - x)
        for d in range(1, max_d + 1):
            if ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1):
                delta += 1
        # Case 3: x is the end
        max_d = x // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x - 2 * d)) & 1) and ((mask >> (x - d)) & 1):
                delta += 1
        return delta

    def greedy_init():
        """Build a valid set using randomized greedy."""
        mask = 0
        elems = list(range(N))
        random.shuffle(elems)
        
        for x in elems:
            if is_valid_add(mask, x):
                # Always add if valid? 
                # Adding an element never decreases the 3-AP count.
                # It might block future additions, but usually more elements = more 3-APs.
                mask |= (1 << x)
        return mask

    best_mask = 0
    best_score = 0

    # Run multiple restarts
    while time.time() < deadline:
        # 1. Initialize
        mask = greedy_init()
        score = count_3aps(mask)
        
        if score > best_score:
            best_score = score
            best_mask = mask

        # 2. Local Search
        # We will try to improve the current mask
        # Moves: Add (if valid), Swap (if valid and improves score)
        # We don't need Remove unless we are swapping.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try a few random moves
            # 1. Try to add an element
            # Pick random elements not in mask
            candidates = [i for i in range(N) if not ((mask >> i) & 1)]
            if candidates:
                # Shuffle to randomize order
                random.shuffle(candidates)
                for x in candidates[:10]: # Try up to 10 random adds
                    if is_valid_add(mask, x):
                        delta = delta_3aps_add(mask, x)
                        # Adding always adds non-negative delta
                        # But we should check if it's worth it.
                        # Since we want to maximize, and add is always non-negative,
                        # we should add if valid.
                        mask |= (1 << x)
                        score += delta
                        improved = True
                        if score > best_score:
                            best_score = score
                            best_mask = mask
                        break # Restart search with new mask? Or continue?
                        # Breaking ensures we re-evaluate other candidates in next loop iteration
                        # which is good because the mask changed.

            if not improved and time.time() < deadline:
                # 2. Try to swap
                # Pick random x in mask, y not in mask
                in_mask = [i for i in range(N) if (mask >> i) & 1]
                not_in_mask = [i for i in range(N) if not ((mask >> i) & 1)]
                
                if in_mask and not_in_mask:
                    # Try a few random swaps
                    for _ in range(10):
                        x = random.choice(in_mask)
                        y = random.choice(not_in_mask)
                        
                        # Check validity of swap:
                        # 1. Removing x must not break validity (it can't, removing elements never creates 4-APs)
                        # 2. Adding y to (mask - x) must be valid
                        new_mask = mask & ~(1 << x)
                        if is_valid_add(new_mask, y):
                            # Calculate delta
                            # delta = delta_add(new_mask, y) - delta_remove(mask, x)
                            # Note: delta_remove(mask, x) is count of 3-APs containing x in mask.
                            # delta_add(new_mask, y) is count of 3-APs containing y in new_mask.
                            # These are disjoint sets of 3-APs?
                            # A 3-AP containing x and y would be counted in both?
                            # If x and y are in the same 3-AP, say (a, x, y), then:
                            # In mask, (a, x, y) is a 3-AP.
                            # In new_mask, y is added. (a, x, y) is NOT in new_mask because x is removed.
                            # So the 3-AP (a, x, y) is lost.
                            # delta_remove counts it.
                            # delta_add does NOT count it because x is not in new_mask.
                            # So we just subtract delta_remove and add delta_add.
                            # Are there 3-APs that contain BOTH x and y?
                            # Yes, if they are part of the same 3-AP.
                            # But delta_add(new_mask, y) only counts 3-APs in new_mask.
                            # So any 3-AP containing x is NOT in new_mask.
                            # So delta_add and delta_remove are disjoint in terms of the 3-APs they count.
                            # Wait, delta_remove counts 3-APs in `mask` containing x.
                            # delta_add counts 3-APs in `new_mask` containing y.
                            # Since x is not in `new_mask`, no 3-AP counted by delta_add contains x.
                            # Since y is not in `mask` (before removal? No, y is not in mask), 
                            # no 3-AP counted by delta_remove contains y.
                            # So they are disjoint.
                            delta_rem = delta_3aps_remove(mask, x)
                            delta_add = delta_3aps_add(new_mask, y)
                            delta = delta_add - delta_rem
                            
                            if delta > 0:
                                mask = new_mask | (1 << y)
                                score += delta
                                improved = True
                                if score > best_score:
                                    best_score = score
                                    best_mask = mask
                                break # Restart search

    # Convert best_mask back to sorted list (1-indexed)
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    return sorted(result)
