import time
import random

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Precompute masks for elements 1..n
    # Element x corresponds to bit (x-1)
    
    # Precompute 3-APs
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps_3.append((a, b, c))
            
    # Precompute neighbors for gain calculation
    # three_ap_neighbors[x] = list of (a, c) such that (a, x, c) is a 3-AP
    three_ap_neighbors = [[] for _ in range(n + 1)]
    for a, b, c in aps_3:
        # b is the middle
        three_ap_neighbors[b].append((a, c))
        # a is the first
        # c is the last
        # For gain, we only care about the middle element's perspective?
        # No, gain is symmetric. If we add x, we form 3-APs where x is a, b, or c.
        # So for each x, we need to know all pairs (y, z) such that {x, y, z} is a 3-AP.
        # If x is a: (x, x+d, x+2d) -> neighbors (x+d, x+2d)
        # If x is b: (x-d, x, x+d) -> neighbors (x-d, x+d)
        # If x is c: (x-2d, x-d, x) -> neighbors (x-2d, x-d)
        
    # Let's re-populate three_ap_neighbors correctly
    for x in range(1, n + 1):
        three_ap_neighbors[x] = []
    for a, b, c in aps_3:
        three_ap_neighbors[a].append((b, c))
        three_ap_neighbors[b].append((a, c))
        three_ap_neighbors[c].append((a, b))

    # Precompute 4-AP conflicts
    # four_aps_with_x[x] = list of masks of the other 3 elements
    four_aps_with_x = [[] for _ in range(n + 1)]
    for d in range(1, (n - 1) // 3 + 1):
        for s in range(1, n - 3 * d + 1):
            ap = (s, s + d, s + 2 * d, s + 3 * d)
            for i, elem in enumerate(ap):
                others = [ap[j] for j in range(4) if j != i]
                mask = 0
                for o in others:
                    mask |= (1 << (o - 1))
                four_aps_with_x[elem].append(mask)

    def can_add(x, mask):
        for others_mask in four_aps_with_x[x]:
            if (mask & others_mask) == others_mask:
                return False
        return True

    def gain_if_add(x, mask):
        gain = 0
        for a, c in three_ap_neighbors[x]:
            if (mask & (1 << (a - 1))) and (mask & (1 << (c - 1))):
                gain += 1
        return gain

    def count_3aps(mask):
        count = 0
        for a, b, c in aps_3:
            if (mask & (1 << (a - 1))) and (mask & (1 << (b - 1))) and (mask & (1 << (c - 1))):
                count += 1
        return count

    def local_search(initial_mask, max_iters=10000):
        mask = initial_mask
        current_score = count_3aps(mask)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try adding elements
            best_gain = 0
            best_x = -1
            for x in range(1, n + 1):
                if not (mask & (1 << (x - 1))):
                    if can_add(x, mask):
                        g = gain_if_add(x, mask)
                        if g > best_gain:
                            best_gain = g
                            best_x = x
            if best_x != -1 and best_gain > 0:
                mask |= (1 << (best_x - 1))
                current_score += best_gain
                improved = True
            
            # Try swapping elements
            # This is more expensive, so do it less frequently or with a limit
            # For each x in mask, for each y not in mask, check swap
            # Optimization: only check swaps for x with low "degree" or y with high potential
            # For now, just iterate all pairs, but break early if time is tight
            if time.time() < deadline - 10:
                swap_improved = False
                for x in range(1, n + 1):
                    if time.time() > deadline - 10:
                        break
                    if not (mask & (1 << (x - 1))):
                        continue
                    mask_without_x = mask & ~(1 << (x - 1))
                    # Calculate score loss of removing x
                    # We don't have loss_if_remove precomputed, but we can calculate gain of adding y to mask_without_x
                    # and compare with current_score
                    # Actually, new_score = current_score - loss(x) + gain(y)
                    # loss(x) = number of 3-APs containing x where other 2 are in mask
                    # Let's just compute new_score directly if a swap is a candidate?
                    # That's O(N) per swap check. Too slow.
                    
                    # Heuristic: Only try swapping x if x is "expensive" (blocks many adds) or y is "valuable"
                    # Let's try a simpler swap: remove x, then greedily add best y
                    # This is essentially "remove and re-optimize"
                    # Let's stick to single swaps but optimize the check.
                    pass # Skipping complex swaps for now to ensure speed, relying on add/remove logic
                
                # Try removing elements that allow better additions
                # This is handled by the "add" phase if we allow removing first?
                # No, the add phase only adds.
                # Let's add a "remove" phase.
                # Try removing x if it reduces score by 0 or 1, and allows adding multiple elements?
                # Too complex. Let's just do hill climbing on adds.
                
            if not improved:
                break
                
        return mask, current_score

    best_mask = 0
    best_score = 0
    
    # Strategy 1: Greedy construction
    mask = 0
    # Random order
    random.seed(42)
    order = list(range(1, n + 1))
    random.shuffle(order)
    for x in order:
        if can_add(x, mask):
            mask |= (1 << (x - 1))
    mask, score = local_search(mask, max_iters=5000)
    if score > best_score:
        best_score = score
        best_mask = mask

    # Strategy 2: Random restarts
    for restart in range(20):
        if time.time() > deadline - 15:
            break
        random.seed(42 + restart)
        order = list(range(1, n + 1))
        random.shuffle(order)
        mask = 0
        for x in order:
            if can_add(x, mask):
                mask |= (1 << (x - 1))
        mask, score = local_search(mask, max_iters=3000)
        if score > best_score:
            best_score = score
            best_mask = mask

    # Strategy 3: Start with dense set and prune
    if time.time() < deadline - 10:
        mask = (1 << n) - 1
        # Prune 4-APs
        # This is hard to do directly with bitmask without iterating 4-APs
        # Let's just use the greedy result as base
        
    # Final polish
    if time.time() < deadline - 5:
        best_mask, best_score = local_search(best_mask, max_iters=5000)

    # Convert mask to sorted list
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
