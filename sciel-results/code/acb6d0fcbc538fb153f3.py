"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0  # 55s safety margin before 60s kill
    N = 256
    
    # Precompute 4-AP masks: mask_4ap[i] is a bitmask where bit j is set
    # if (i, i+d, i+2d, i+3d) or (i-3d, i-2d, i-d, i) or (i-d, i, i+d, i+2d) or (i-2d, i-d, i, i+d)
    # Actually, for checking if adding x creates a 4-AP, we need to know all 4-APs that contain x.
    # Let's precompute for each x, the list of (other positions) that would form a 4-AP with x.
    
    # A 4-AP is (a, a+d, a+2d, a+3d) with d > 0.
    # For a given x, x can be at position 0, 1, 2, or 3 in the 4-AP.
    # Precompute: for each x, a bitmask of "conflicts" - positions that, together with x, are part of some 4-AP.
    # But we need to know if adding x creates a NEW 4-AP given the current set.
    # Better: precompute all 4-APs as tuples (a, b, c, d) with a < b < c < d.
    # For each x, store which 4-APs contain x.
    
    # Actually, simpler: for each x, precompute a bitmask conflict_mask[x] where bit y is set
    # if x and y are both in some 4-AP. But that's not quite right either - we need to check
    # if adding x to the current set creates a 4-AP, which means there exist d>0 such that
    # three of {x, x±d, x±2d, x±3d} (in a 4-term AP pattern) are already in the set.
    
    # Let me think differently. For a 4-AP (a, a+d, a+2d, a+3d):
    # - If we're adding x, a 4-AP is created if x is one of the four terms AND the other three are already in the set.
    # So for each x, I need to enumerate all 4-APs that contain x, and for each such 4-AP,
    # check if the other three elements are in the current set.
    
    # Precompute: for each x (0-indexed), a list of lists. Each inner list contains the other 3 elements of a 4-AP containing x.
    ap4_containing = [[] for _ in range(N)]
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= N:
                break
            # 4-AP: (a, b, c, e)
            ap4_containing[a].append((b, c, e))
            ap4_containing[b].append((a, c, e))
            ap4_containing[c].append((a, b, e))
            ap4_containing[e].append((a, b, c))
    
    def has_4ap_after_add(mask, x):
        """Check if adding x to the set represented by mask creates a 4-AP."""
        for (y, z, w) in ap4_containing[x]:
            # Check if y, z, w are all in mask
            if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                return True
        return False
    
    def count_3aps(mask):
        """Count the number of 3-term APs in the set represented by mask."""
        count = 0
        for i in range(N):
            if not ((mask >> i) & 1):
                continue
            for d in range(1, N - i):
                j = i + d
                if j >= N:
                    break
                if not ((mask >> j) & 1):
                    continue
                k = j + d
                if k < N and ((mask >> k) & 1):
                    count += 1
        return count
    
    def delta_3aps_add(mask, x):
        """Compute the change in 3-AP count when adding x to the set."""
        delta = 0
        for i in range(x + 1):
            if not ((mask >> i) & 1):
                continue
            d = x - i
            j = x + d
            if j < N and ((mask >> j) & 1):
                delta += 1
        for j in range(x + 1, N):
            if not ((mask >> j) & 1):
                continue
            d = j - x
            i = x - d
            if i >= 0 and ((mask >> i) & 1):
                delta += 1
            # Also check if (x, j, j+d) forms a 3-AP
            k = j + d
            if k < N and ((mask >> k) & 1):
                delta += 1
        return delta
    
    # Local search with bitmask operations
    random.seed(42)
    
    best_ever_mask = 0
    best_ever_count = 0
    
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize with random subset
        mask = 0
        count = 0
        for i in range(N):
            if random.random() < 0.5:
                if not has_4ap_after_add(mask, i):
                    mask |= (1 << i)
                    count += delta_3aps_add(mask ^ (1 << i), i)  # delta before adding
        
        # Actually, let me recompute count properly
        count = count_3aps(mask)
        
        if count > best_ever_count:
            best_ever_count = count
            best_ever_mask = mask
        
        # Local search: try additions
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_gain = 0
            best_x = -1
            
            for x in range(N):
                if (mask >> x) & 1:
                    continue
                if has_4ap_after_add(mask, x):
                    continue
                gain = delta_3aps_add(mask, x)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x >= 0 and best_gain > 0:
                mask |= (1 << best_x)
                count += best_gain
                improved = True
                if count > best_ever_count:
                    best_ever_count = count
                    best_ever_mask = mask
            else:
                # Try removals to escape local optima (kick)
                if random.random() < 0.5:
                    # Remove a random element
                    bits = []
                    for i in range(N):
                        if (mask >> i) & 1:
                            bits.append(i)
                    if bits:
                        rem = random.choice(bits)
                        old_count = count
                        mask &= ~(1 << rem)
                        count = count_3aps(mask)
                        # Don't update best if we lost
                        improved = True
                    else:
                        break
        
        if time.time() > deadline - 5.0:
            break
    
    # Convert best mask to result
    result = [i + 1 for i in range(N) if (best_ever_mask >> i) & 1]
    if not result:
        result = [1]
    return sorted(result)
# EVOLVE-BLOCK-END
