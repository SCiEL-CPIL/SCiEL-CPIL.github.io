"""Construct a subset of [1,256] with no 4-term AP, maximizing 3-term AP count."""
import time
import random

N = 256
# Precompute: for each element x and difference d, the 4-AP {x, x+d, x+2d, x+3d}
# and 3-AP {x, x+d, x+2d}
# Use bitmasks for fast membership testing (256 bits fits in Python int)

def _precompute():
    # For 4-AP violation checking: for each x, list of (d, mask) where mask has bits
    # set for x, x+d, x+2d, x+3d (all must be in set to form 4-AP)
    four_aps = []  # list of (x, d, mask)
    for x in range(1, N + 1):
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            a, b, c, e = x, x + d, x + 2*d, x + 3*d
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << e)
            four_aps.append((x, d, mask))
    
    # For 3-AP counting: for each x, list of (d, mask) for {x, x+d, x+2d}
    three_aps = []
    for x in range(1, N + 1):
        max_d = (N - x) // 2
        for d in range(1, max_d + 1):
            a, b, c = x, x + d, x + 2*d
            mask = (1 << a) | (1 << b) | (1 << c)
            three_aps.append((x, d, mask))
    
    return four_aps, three_aps

FOUR_APS, THREE_APS = _precompute()

def _has_4ap(mask):
    """Check if bitmask contains any 4-term AP."""
    for x, d, fm in FOUR_APS:
        if (mask & fm) == fm:
            return True
    return False

def _count_3ap(mask):
    """Count distinct 3-term APs in the set represented by bitmask."""
    count = 0
    for x, d, tm in THREE_APS:
        if (mask & tm) == tm:
            count += 1
    return count

def _construct_greedy(seed, time_limit):
    """Greedy construction with randomized order, avoiding 4-APs."""
    random.seed(seed)
    order = list(range(1, N + 1))
    random.shuffle(order)
    
    mask = 0
    # Build set greedily: add elements in random order if they don't create a 4-AP
    for x in order:
        bit = 1 << x
        new_mask = mask | bit
        # Check if adding x creates a 4-AP
        # Only need to check 4-APs that include x
        creates_4ap = False
        for x2, d, fm in FOUR_APS:
            if (fm & bit) != 0:  # this 4-AP includes x
                if (new_mask & fm) == fm:
                    creates_4ap = True
                    break
        if not creates_4ap:
            mask = new_mask
    
    return mask

def _local_search(mask, time_limit, deadline):
    """Local search: try additions, removals, and swaps to increase 3-AP count."""
    best_mask = mask
    best_count = _count_3ap(mask)
    
    while time.time() < deadline:
        improved = False
        
        # Try adding each element not in the set
        for x in range(1, N + 1):
            if time.time() >= deadline:
                break
            bit = 1 << x
            if (mask & bit) != 0:
                continue
            new_mask = mask | bit
            if not _has_4ap(new_mask):
                new_count = _count_3ap(new_mask)
                if new_count > best_count:
                    best_mask = new_mask
                    best_count = new_count
                    improved = True
                    mask = new_mask
        
        # Try removing each element in the set (to allow better additions)
        for x in range(1, N + 1):
            if time.time() >= deadline:
                break
            bit = 1 << x
            if (mask & bit) == 0:
                continue
            new_mask = mask & ~bit
            new_count = _count_3ap(new_mask)
            # Only keep removal if it allows improvement later; for now just track
            # Actually, removal alone decreases count, so skip unless combined with add
        
        # Try swaps: remove x, add y
        for x in range(1, N + 1):
            if time.time() >= deadline:
                break
            bit_x = 1 << x
            if (mask & bit_x) == 0:
                continue
            for y in range(1, N + 1):
                if time.time() >= deadline:
                    break
                if x == y:
                    continue
                bit_y = 1 << y
                if (mask & bit_y) != 0:
                    continue
                new_mask = (mask & ~bit_x) | bit_y
                if not _has_4ap(new_mask):
                    new_count = _count_3ap(new_mask)
                    if new_count > best_count:
                        best_mask = new_mask
                        best_count = new_count
                        improved = True
                        mask = new_mask
                        break  # restart swap search with new mask
        
        if not improved:
            break
    
    return best_mask, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin before 90s kill
    
    best_mask = 0
    best_count = 0
    best_result = []
    
    # Phase 1: Multiple greedy constructions with different seeds
    seed = 0
    while time.time() < deadline - 20:  # leave 20s for local search
        mask = _construct_greedy(seed, deadline)
        count = _count_3ap(mask)
        if count > best_count:
            best_mask = mask
            best_count = count
        seed += 1
    
    # Phase 2: Local search on the best greedy result
    if time.time() < deadline - 2:
        best_mask, best_count = _local_search(best_mask, deadline, deadline - 2)
    
    # Convert bitmask to sorted list
    result = [x for x in range(1, N + 1) if (best_mask >> x) & 1]
    
    if not result:
        # Fallback: return a valid small set
        result = [1]
    
    return result

# Pre-run to warm up (optional, but ensures _precompute is done)
# Actually _precompute is done at module load
