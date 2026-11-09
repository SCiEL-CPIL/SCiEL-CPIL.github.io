"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

N = 256

# Precompute 4-AP forbidden sets: for each x, which other elements would create a 4-AP with x
# forbidden_4ap[x] is a bitmask of elements that, if present along with x, create a 4-AP
# Actually, we need: for a given mask and candidate x, does adding x create a 4-AP?
# A 4-AP (a, a+d, a+2d, a+3d) is forbidden.
# If we add x, it could be at position 0,1,2,3 of some 4-AP.
# Precompute for each x, the set of "witness" masks that would complete a 4-AP.

# Better approach: Precompute for each pair of elements (i,j) with i<j, 
# the set of differences d such that i,j are part of a 4-AP with step d.
# Then for checking if adding x creates a 4-AP, we check if any existing pair 
# combined with x completes a 4-AP.

# Even better: For each x, precompute all 4-APs that contain x.
# For each such 4-AP, store the other 3 elements.
# Then adding x creates a 4-AP iff any of those "other 3" sets is fully present in mask.

# Let's precompute: for each x in 1..256, list of frozensets of 3 elements 
# such that {x} union that set forms a 4-AP.

_4ap_witnesses = {}  # x -> list of bitmasks (each bitmask has exactly 3 bits set)

def _precompute_4ap_witnesses():
    global _4ap_witnesses
    for x in range(1, N+1):
        witnesses = []
        # x as position 0: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            elems = [x+d, x+2*d, x+3*d]
            if all(1 <= e <= N for e in elems):
                bm = 0
                for e in elems:
                    bm |= (1 << (e-1))
                witnesses.append(bm)
        # x as position 1: x-d, x, x+d, x+2d
        for d in range(1, min(x-1, (N-x)//2) + 1):
            elems = [x-d, x+d, x+2*d]
            if all(1 <= e <= N for e in elems):
                bm = 0
                for e in elems:
                    bm |= (1 << (e-1))
                witnesses.append(bm)
        # x as position 2: x-2d, x-d, x, x+d
        for d in range(1, min((x-1)//2, N-x) + 1):
            elems = [x-2*d, x-d, x+d]
            if all(1 <= e <= N for e in elems):
                bm = 0
                for e in elems:
                    bm |= (1 << (e-1))
                witnesses.append(bm)
        # x as position 3: x-3d, x-2d, x-d, x
        for d in range(1, (x-1)//3 + 1):
            elems = [x-3*d, x-2*d, x-d]
            if all(1 <= e <= N for e in elems):
                bm = 0
                for e in elems:
                    bm |= (1 << (e-1))
                witnesses.append(bm)
        _4ap_witnesses[x] = witnesses

_precompute_4ap_witnesses()

# Precompute 3-AP contributions: for each x, the bitmask of pairs (a,c) such that (a,x,c) is a 3-AP
# i.e., x is the middle. Also (x is start) and (x is end).
# For delta calculation:
# If we add x, the new 3-APs are those where x participates.
# x as middle: (x-d, x, x+d) -> need both x-d and x+d in mask
# x as start: (x, x+d, x+2d) -> need both x+d and x+2d in mask  
# x as end: (x-2d, x-d, x) -> need both x-2d and x-d in mask

# Precompute for each x:
# mid_pairs[x] = list of (a, c) where a < x < c and (a+x)/2 = x, i.e., a = x-d, c = x+d
# start_pairs[x] = list of (b, c) where b = x+d, c = x+2d
# end_pairs[x] = list of (a, b) where a = x-2d, b = x-d

_mid_pairs = {}
_start_pairs = {}
_end_pairs = {}

def _precompute_3ap_pairs():
    global _mid_pairs, _start_pairs, _end_pairs
    for x in range(1, N+1):
        mid = []
        for d in range(1, min(x-1, N-x) + 1):
            a = x - d
            c = x + d
            mid.append((a, c))
        _mid_pairs[x] = mid
        
        start = []
        for d in range(1, (N-x)//2 + 1):
            b = x + d
            c = x + 2*d
            start.append((b, c))
        _start_pairs[x] = start
        
        end = []
        for d in range(1, (x-1)//2 + 1):
            a = x - 2*d
            b = x - d
            end.append((a, b))
        _end_pairs[x] = end

_precompute_3ap_pairs()

def _count_3aps_bitmask(mask):
    """Count distinct 3-term APs in the set represented by bitmask."""
    count = 0
    # For each x as middle, count pairs (a,c) both in mask
    for x in range(1, N+1):
        if not ((mask >> (x-1)) & 1):
            continue
        for a, c in _mid_pairs[x]:
            if ((mask >> (a-1)) & 1) and ((mask >> (c-1)) & 1):
                count += 1
    return count

def _delta_3ap(mask, x, is_add):
    """Calculate change in 3-AP count when adding/removing x."""
    delta = 0
    # x as middle
    for a, c in _mid_pairs[x]:
        if ((mask >> (a-1)) & 1) and ((mask >> (c-1)) & 1):
            delta += 1
    # x as start
    for b, c in _start_pairs[x]:
        if ((mask >> (b-1)) & 1) and ((mask >> (c-1)) & 1):
            delta += 1
    # x as end
    for a, b in _end_pairs[x]:
        if ((mask >> (a-1)) & 1) and ((mask >> (b-1)) & 1):
            delta += 1
    return delta if is_add else -delta

def _creates_4ap(mask, x):
    """Check if adding x creates a 4-AP. x is NOT in mask."""
    for witness_bm in _4ap_witnesses[x]:
        if (mask & witness_bm) == witness_bm:
            return True
    return False

def _has_4ap_fast(mask):
    """Check if mask contains any 4-term AP."""
    # Check all 4-APs
    for d in range(1, (N)//3 + 1):
        for a in range(1, N - 3*d + 1):
            if all((mask >> (a + k*d - 1)) & 1 for k in range(4)):
                return True
    return False

def _construct_greedy():
    """Greedy construction of a set with no 4-APs."""
    mask = 0
    for x in range(1, N+1):
        new_mask = mask | (1 << (x-1))
        if not _creates_4ap(mask, x):
            mask = new_mask
    return mask

def _local_search_sa(mask, time_limit):
    """Simulated Annealing local search with precomputed structures."""
    start_time = time.time()
    best_mask = mask
    best_score = _count_3aps_bitmask(mask)
    current_mask = mask
    current_score = best_score
    
    # Track elements in set for efficient removal
    elements = set()
    for i in range(N):
        if (current_mask >> i) & 1:
            elements.add(i+1)
    
    # SA parameters
    T = 5.0
    cooling_rate = 0.99995
    min_T = 0.01
    
    while time.time() < start_time + time_limit:
        # Choose move type
        r = random.random()
        
        if r < 0.35:
            # Add element
            x = random.randint(1, N)
            if x in elements:
                continue
            if _creates_4ap(current_mask, x):
                continue
            delta = _delta_3ap(current_mask, x, True)
            new_score = current_score + delta
            if new_score > current_score or random.random() < np.exp((new_score - current_score) / T):
                current_mask |= (1 << (x-1))
                current_score = new_score
                elements.add(x)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        elif r < 0.6:
            # Remove element
            if not elements:
                continue
            x = random.choice(list(elements))
            delta = _delta_3ap(current_mask, x, False)
            new_score = current_score + delta
            if new_score > current_score or random.random() < np.exp((new_score - current_score) / T):
                current_mask &= ~(1 << (x-1))
                current_score = new_score
                elements.discard(x)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        else:
            # Swap
            if not elements:
                continue
            x_remove = random.choice(list(elements))
            x_add = random.randint(1, N)
            if x_add == x_remove or x_add in elements:
                continue
            # Check validity: remove x_remove, add x_add
            temp_mask = current_mask & ~(1 << (x_remove-1))
            if _creates_4ap(temp_mask, x_add):
                continue
            delta1 = _delta_3ap(current_mask, x_remove, False)
            delta2 = _delta_3ap(temp_mask, x_add, True)
            new_score = current_score + delta1 + delta2
            if new_score > current_score or random.random() < np.exp((new_score - current_score) / T):
                current_mask = temp_mask | (1 << (x_add-1))
                current_score = new_score
                elements.discard(x_remove)
                elements.add(x_add)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        
        T = max(min_T, T * cooling_rate)

    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Strict safety margin for 60s limit
    
    # Initial construction with multiple strategies
    best_initial_mask = 0
    best_initial_score = 0
    
    # Strategy 1: Standard greedy
    mask1 = _construct_greedy()
    score1 = _count_3aps_bitmask(mask1)
    best_initial_mask = mask1
    best_initial_score = score1
    
    # Strategy 2: Random order greedy
    for _ in range(15):
        if time.time() > deadline - 15:
            break
        random_order = list(range(1, N+1))
        random.shuffle(random_order)
        mask = 0
        elements = set()
        for x in random_order:
            if _creates_4ap(mask, x):
                continue
            mask |= (1 << (x-1))
            elements.add(x)
        score = _count_3aps_bitmask(mask)
        if score > best_initial_score:
            best_initial_score = score
            best_initial_mask = mask
    
    # Local search
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        best_mask = _local_search_sa(best_initial_mask, remaining_time - 2)
    else:
        best_mask = best_initial_mask
    
    result = [i+1 for i in range(N) if (best_mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
