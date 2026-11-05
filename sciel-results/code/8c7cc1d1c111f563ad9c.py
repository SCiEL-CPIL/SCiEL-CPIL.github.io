"""
Structural change: bitmask-based local search with delta evaluation.
The key insight from inspirations is that bitmask operations are much faster
than set membership checks, allowing more iterations within the time budget.

We use:
1. A bitmask representation of the subset
2. Precomputed 4-AP conflict masks for fast feasibility checking
3. Delta-based 3-AP counting for fast objective evaluation
4. Multi-restart local search with random initialization + greedy polish
5. Deadline guard to ensure we return before the 90s limit
"""
import time
import random
import numpy as np

N = 256  # ambient size [1, 256]

# Precompute all 4-term APs (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= 256
# For each element x, store the mask of elements that form a 4-AP with x as a member
# Actually, for feasibility: adding x to set S is invalid if there exist a, d such that
# {a, a+d, a+2d, a+3d} contains x and at least 3 other elements from S.
# Simpler: for each x, precompute conflict_mask[x] = set of y such that x and y are in some 4-AP
# No wait, the constraint is: no 4 distinct elements forming a 4-AP.
# So adding x is invalid if there exists a 4-AP {x, y1, y2, y3} where y1,y2,y3 are all in S.

# Let's precompute for each x, the list of 4-APs that contain x.
# For fast checking: adding x is OK iff for every 4-AP containing x, at most 2 of the other 3 are in S.

# Precomputation: for each x in 1..256, list of tuples (y1, y2, y3) where {x,y1,y2,y3} is a 4-AP
# with x being one of the four positions.

def precompute_conflicts():
    """For each x (0-indexed: x-1), precompute all triples (a,b,c) such that {x,a,b,c} forms a 4-AP."""
    conflicts = [[] for _ in range(N)]
    for d in range(1, (N-1)//3 + 1):
        for a in range(1, N - 3*d + 1):
            ap = [a, a+d, a+2*d, a+3*d]
            for pos, x in enumerate(ap):
                others = [ap[i] for i in range(4) if i != pos]
                conflicts[x-1].append(tuple(others))
    return conflicts

CONFLICTS = precompute_conflicts()

def count_3aps_mask(mask):
    """Count distinct 3-term APs (a, a+d, a+2d) with positive d, all in mask."""
    count = 0
    for d in range(1, N//2 + 1):
        for a in range(1, N - 2*d + 1):
            if (mask >> (a-1)) & 1 and (mask >> (a+d-1)) & 1 and (mask >> (a+2*d-1)) & 1:
                count += 1
    return count

def is_valid_addition(mask, x):
    """Check if adding x (1-indexed) to mask keeps it 4-AP-free."""
    xi = x - 1
    for (a, b, c) in CONFLICTS[xi]:
        # Check if all three are in mask
        if ((mask >> (a-1)) & 1) and ((mask >> (b-1)) & 1) and ((mask >> (c-1)) & 1):
            return False
    return True

def delta_3aps_addition(mask, x):
    """Compute change in 3-AP count when adding x to mask."""
    delta = 0
    xi = x - 1
    # x as first term: x, x+d, x+2d
    for d in range(1, (N - x + 1) // 2 + 1):
        if (mask >> (x+d-1)) & 1 and (mask >> (x+2*d-1)) & 1:
            delta += 1
    # x as middle term: x-d, x, x+d
    for d in range(1, min(x-1, N-x) + 1):
        if (mask >> (x-d-1)) & 1 and (mask >> (x+d-1)) & 1:
            delta += 1
    # x as last term: x-2d, x-d, x
    for d in range(1, (x-1) // 2 + 1):
        if (mask >> (x-2*d-1)) & 1 and (mask >> (x-d-1)) & 1:
            delta += 1
    return delta

def greedy_initial():
    """Greedy construction: add elements that maximize 3-AP gain while maintaining 4-AP-free."""
    mask = 0
    for x in range(1, N+1):
        if is_valid_addition(mask, x):
            mask |= (1 << (x-1))
    return mask

def greedy_improve(mask):
    """Try to add more elements greedily in random order."""
    order = list(range(1, N+1))
    random.shuffle(order)
    for x in order:
        if not ((mask >> (x-1)) & 1):
            if is_valid_addition(mask, x):
                mask |= (1 << (x-1))
    return mask

def local_search(mask, max_iter=200000, time_limit=None):
    """Local search: try adding or swapping elements to improve 3-AP count."""
    best_mask = mask
    best_count = count_3aps_mask(mask)
    
    if time_limit is None:
        time_limit = time.time() + 5.0
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        if time_limit and time.time() > time_limit:
            break
        
        # Try adding elements
        added = False
        for x in range(1, N+1):
            if not ((best_mask >> (x-1)) & 1):
                if is_valid_addition(best_mask, x):
                    new_mask = best_mask | (1 << (x-1))
                    new_count = best_count + delta_3aps_addition(best_mask, x)
                    if new_count > best_count:
                        best_mask = new_mask
                        best_count = new_count
                        improved = True
                        added = True
        if added:
            continue
        
        # Try swaps: remove one, add another
        for x_out in range(1, N+1):
            if not ((best_mask >> (x_out-1)) & 1):
                continue
            for x_in in range(1, N+1):
                if (best_mask >> (x_in-1)) & 1:
                    continue
                # Remove x_out, check if x_in can be added
                mask_wo = best_mask & ~(1 << (x_out-1))
                if is_valid_addition(mask_wo, x_in):
                    # Compute delta: -3APs lost by removing x_out + 3APs gained by adding x_in
                    # 3APs containing x_out that are lost (only those where all 3 are in mask_wo... 
                    # actually we need to count 3APs that include x_out and at least one other in mask_wo)
                    # Simpler: compute new count directly for this pair (expensive but rare)
                    new_mask = mask_wo | (1 << (x_in-1))
                    new_count = count_3aps_mask(new_mask)
                    if new_count > best_count:
                        best_mask = new_mask
                        best_count = new_count
                        improved = True
                        break
            if improved:
                break
    
    return best_mask, best_count

def multi_restart_local_search(total_time=85.0):
    """Multiple restarts with different random seeds, keeping best result."""
    deadline = time.time() + total_time
    best_mask = 0
    best_count = 0
    
    # Start with greedy
    mask = greedy_initial()
    mask = greedy_improve(mask)
    mask, cnt = local_search(mask, max_iter=50000, time_limit=min(deadline, time.time()+5))
    if cnt > best_count:
        best_mask = mask
        best_count = cnt
    
    # Multiple restarts
    restart = 0
    while time.time() < deadline - 3.0:
        restart += 1
        # Random subset initialization
        mask = 0
        # Start with a random subset of ~40-60% density
        for x in range(1, N+1):
            if random.random() < 0.5:
                if is_valid_addition(mask, x):
                    mask |= (1 << (x-1))
        
        # Greedy polish
        mask = greedy_improve(mask)
        
        # Local search with tighter time
        remaining = deadline - 3.0 - time.time()
        if remaining < 1.0:
            break
        mask, cnt = local_search(mask, max_iter=30000, time_limit=time.time() + min(remaining*0.9, 8.0))
        
        if cnt > best_count:
            best_mask = mask
            best_count = cnt
    
    return best_mask, best_count

def mask_to_list(mask):
    """Convert bitmask to sorted list of 1-indexed elements."""
    return [i+1 for i in range(N) if (mask >> i) & 1]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin for 90s limit
    
    best_mask, best_count = multi_restart_local_search(total_time=85.0)
    
    # Ensure we have at least one element
    if best_mask == 0:
        best_mask = 1  # element 1
    
    result = mask_to_list(best_mask)
    if not result:
        result = [1]
    
    return sorted(result)
