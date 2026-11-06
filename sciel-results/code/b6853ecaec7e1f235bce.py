import time
import random

# EVOLVE-BLOCK-START

N = 512
# Precompute conflict masks for each x in [1, N]
# conflict_masks[x] is a list of masks (integers)
# If S & mask == mask, then x cannot be added to S
conflict_masks = [[] for _ in range(N + 1)]

# Build conflict masks
for x in range(1, N + 1):
    masks = []
    # x as 4th term: x-3d, x-2d, x-d, x
    # Need x-3d >= 1 => d <= (x-1)//3
    for d in range(1, (x - 1) // 3 + 1):
        a, b, c = x - 3*d, x - 2*d, x - d
        if a >= 1:
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            masks.append(mask)
    
    # x as 1st term: x, x+d, x+2d, x+3d
    # Need x+3d <= N => d <= (N-x)//3
    for d in range(1, (N - x) // 3 + 1):
        a, b, c = x + d, x + 2*d, x + 3*d
        if a <= N:
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            masks.append(mask)
    
    # x as 2nd term: x-d, x, x+d, x+2d
    # Need x-d >= 1 and x+2d <= N
    # d < x and 2d <= N-x => d <= (N-x)//2
    for d in range(1, min(x, (N - x) // 2 + 1)):
        a, b, c = x - d, x + d, x + 2*d
        if a >= 1 and c <= N:
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            masks.append(mask)
            
    # x as 3rd term: x-2d, x-d, x, x+d
    # Need x-2d >= 1 and x+d <= N
    # 2d < x and d <= N-x
    for d in range(1, min((x - 1) // 2 + 1, N - x + 1)):
        a, b, c = x - 2*d, x - d, x + d
        if a >= 1 and c <= N:
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            masks.append(mask)
            
    conflict_masks[x] = masks

def can_add(x, S_mask):
    """Check if x can be added to set represented by S_mask."""
    for mask in conflict_masks[x]:
        if S_mask & mask == mask:
            return False
    return True

def get_set_from_mask(S_mask):
    """Convert bit mask to set of integers."""
    s = set()
    m = S_mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        idx = lsb.bit_length() - 1
        s.add(idx + 1)
        m ^= lsb
    return s

def greedy_forward_bitset():
    S_mask = 0
    for x in range(1, N + 1):
        if can_add(x, S_mask):
            S_mask |= (1 << (x - 1))
    return S_mask

def greedy_random_bitset(seed):
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    S_mask = 0
    for x in order:
        if can_add(x, S_mask):
            S_mask |= (1 << (x - 1))
    return S_mask

def local_improve_bitset(S_mask, deadline):
    """
    Local search to maximize size.
    1. Try to add any valid element.
    2. If stuck, try swaps (remove x, add y) to reach a state where addition is possible.
    """
    S_mask = S_mask
    S_set = get_set_from_mask(S_mask)
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        
        # Phase 1: Add any possible element
        added = False
        for x in range(1, N + 1):
            if not (S_mask & (1 << (x - 1))):
                if can_add(x, S_mask):
                    S_mask |= (1 << (x - 1))
                    added = True
                    improved = True
        if added:
            continue # Restart search with new set
            
        # Phase 2: If no addition possible, try swaps
        # We want to find a swap (x in S, y not in S) such that S' = S - {x} + {y} is valid.
        # And ideally, S' allows adding some z later.
        # For now, just accept any valid swap to explore the space.
        # To be more directed, we can prioritize swaps that unblock more elements.
        
        # Simple approach: Random restart of swaps if stuck?
        # Or deterministic: Try all x in S, find y that can be added to S-{x}.
        
        found_swap = False
        # To avoid getting stuck in a loop, we can randomize the order of x
        x_list = list(get_set_from_mask(S_mask))
        random.shuffle(x_list)
        
        for x in x_list:
            if time.time() > deadline - 1:
                break
            S_without_x = S_mask & ~(1 << (x - 1))
            
            # Find a y not in S that can be added to S_without_x
            # Prefer y that allows more future additions?
            # For now, just pick the first valid y
            # Or pick y that minimizes "conflict" with remaining S?
            
            best_y = -1
            best_score = -1
            
            # Iterate y not in S
            # Optimization: Only check y that are "close" to x or have low index?
            # Just iterate all y for now, N=512 is small.
            for y in range(1, N + 1):
                if y == x:
                    continue
                if S_mask & (1 << (y - 1)):
                    continue
                if can_add(y, S_without_x):
                    # Score: How many new elements can be added to S_without_x + y?
                    # This is expensive.
                    # Simple score: 0
                    # Or: Check if y "unblocks" something.
                    # Let's just accept the first one to keep it fast.
                    best_y = y
                    break
                    
            if best_y != -1:
                # Perform swap
                S_mask = S_without_x | (1 << (best_y - 1))
                found_swap = True
                improved = True
                break # Restart the outer loop to try additions again
                
    return S_mask

def construct():
    deadline = time.time() + 85
    
    # 1. Greedy Initializations
    best_mask = greedy_forward_bitset()
    best_size = bin(best_mask).count('1')
    
    # Try a few random greedy orders
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        m = greedy_random_bitset(seed)
        s = bin(m).count('1')
        if s > best_size:
            best_mask = m
            best_size = s
            
    # 2. Local Improvement on best
    if time.time() < deadline - 5:
        improved_mask = local_improve_bitset(best_mask, deadline)
        if bin(improved_mask).count('1') > best_size:
            best_mask = improved_mask
            best_size = bin(best_mask).count('1')
            
    # 3. Final Verification and Conversion
    final_set = get_set_from_mask(best_mask)
    
    # Verify 4-AP free
    s = final_set
    for a in s:
        for d in range(1, (N - a) // 3 + 1):
            if (a + d) in s and (a + 2*d) in s and (a + 3*d) in s:
                # Fallback
                final_set = get_set_from_mask(greedy_forward_bitset())
                break
    
    return sorted(final_set)

# EVOLVE-BLOCK-END
