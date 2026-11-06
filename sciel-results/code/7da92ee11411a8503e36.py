import time
import random

# Precompute 4-APs for N=512
N = 512
# A 4-AP is (a, a+d, a+2d, a+3d)
# We represent the set as an integer bitmask.
# bit i represents element i+1 (since range is 1..512)
# Actually, let's use 0-indexed bits for 0..511 corresponding to 1..512.
# Element x in 1..512 maps to bit x-1.

ap_masks = []
# For each d, find all a such that a+3d <= 512
for d in range(1, (N - 3) // 3 + 1):
    max_a = N - 3 * d
    for a in range(1, max_a + 1):
        # Elements are a, a+d, a+2d, a+3d
        # Bits are a-1, a+d-1, a+2d-1, a+3d-1
        mask = 0
        mask |= 1 << (a - 1)
        mask |= 1 << (a + d - 1)
        mask |= 1 << (a + 2 * d - 1)
        mask |= 1 << (a + 3 * d - 1)
        ap_masks.append(mask)

# Also precompute for each element x, which APs contain it
# This is useful for faster conflict analysis, but for now, simple validity check is fine.
# To speed up "can add x", we can precompute APs containing x.
aps_containing = [[] for _ in range(N + 1)]
for i, mask in enumerate(ap_masks):
    # Find set bits in mask
    m = mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        x = idx + 1
        aps_containing[x].append(i)
        m ^= lsb

def is_valid(S):
    """Check if bitmask S contains no 4-AP."""
    for mask in ap_masks:
        if (S & mask) == mask:
            return False
    return True

def greedy_construct():
    S = 0
    for x in range(1, N + 1):
        bit = 1 << (x - 1)
        # Check if adding x creates a 4-AP
        # Only need to check APs containing x
        valid = True
        for idx in aps_containing[x]:
            mask = ap_masks[idx]
            if (S & mask) == (mask ^ bit):
                # If S contains the other 3 bits, adding x completes the AP
                valid = False
                break
        if valid:
            S |= bit
    return S

def local_search(S, time_limit):
    """
    Local search to improve S.
    Strategy:
    1. Try to add any element that doesn't create a 4-AP.
    2. If no such element, try swaps (remove v, add u) to change configuration.
    """
    start_time = time.time()
    S = S
    current_size = S.bit_count()
    
    # Precompute list of bits for faster iteration?
    # We can just iterate 1..N
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Try to add
        # Iterate through elements not in S
        # To avoid bias, shuffle the order of attempts occasionally?
        # For now, try all addable elements.
        addable = []
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            if S & bit:
                continue
            # Check if adding x is valid
            valid = True
            for idx in aps_containing[x]:
                mask = ap_masks[idx]
                if (S & mask) == (mask ^ bit):
                    valid = False
                    break
            if valid:
                addable.append(x)
                
        if addable:
            # Pick one. Maybe pick the one that "blocks" the fewest future adds?
            # Or just pick the first one.
            # Let's pick the one with smallest index for determinism in greedy,
            # but for local search, maybe random?
            # Let's pick the first one found.
            x = addable[0]
            S |= 1 << (x - 1)
            current_size += 1
            improved = True
            # Continue to see if we can add more immediately
            continue 
        else:
            # No single add is possible. Try swaps.
            # Pick random u not in S, random v in S
            # Check if S' = S \ {v} U {u} is valid.
            # To make this efficient, we can check validity of S' directly.
            
            # Get list of bits in S
            # We can maintain a list of elements in S to pick v from
            # But S is an int.
            
            # Let's try a few random swaps
            found_swap = False
            for _ in range(100): # Limit swaps per iteration to avoid infinite loops if stuck
                # Pick random u not in S
                u = random.randint(1, N)
                u_bit = 1 << (u - 1)
                if S & u_bit:
                    continue
                    
                # Pick random v in S
                # We need to pick a random set bit.
                # We can use random bits from S.
                # Or maintain a list. Let's use a list for simplicity and speed of random access.
                # But converting int to list every time is slow.
                # Let's just pick random v in 1..N and check if in S.
                v = random.randint(1, N)
                v_bit = 1 << (v - 1)
                if not (S & v_bit):
                    continue
                
                # Try swap
                S_new = S ^ v_bit | u_bit
                
                # Check validity of S_new
                # Optimization: Only check APs that are affected by the change.
                # APs containing u or v.
                # An AP is invalid in S_new if it was valid in S (and not broken by removing v) 
                # OR if it is newly formed by adding u.
                # Actually, S was valid.
                # Removing v can only break APs.
                # Adding u can only form APs.
                # So we only need to check APs that contain u.
                # Because any AP that was valid in S is either:
                # 1. Not containing v: still valid in S_new (BAD, S was valid so this shouldn't happen)
                # 2. Containing v: broken in S_new (GOOD)
                # Wait, S was valid, so no AP was fully contained in S.
                # So removing v doesn't "break" a valid AP, it just removes an element.
                # The only way S_new becomes invalid is if adding u creates a NEW 4-AP.
                # So we only need to check APs containing u.
                
                valid = True
                for idx in aps_containing[u]:
                    mask = ap_masks[idx]
                    if (S_new & mask) == mask:
                        valid = False
                        break
                
                if valid:
                    S = S_new
                    found_swap = True
                    improved = True
                    break # Restart the add/swap loop with new S
            
            if not improved:
                # If no swap worked in 100 tries, maybe we are stuck in a local minimum
                # that requires multiple swaps? Or just random restart?
                # For now, continue trying. If we loop forever, the time limit will kill it.
                # To avoid wasting time, we could do a random restart if stuck for too long.
                pass

    return S

def construct():
    # Deadline guard
    start_time = time.time()
    total_time_limit = 58.0 # 60s limit, keep 2s margin
    
    # 1. Greedy Initialization
    S = greedy_construct()
    current_size = S.bit_count()
    
    # 2. Local Search
    # We can run local search until time runs out.
    S = local_search(S, total_time_limit - (time.time() - start_time))
    
    # Convert bitmask to sorted list
    result = []
    m = S
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    return result
