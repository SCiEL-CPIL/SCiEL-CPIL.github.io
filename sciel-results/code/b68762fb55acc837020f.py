import time
import random
import math

N = 256

# Precomputation
# Bitmask representation: bit i corresponds to element i+1.
# Element 1 is bit 0, Element 256 is bit 255.

# 1. Precompute 4-AP conflicts
# For each element x, store a list of bitmasks representing the other 3 elements
# of every 4-AP containing x.
# If (mask & conflict_mask) == conflict_mask, then adding x creates a 4-AP.
conflict_masks_4 = [[] for _ in range(N + 1)]

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        # For each element, the "other 3" mask
        # a: b, c, e
        conflict_masks_4[a].append((1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1)))
        # b: a, c, e
        conflict_masks_4[b].append((1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1)))
        # c: a, b, e
        conflict_masks_4[c].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1)))
        # e: a, b, c
        conflict_masks_4[e].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)))

# 2. Precompute 3-AP partners
# For each element x, store a list of bitmasks representing the other 2 elements
# of every 3-AP containing x.
# Used to calculate delta in 3-AP count when adding/removing x.
partner_masks_3 = [[] for _ in range(N + 1)]

for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        # a: b, c
        partner_masks_3[a].append((1 << (b - 1)) | (1 << (c - 1)))
        # b: a, c
        partner_masks_3[b].append((1 << (a - 1)) | (1 << (c - 1)))
        # c: a, b
        partner_masks_3[c].append((1 << (a - 1)) | (1 << (b - 1)))

def count_3aps_full(mask):
    """Count total 3-APs in the mask. O(N^2) roughly, but fast enough for init/check."""
    count = 0
    # Iterate over all possible 3-APs
    # To be efficient, we can iterate over pairs (a,c) with same parity and check b?
    # Or just iterate over precomputed list of all 3-AP masks?
    # Let's precompute a list of all 3-AP masks for fast full count if needed.
    # Actually, for N=256, number of 3-APs is ~ sum_{d=1}^{128} (256-2d) ~ 128*256 - 2*128*129/2 ~ 32k.
    # Iterating 32k masks is fast.
    # But we can do better: for each x in mask, count partners?
    # That's O(|A|^2) which is ~ 1000^2 = 1M ops. Also fast.
    # Let's use the partner list approach for consistency, but summing over all elements in mask.
    # Wait, if I sum delta_add for all x in mask, I count each 3-AP 3 times?
    # No, delta_add(x) counts 3-APs where x is added.
    # If I build from scratch:
    # For each x in mask, count how many pairs (y,z) in mask form a 3-AP with x.
    # This counts each 3-AP exactly 3 times (once for each member).
    # So count = sum(delta_add(mask_without_x, x) for x in mask) / 3.
    # This is efficient.
    
    total = 0
    # Iterate over set bits
    m = mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        x = lsb.bit_length() # 1-indexed element
        # Calculate delta for adding x to (mask without x)
        base_mask = mask ^ lsb
        d = 0
        for pm in partner_masks_3[x]:
            if (base_mask & pm) == pm:
                d += 1
        total += d
        m ^= lsb
    return total // 3

def is_4ap_free_addition(mask, x):
    """Check if adding x to mask creates a 4-AP."""
    bit_x = 1 << (x - 1)
    if mask & bit_x:
        return False
    for cm in conflict_masks_4[x]:
        if (mask & cm) == cm:
            return False
    return True

def delta_3aps_addition(mask, x):
    """Calculate number of new 3-APs formed by adding x to mask."""
    delta = 0
    for pm in partner_masks_3[x]:
        if (mask & pm) == pm:
            delta += 1
    return delta

def greedy_construct(deadline, seed_val):
    """
    Construct a 4-AP-free set using a greedy heuristic.
    We prioritize adding elements that create the most new 3-APs.
    """
    random.seed(seed_val)
    mask = 0
    score = 0
    # We will iterate through potential additions.
    # To make it fast, we can just iterate 1..N and add if valid and beneficial?
    # Or use a priority queue?
    # Given N=256, a simple loop is fine.
    # Strategy:
    # 1. Start with empty set.
    # 2. At each step, evaluate all x not in mask.
    # 3. If valid (no 4-AP), calculate delta 3-APs.
    # 4. Pick the x with max delta. If max delta > 0, add it.
    # 5. If no x has positive delta, stop? Or pick random valid x to allow future gains?
    #    Usually, adding an element with 0 immediate gain might enable future gains.
    #    However, for 4-AP free sets, density is the main driver.
    #    Let's try: Add if valid. If multiple, pick max delta. If all deltas 0, pick random valid.
    
    # Precompute list of elements
    elements = list(range(1, N + 1))
    
    # To speed up, we can maintain a list of "active" candidates?
    # Or just loop over all N elements each time? N=256, steps ~ 100-200. 256*200 = 51k ops. Very fast.
    
    while time.time() < deadline:
        best_x = -1
        best_delta = -1
        valid_candidates = []
        
        for x in elements:
            if mask & (1 << (x - 1)):
                continue
            if not is_4ap_free_addition(mask, x):
                continue
            d = delta_3aps_addition(mask, x)
            valid_candidates.append((d, x))
            if d > best_delta:
                best_delta = d
                best_x = x
                
        if best_x == -1:
            break # No valid moves
            
        # If best_delta is 0, we might still want to add to explore structure,
        # but typically greedy stops or picks random.
        # Let's pick the best one. If best_delta == 0, pick random from valid_candidates with d=0?
        # Actually, if best_delta < 0 (impossible, delta >= 0), stop.
        
        # Add the best one
        mask |= (1 << (best_x - 1))
        score += best_delta
        
    return mask, score

def local_search(mask, score, deadline):
    """
    Fast local search.
    Moves:
    1. Add x (if valid and delta > 0)
    2. Remove x (if delta < 0, i.e., removing it increases score? No, removing always decreases score unless it unlocks adds? 
       Standard local search: only accept improvements.
       Removing x decreases score by delta_remove(x). So we only remove if it allows a better add?
       That's a swap.
    )
    
    Let's implement:
    1. Try all adds. If any valid add has delta > 0, do the best one.
    2. Try all swaps (remove x, add y). If valid and net_delta > 0, do the best one.
    3. If no improvement, try random moves with acceptance? (SA-like but restricted).
    4. Or just stop if no improvement (Hill Climbing).
    
    Given the landscape, Hill Climbing might get stuck. 
    Let's add a "shake" phase: random valid swaps even if neutral or slightly negative, to escape plateaus.
    """
    
    # Precompute set bits list for faster iteration?
    # For N=256, iterating 1..N is fine.
    
    no_improve_count = 0
    max_no_improve = 1000 # Shake after this many steps
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try Adds
        best_add_x = -1
        best_add_delta = 0
        for x in range(1, N + 1):
            if mask & (1 << (x - 1)):
                continue
            if not is_4ap_free_addition(mask, x):
                continue
            d = delta_3aps_addition(mask, x)
            if d > best_add_delta:
                best_add_delta = d
                best_add_x = x
        
        if best_add_x != -1 and best_add_delta > 0:
            mask |= (1 << (best_add_x - 1))
            score += best_add_delta
            improved = True
            no_improve_count = 0
            continue
            
        # 2. Try Swaps
        # Remove x, Add y
        # Net delta = delta_add(y, mask\x) - delta_remove(x, mask)
        # delta_remove(x, mask) is number of 3-APs containing x in mask.
        
        best_swap_x = -1
        best_swap_y = -1
        best_swap_delta = 0
        
        # To speed up, we can precompute delta_remove for all x in mask?
        # It's O(|A| * avg_partners).
        
        # Iterate x in mask
        # Iterate y not in mask
        # This is O(N^2) per step. 256^2 = 65k. 
        # If we do 1000 steps, 65M ops. Might be slow in Python.
        
        # Optimization: 
        # Only try swaps where x has low value and y has high potential?
        # Or just limit the search space.
        
        # Let's do a randomized subset of swaps to keep it fast.
        # Pick random x in mask, random y not in mask.
        # If valid and delta > 0, accept.
        # This is stochastic local search.
        
        # Actually, let's stick to deterministic "Best Improvement" but restricted.
        # For x in mask:
        #   For y in not_in_mask:
        #      Check validity of y in mask\x
        #      Calc delta
        #      Track best
        
        # To make it faster, we can precompute "removal cost" for all x in mask once per iteration?
        # removal_cost[x] = delta_remove(x, mask)
        
        removal_costs = {}
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length()
            # Calculate removal cost
            rc = 0
            for pm in partner_masks_3[x]:
                if (mask & pm) == pm:
                    rc += 1
            removal_costs[x] = rc
            m ^= lsb
            
        # Now iterate y not in mask
        # And check against each x?
        # This is still O(N^2).
        
        # Let's try a different approach for swaps:
        # Identify "bad" elements (high removal cost, low contribution to connectivity?)
        # And "good" candidates.
        
        # For now, let's do a limited random search for swaps.
        # 100 random swap attempts per iteration.
        
        found_swap = False
        for _ in range(100):
            # Pick random x in mask
            # Pick random y not in mask
            # To pick random x in mask efficiently:
            # Generate random number 0..N-1, check bit. Retry if not set.
            # Or maintain a list? List maintenance is O(N) per change.
            
            # Let's just pick random indices
            x = random.randint(1, N)
            if not (mask & (1 << (x - 1))):
                continue
                
            y = random.randint(1, N)
            if y == x or (mask & (1 << (y - 1))):
                continue
                
            # Check if y can be added to mask without x
            temp_mask = mask & ~(1 << (x - 1))
            if not is_4ap_free_addition(temp_mask, y):
                continue
                
            # Calculate delta
            # Remove x: lose removal_costs[x]
            # Add y: gain delta_add(y, temp_mask)
            d_add = 0
            for pm in partner_masks_3[y]:
                if (temp_mask & pm) == pm:
                    d_add += 1
                    
            d_rem = removal_costs[x]
            net_delta = d_add - d_rem
            
            if net_delta > 0:
                # Accept
                mask = temp_mask | (1 << (y - 1))
                score += net_delta
                improved = True
                no_improve_count = 0
                found_swap = True
                break
                
        if found_swap:
            continue
            
        # 3. No improvement found. 
        # Try "Shake": Accept a neutral or slightly negative move to escape local optimum?
        # Or restart?
        
        no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Shake: Randomly remove a few elements and re-greedy?
            # Or just accept a random valid swap with net_delta >= -1?
            # Let's try to accept a random valid swap with net_delta >= 0 (neutral)
            # If none, accept net_delta == -1?
            
            # For now, let's just break out of this local search and start a new one?
            # Or continue with higher "temperature" (accept negative moves)?
            # Let's implement a simple SA step here.
            
            T = 1.0 # Fixed temp for shake
            # Try 100 random moves
            moved = False
            for _ in range(100):
                # Try Add
                if random.random() < 0.5:
                    x = random.randint(1, N)
                    if mask & (1 << (x - 1)): continue
                    if not is_4ap_free_addition(mask, x): continue
                    d = delta_3aps_addition(mask, x)
                    # Accept if d > 0 or random < exp(d/T)
                    if d > 0 or random.random() < math.exp(d / T):
                        mask |= (1 << (x - 1))
                        score += d
                        moved = True
                        no_improve_count = 0
                        break
                else:
                    # Try Remove
                    x = random.randint(1, N)
                    if not (mask & (1 << (x - 1))): continue
                    d = -delta_3aps_addition(mask & ~(1 << (x - 1)), x) # Wait, delta_remove is same as delta_add on base mask
                    # Actually delta_remove(x, mask) = count of 3-APs containing x in mask.
                    # Which is delta_add(x, mask without x).
                    d_rem = 0
                    base = mask & ~(1 << (x - 1))
                    for pm in partner_masks_3[x]:
                        if (base & pm) == pm:
                            d_rem += 1
                    d = -d_rem
                    if d > 0 or random.random() < math.exp(d / T):
                        mask &= ~(1 << (x - 1))
                        score += d
                        moved = True
                        no_improve_count = 0
                        break
                        
            if moved:
                continue
                
            # If still no move, break to save time for next restart
            break
            
    return mask, score

def mask_to_list(mask):
    elements = []
    for i in range(N):
        if mask & (1 << i):
            elements.append(i + 1)
    return elements

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    best_mask = 0
    best_score = -1
    
    # Multi-start strategy
    # Run several greedy+local_search instances with different seeds
    # Allocate time: 10s per run? 8 runs?
    # Or dynamic: run until time/2, then refine best?
    
    num_starts = 8
    time_per_start = (deadline - start_time) / num_starts
    
    for i in range(num_starts):
        if time.time() > deadline - 5:
            break
            
        # Greedy phase
        g_deadline = time.time() + time_per_start * 0.2
        if g_deadline > deadline - 5:
            g_deadline = deadline - 5
            
        g_mask, g_score = greedy_construct(g_deadline, seed_val=i)
        
        # Local Search phase
        ls_deadline = time.time() + time_per_start * 0.8
        if ls_deadline > deadline - 5:
            ls_deadline = deadline - 5
            
        ls_mask, ls_score = local_search(g_mask, g_score, ls_deadline)
        
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask
            
    # Final Polish: Run local search on best_mask with remaining time
    if best_mask != 0 and time.time() < deadline - 2:
        # Refine
        final_mask, final_score = local_search(best_mask, best_score, deadline)
        if final_score > best_score:
            best_mask = final_mask
            best_score = final_score
            
    if best_mask == 0:
        # Fallback: Just return a simple set
        # E.g. all numbers with odd number of 1s in binary? (No 4-AP? Not sure)
        # Or just 1..10
        return list(range(1, 11))
        
    return mask_to_list(best_mask)

# EVOLVE-BLOCK-END
