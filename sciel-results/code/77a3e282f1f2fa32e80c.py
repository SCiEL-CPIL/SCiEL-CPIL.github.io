import time
import random

# Precomputation
N = 256
four_aps = []
three_aps = []

# For each element, store the bitmask of the *other* elements in each 4-AP containing it
# This allows fast checking: if (current_mask & other_mask) == other_mask, adding x creates a 4-AP
element_to_4ap_others = {i: [] for i in range(1, N+1)}
# For each element, store the bitmask of the *other* elements in each 3-AP containing it
# This allows fast counting of new 3-APs
element_to_3ap_others = {i: [] for i in range(1, N+1)}

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        four_aps.append((a, b, c, e))
        # For each element in the 4-AP, store the mask of the other three
        element_to_4ap_others[a].append((1 << (b-1)) | (1 << (c-1)) | (1 << (e-1)))
        element_to_4ap_others[b].append((1 << (a-1)) | (1 << (c-1)) | (1 << (e-1)))
        element_to_4ap_others[c].append((1 << (a-1)) | (1 << (b-1)) | (1 << (e-1)))
        element_to_4ap_others[e].append((1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)))

for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        three_aps.append((a, b, c))
        # For each element in the 3-AP, store the mask of the other two
        element_to_3ap_others[a].append((1 << (b-1)) | (1 << (c-1)))
        element_to_3ap_others[b].append((1 << (a-1)) | (1 << (c-1)))
        element_to_3ap_others[c].append((1 << (a-1)) | (1 << (b-1)))

def is_4ap_free_addition(A_mask, x):
    """Check if adding x to A_mask creates a 4-AP."""
    for other_mask in element_to_4ap_others[x]:
        if (A_mask & other_mask) == other_mask:
            return False
    return True

def delta_3aps_for_addition(A_mask, x):
    """Calculate how many new 3-APs are created by adding x to A_mask."""
    count = 0
    for other_mask in element_to_3ap_others[x]:
        if (A_mask & other_mask) == other_mask:
            count += 1
    return count

def count_3aps_mask(A_mask):
    """Count distinct 3-APs in the set represented by A_mask."""
    count = 0
    for a, b, c in three_aps:
        if (A_mask & (1 << (a-1))) and (A_mask & (1 << (b-1))) and (A_mask & (1 << (c-1))):
            count += 1
    return count

def greedy_initial_mask(seed_val=0):
    """Generate a 4-AP free set using a greedy approach with optional randomization."""
    random.seed(seed_val)
    A_mask = 0
    candidates = list(range(1, N + 1))
    if seed_val % 2 == 0:
        random.shuffle(candidates)
    for x in candidates:
        if is_4ap_free_addition(A_mask, x):
            A_mask |= (1 << (x - 1))
    return A_mask

def local_search_mask(A_mask_init, deadline):
    """
    Perform local search to maximize 3-APs while maintaining 4-AP freedom.
    """
    A_mask = A_mask_init
    current_score = count_3aps_mask(A_mask)
    
    while time.time() < deadline:
        improved = False
        
        # Try Additions
        best_add_x = -1
        best_add_delta = 0
        for x in range(1, N + 1):
            if A_mask & (1 << (x - 1)):
                continue
            if is_4ap_free_addition(A_mask, x):
                delta = delta_3aps_for_addition(A_mask, x)
                if delta > best_add_delta:
                    best_add_delta = delta
                    best_add_x = x
        
        if best_add_x != -1:
            A_mask |= (1 << (best_add_x - 1))
            current_score += best_add_delta
            improved = True
            continue
        
        # Try Swaps
        best_swap = None
        best_swap_score = current_score
        
        # Optimization: Only consider swaps that might improve the score
        # We iterate over x in A, and for each x, we find the best y to swap in.
        
        for x in range(1, N + 1):
            if not (A_mask & (1 << (x - 1))):
                continue
            
            A_mask_without_x = A_mask & ~(1 << (x - 1))
            
            # Calculate L_x: number of 3-APs lost by removing x
            L_x = 0
            for other_mask in element_to_3ap_others[x]:
                if (A_mask_without_x & other_mask) == other_mask:
                    L_x += 1
            
            # If L_x is very high, swapping x out is likely bad, but we still check
            # We can prune if L_x is greater than the maximum possible gain from any y
            # Max gain is roughly the number of 3-APs y can form, which is bounded.
            
            for y in range(1, N + 1):
                if y == x or (A_mask & (1 << (y - 1))):
                    continue
                
                if not is_4ap_free_addition(A_mask_without_x, y):
                    continue
                
                # Calculate G_y: number of 3-APs gained by adding y to A_mask \ {x}
                G_y = 0
                for other_mask in element_to_3ap_others[y]:
                    if (A_mask_without_x & other_mask) == other_mask:
                        G_y += 1
                
                new_score = current_score - L_x + G_y
                if new_score > best_swap_score:
                    best_swap_score = new_score
                    best_swap = (x, y)
        
        if best_swap:
            x, y = best_swap
            A_mask = (A_mask & ~(1 << (x - 1))) | (1 << (y - 1))
            current_score = best_swap_score
            improved = True
            continue
        
        if not improved:
            break
            
    return A_mask

def polish_pass(A_mask, deadline):
    """
    A more aggressive pass that also considers removals if they improve the score?
    No, removals never improve the score in this objective (maximize 3-APs).
    Wait, removals reduce the set size, which reduces the number of potential 3-APs.
    So removals are never beneficial for maximizing 3-APs, UNLESS they allow for a swap
    that wasn't available before? No, swaps are already handled.
    Actually, the local search already handles additions and swaps.
    Removals are only useful if we are trying to minimize size, which we aren't.
    So polish_pass is essentially the same as local_search but maybe with different parameters or order.
    For now, I'll just use local_search_mask for the main optimization.
    """
    return local_search_mask(A_mask, deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s budget, 5s safety margin
    
    best_mask = None
    best_score = -1
    
    # Generate diverse initial candidates using bitmask greedy
    candidates = []
    
    # Candidate 0: Deterministic Greedy
    candidates.append(greedy_initial_mask(0))
    
    # Candidates 1-10: Randomized Greedies
    for i in range(1, 11):
        if time.time() > deadline - 20:
            break
        candidates.append(greedy_initial_mask(i))
    
    # Run local search on each candidate
    for idx, init_mask in enumerate(candidates):
        if time.time() > deadline - 5:
            break
        
        # Allocate time for local search
        # We want to give each candidate a fair chance, but prioritize earlier ones if time is short
        # A simple strategy: give each candidate 1/len(candidates) of the remaining time, with a cap
        remaining_time = deadline - time.time()
        num_candidates_left = len(candidates) - idx
        time_per_candidate = max(1.0, remaining_time / num_candidates_left)
        
        search_deadline = min(deadline - 2, time.time() + time_per_candidate)
        
        optimized = local_search_mask(init_mask, search_deadline)
        opt_score = count_3aps_mask(optimized)
        
        if opt_score > best_score:
            best_score = opt_score
            best_mask = optimized
    
    # If we have time left, do a final polish on the best mask found
    if best_mask is not None and time.time() < deadline - 2:
        final_deadline = deadline - 1
        best_mask = polish_pass(best_mask, final_deadline)
        final_score = count_3aps_mask(best_mask)
        if final_score > best_score:
            best_score = final_score
    
    if best_mask is None:
        best_mask = candidates[0] if candidates else greedy_initial_mask(0)
    
    # Convert to sorted list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
    
    return sorted(result)
