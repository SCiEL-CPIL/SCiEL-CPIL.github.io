"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128

def _is_valid_mask(mask):
    """
    Check if the set represented by the bitmask violates the condition.
    Bit i represents integer i+1 (1..128).
    Condition: No three distinct members a, b, c in S such that a+b, a+c, b+c are also in S.
    """
    # Iterate over potential pairs (a, b) in S
    # If a+b is in S, check if there exists c in S (c != a, c != b) 
    # such that a+c in S and b+c in S.
    
    for i in range(N):
        if not (mask >> i) & 1:
            continue
        a = i + 1
        for j in range(i + 1, N):
            if not (mask >> j) & 1:
                continue
            b = j + 1
            
            # Check if a+b is in S
            # a+b = i+j+2. Bit index is (i+j+2)-1 = i+j+1.
            if i + j + 1 >= N:
                continue
            if not (mask >> (i + j + 1)) & 1:
                continue
            
            # If a+b in S, check if there exists c in S, c != a, c != b
            # such that a+c in S and b+c in S.
            # c is value k+1.
            # a+c = i+k+2 => Bit i+k+1
            # b+c = j+k+2 => Bit j+k+1
            # We need bit k in mask, mask>>(i+1), mask>>(j+1).
            # And k != i, k != j.
            
            # candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
            # This finds all k such that:
            # k is in S
            # k + i + 1 is in S (which is a+c)
            # k + j + 1 is in S (which is b+c)
            
            candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
            
            # We need to ensure c != a and c != b.
            # c = k+1.
            # a = i+1 => k != i
            # b = j+1 => k != j
            
            # Mask out bits i and j from candidates
            # candidates &= ~(1 << i)
            # candidates &= ~(1 << j)
            
            # If any candidate remains, it's a violation.
            if candidates & ~( (1 << i) | (1 << j) ):
                return False
    return True

def _construct_baseline():
    """
    Construct the known strong baseline: {65..128} U {1,3,...,31, 64}.
    """
    mask = 0
    # Add 65..128
    for v in range(65, 129):
        mask |= (1 << (v - 1))
    
    # Add specific small numbers
    for v in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 64]:
        mask |= (1 << (v - 1))
        
    return mask

def _greedy_fill(mask, order=None):
    """
    Try to add elements from 'order' to mask if valid.
    """
    if order is None:
        order = list(range(1, N + 1))
        
    for v in order:
        bit = 1 << (v - 1)
        if mask & bit:
            continue
        trial = mask | bit
        if _is_valid_mask(trial):
            mask = trial
    return mask

def _local_search(mask, deadline):
    """
    Local search to improve mask size.
    Strategy: 
    1. Try adding any single missing element.
    2. Try 'remove one, add many' moves.
    """
    current_mask = mask
    current_size = current_mask.bit_count()
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try adding single elements
        # Check all missing elements
        for v in range(1, N + 1):
            bit = 1 << (v - 1)
            if current_mask & bit:
                continue
            trial = current_mask | bit
            if _is_valid_mask(trial):
                current_mask = trial
                current_size += 1
                improved = True
                # Restart the loop to check for more additions
                break
        
        if improved:
            continue
            
        # Phase 2: Try 'remove one, add many'
        # For each element x in current_mask, remove it, then greedily add back
        # all missing elements (including x potentially) to see if we get a larger set.
        
        # Optimization: Only try removing elements that are "small" or "central"
        # or just iterate all. N=128 is small.
        
        found_better = False
        best_candidate = current_mask
        best_candidate_size = current_size
        
        # Iterate through elements in the set
        # We can iterate through all 128 positions and check if set
        for v in range(1, N + 1):
            bit = 1 << (v - 1)
            if not (current_mask & bit):
                continue
            
            # Remove v
            reduced_mask = current_mask & ~bit
            reduced_size = reduced_size = reduced_mask.bit_count() # which is current_size - 1
            
            # Try to add all missing elements greedily
            # To be fast, we can just try adding in ascending order
            candidate = reduced_mask
            for add_v in range(1, N + 1):
                add_bit = 1 << (add_v - 1)
                if candidate & add_bit:
                    continue
                trial = candidate | add_bit
                if _is_valid_mask(trial):
                    candidate = trial
            
            if candidate.bit_count() > best_candidate_size:
                best_candidate = candidate
                best_candidate_size = candidate.bit_count()
                found_better = True
                
        if found_better:
            current_mask = best_candidate
            current_size = best_candidate_size
            # Loop back to try adding more singles
        else:
            # No improvement found with this strategy.
            # Try random swaps or perturbations?
            # Or just break if we've stuck for a while?
            # For now, break to avoid infinite loop on stuck state.
            # We can add a random restart mechanism in construct().
            break
            
    return current_mask

def construct():
    start_time = time.time()
    # Leave 10 seconds margin for safety
    deadline = start_time + 80.0
    
    # 1. Start with strong baseline
    best_mask = _construct_baseline()
    best_size = best_mask.bit_count()
    
    # 2. Greedy fill
    if time.time() < deadline - 10:
        filled_mask = _greedy_fill(best_mask, list(range(1, N + 1)))
        if filled_mask.bit_count() > best_size:
            best_mask = filled_mask
            best_size = best_mask.bit_count()
    
    # 3. Local search
    if time.time() < deadline - 5:
        searched_mask = _local_search(best_mask, deadline - 5)
        if searched_mask.bit_count() > best_size:
            best_mask = searched_mask
            best_size = best_mask.bit_count()
            
    # 4. Random restarts if time remains
    # Try different initial orders for greedy
    if time.time() < deadline - 2:
        for restart in range(10):
            if time.time() > deadline - 2:
                break
            random.seed(restart)
            perm = list(range(1, N + 1))
            random.shuffle(perm)
            
            # Start from empty or small base?
            # Starting from empty might be slow to reach high sizes.
            # Better to start from a random subset of the current best or baseline.
            # Let's just try greedy from scratch with random order.
            # It might find a different local optimum.
            
            # Actually, greedy from scratch often gets stuck at ~60-70.
            # The baseline is 81.
            # So we should perturb the best found so far.
            
            # Perturbation: remove 5 random elements, then local search.
            if best_size > 0:
                elems = [i + 1 for i in range(N) if (best_mask >> i) & 1]
                if len(elems) >= 5:
                    to_remove = random.sample(elems, 5)
                    perturbed = best_mask
                    for r in to_remove:
                        perturbed &= ~(1 << (r - 1))
                    
                    # Greedy fill on perturbed
                    filled_p = _greedy_fill(perturbed, perm)
                    
                    # Local search on filled_p
                    searched_p = _local_search(filled_p, deadline - 2)
                    
                    if searched_p.bit_count() > best_size:
                        best_mask = searched_p
                        best_size = best_mask.bit_count()

    # Final validity check
    if not _is_valid_mask(best_mask):
        # Fallback to baseline
        best_mask = _construct_baseline()
        
    # Convert mask to list of ints
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
