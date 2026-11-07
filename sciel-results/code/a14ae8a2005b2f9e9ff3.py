"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128

# Precompute valid pairs (i, j) where i < j and (i+1)+(j+1) <= 128
# i.e., i + j + 2 <= 128, so i + j <= 126
_VALID_PAIRS = []
for i in range(N):
    for j in range(i + 1, N):
        if i + j + 1 < N:  # bit index of (i+1)+(j+1) is i+j+1, must be < N
            _VALID_PAIRS.append((i, j))

def _is_valid_mask(mask):
    """
    Check if the set represented by the bitmask violates the condition.
    Bit i represents integer i+1 (1..128).
    Condition: No three distinct members a, b, c in S such that a+b, a+c, b+c are also in S.
    """
    for i, j in _VALID_PAIRS:
        if not ((mask >> i) & 1):
            continue
        if not ((mask >> j) & 1):
            continue
        # Check if a+b is in S (bit i+j+1)
        if not ((mask >> (i + j + 1)) & 1):
            continue
        # Find c (bit k) such that c in S, a+c in S (bit k+i+1), b+c in S (bit k+j+1)
        # k != i, k != j
        candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
        if candidates & ~((1 << i) | (1 << j)):
            return False
    return True

def _construct_baseline():
    """
    Construct the known strong baseline: {65..128} U {1,3,...,31, 64}.
    """
    mask = 0
    for v in range(65, 129):
        mask |= (1 << (v - 1))
    for v in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 64]:
        mask |= (1 << (v - 1))
    return mask

def _greedy_fill(mask, order=None, deadline=None):
    """
    Try to add elements from 'order' to mask if valid.
    """
    if order is None:
        order = list(range(1, N + 1))
    for v in order:
        if deadline is not None and time.time() > deadline:
            break
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
    2. Try 'remove one, add many' moves (time-limited greedy).
    3. Try random swaps.
    """
    current_mask = mask
    current_size = current_mask.bit_count()
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try adding single elements
        for v in range(1, N + 1):
            if time.time() > deadline:
                break
            bit = 1 << (v - 1)
            if current_mask & bit:
                continue
            trial = current_mask | bit
            if _is_valid_mask(trial):
                current_mask = trial
                current_size += 1
                improved = True
                break
        
        if improved:
            continue
            
        # Phase 2: Try 'remove one, add many' with time-limited greedy
        found_better = False
        best_candidate = current_mask
        best_candidate_size = current_size
        
        # Only try removing elements that might help - focus on smaller values
        # and values that are "blocking" many additions
        for v in range(1, N + 1):
            if time.time() > deadline:
                break
            bit = 1 << (v - 1)
            if not (current_mask & bit):
                continue
            
            reduced_mask = current_mask & ~bit
            
            # Time-limited greedy fill
            candidate = reduced_mask
            # Try adding in ascending order, but stop if time runs out
            for add_v in range(1, N + 1):
                if time.time() > deadline:
                    break
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
        else:
            # Phase 3: Random swap - remove one, try to add a different one
            swapped = False
            for _ in range(50):  # Try up to 50 random swaps
                if time.time() > deadline:
                    break
                # Pick a random element to remove
                elems = [i + 1 for i in range(N) if (current_mask >> i) & 1]
                if not elems:
                    break
                remove_v = random.choice(elems)
                remove_bit = 1 << (remove_v - 1)
                reduced = current_mask & ~remove_bit
                
                # Try to add a random missing element
                missing = [i + 1 for i in range(N) if not (current_mask >> i) & 1]
                if not missing:
                    break
                add_v = random.choice(missing)
                add_bit = 1 << (add_v - 1)
                trial = reduced | add_bit
                if _is_valid_mask(trial) and trial.bit_count() >= current_size:
                    if trial.bit_count() > current_size:
                        current_mask = trial
                        current_size += 1
                        swapped = True
                        break
                    elif random.random() < 0.3:  # Accept equal-size swaps sometimes for diversity
                        current_mask = trial
                        swapped = True
                        break
            
            if not swapped:
                break
                
    return current_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Start with strong baseline
    best_mask = _construct_baseline()
    best_size = best_mask.bit_count()
    
    # 2. Greedy fill
    if time.time() < deadline - 15:
        filled_mask = _greedy_fill(best_mask, list(range(1, N + 1)), deadline - 15)
        if filled_mask.bit_count() > best_size:
            best_mask = filled_mask
            best_size = best_mask.bit_count()
    
    # 3. Local search
    if time.time() < deadline - 5:
        searched_mask = _local_search(best_mask, deadline - 5)
        if searched_mask.bit_count() > best_size:
            best_mask = searched_mask
            best_size = best_mask.bit_count()
            
    # 4. Random restarts with perturbation
    if time.time() < deadline - 3:
        for restart in range(20):
            if time.time() > deadline - 3:
                break
            random.seed(restart * 7 + 13)
            perm = list(range(1, N + 1))
            random.shuffle(perm)
            
            # Perturbation: remove 3-7 random elements, then greedy + local search
            if best_size > 0:
                elems = [i + 1 for i in range(N) if (best_mask >> i) & 1]
                if len(elems) >= 3:
                    n_remove = min(random.randint(3, 7), len(elems))
                    to_remove = random.sample(elems, n_remove)
                    perturbed = best_mask
                    for r in to_remove:
                        perturbed &= ~(1 << (r - 1))
                    
                    # Greedy fill on perturbed
                    filled_p = _greedy_fill(perturbed, perm, deadline - 3)
                    
                    # Local search on filled_p
                    searched_p = _local_search(filled_p, deadline - 3)
                    
                    if searched_p.bit_count() > best_size:
                        best_mask = searched_p
                        best_size = best_mask.bit_count()

    # Final validity check
    if not _is_valid_mask(best_mask):
        best_mask = _construct_baseline()
        
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
