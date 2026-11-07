"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128
MAX_VAL = 128

def _is_valid_mask(mask):
    """
    Check if the set represented by the bitmask violates the condition.
    Bit i represents integer i+1 (1..128).
    Condition: No three distinct members a, b, c in S such that a+b, a+c, b+c are also in S.
    
    Optimized: For each pair (a,b) with a+b in S, check if there's a c != a,b
    such that a+c and b+c are both in S.
    """
    # Precompute: for each value v in S, which other values u are in S such that u+v is in S?
    # Actually, let's use a different approach:
    # For each a in S, for each b in S with b > a:
    #   If a+b in S, then for each c in S with c != a, c != b:
    #     Check if a+c in S and b+c in S
    # 
    # Faster: For each pair (a,b) where a+b is in S,
    #   The set of valid c's would be: {c in S : c != a, c != b, a+c in S, b+c in S}
    #   a+c in S means: for each c, check bit (a+c-1)
    #   b+c in S means: for each c, check bit (b+c-1)
    #
    # Even faster using bit tricks:
    #   Let S_a = {c : a+c in S} = {c : bit(c+a-1) is set}
    #   This is mask shifted left by a-1 positions (but we need to be careful about bounds)
    #   Actually, if we define T_a = {x : a+x in S}, then bit x of T_a is bit (x+a-1) of mask
    #   So T_a = mask >> (a-1) but only considering valid positions
    #
    # Let me use a cleaner approach with precomputed shift masks:
    
    # For each a in S:
    #   shift_a = a - 1
    #   The values c such that a+c is in S: these are c where bit(c + shift_a) is set in mask
    #   i.e., c is in S AND (mask >> shift_a) has bit c set
    #   So candidates_for_a = mask & (mask >> shift_a)
    #   But we need c >= 1, so bit 0 of (mask >> shift_a) corresponds to c=1, which means a+1 in S
    #   That's correct.
    #
    # For a pair (a,b):
    #   c must be in S, a+c in S, b+c in S, c != a, c != b
    #   candidates = mask & (mask >> (a-1)) & (mask >> (b-1))
    #   Remove c=a and c=b from candidates
    #   If any remain, violation
    
    # Optimization: only check pairs where a+b <= 128 (since a+b must be in S, and S subset of 1..128)
    
    for i in range(N):
        if not (mask >> i) & 1:
            continue
        a = i + 1
        # b must satisfy a+b <= 128, so b <= 128-a
        max_b = 128 - a
        for j in range(i + 1, min(max_b, N)):
            if not (mask >> j) & 1:
                continue
            b = j + 1
            
            # a+b = i+j+2, bit index = i+j+1
            ab_bit = i + j + 1
            if ab_bit >= N:
                continue
            if not (mask >> ab_bit) & 1:
                continue
            
            # Find c such that c in S, a+c in S, b+c in S
            # a+c in S: bit (c+a-1) = bit(c+i) in mask
            # b+c in S: bit (c+b-1) = bit(c+j) in mask
            # So: c in mask AND bit(c+i) in mask AND bit(c+j) in mask
            # candidates = mask & (mask >> i) & (mask >> j)
            
            candidates = mask & (mask >> i) & (mask >> j)
            
            # Remove c=a (bit i) and c=b (bit j)
            # c=a means c=i+1, so bit i
            # c=b means c=j+1, so bit j
            candidates &= ~((1 << i) | (1 << j))
            
            if candidates:
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
    1. Try adding any single missing element (multiple passes).
    2. Try 'remove one, add many' moves.
    3. Try 'remove two, add many' moves for escaping local minima.
    """
    current_mask = mask
    current_size = current_mask.bit_count()
    
    # First, do multiple passes of single-element additions
    while time.time() < deadline:
        improved = False
        for v in range(1, N + 1):
            bit = 1 << (v - 1)
            if current_mask & bit:
                continue
            trial = current_mask | bit
            if _is_valid_mask(trial):
                current_mask = trial
                current_size += 1
                improved = True
                break
        if not improved:
            break
    
    # Now try exchange moves
    while time.time() < deadline:
        found_better = False
        best_candidate = current_mask
        best_candidate_size = current_size
        
        # Get list of elements in current set
        elems = [i + 1 for i in range(N) if (current_mask >> i) & 1]
        
        # Phase 2: Remove one, add many
        for v in elems:
            if time.time() > deadline:
                break
            bit = 1 << (v - 1)
            reduced_mask = current_mask & ~bit
            
            # Greedy fill
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
        
        # Phase 3: Remove two, add many (sampled for larger sets)
        if not found_better and len(elems) >= 2:
            # Sample pairs if set is large
            if len(elems) > 20:
                num_samples = min(200, len(elems) * (len(elems) - 1) // 2)
                for _ in range(num_samples):
                    if time.time() > deadline:
                        break
                    a, b = random.sample(elems, 2)
                    reduced = current_mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
                    candidate = reduced
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
            else:
                # Try all pairs for small sets
                for i in range(len(elems)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(elems)):
                        a, b = elems[i], elems[j]
                        reduced = current_mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
                        candidate = reduced
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
            # After improvement, try single additions again
            while time.time() < deadline:
                improved = False
                for v in range(1, N + 1):
                    bit = 1 << (v - 1)
                    if current_mask & bit:
                        continue
                    trial = current_mask | bit
                    if _is_valid_mask(trial):
                        current_mask = trial
                        current_size += 1
                        improved = True
                        break
                if not improved:
                    break
        else:
            break
    
    return current_mask

def _branch_and_bound(mask, depth=0, deadline=0.0):
    """
    Heuristic search: try various exchange sizes (1, 2, 3) with greedy refill.
    """
    if time.time() > deadline:
        return mask
    
    current_size = mask.bit_count()
    best_mask = mask
    best_size = current_size
    
    elems = [i + 1 for i in range(N) if (mask >> i) & 1]
    
    # Try 1-exchange
    for v in elems:
        if time.time() > deadline:
            return best_mask
        reduced = mask & ~(1 << (v - 1))
        candidate = _greedy_fill(reduced)
        if candidate.bit_count() > best_size:
            best_size = candidate.bit_count()
            best_mask = candidate
    
    # Try 2-exchange
    if len(elems) >= 2:
        if len(elems) <= 25:
            for i in range(len(elems)):
                if time.time() > deadline:
                    return best_mask
                for j in range(i + 1, len(elems)):
                    a, b = elems[i], elems[j]
                    reduced = mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
                    candidate = _greedy_fill(reduced)
                    if candidate.bit_count() > best_size:
                        best_size = candidate.bit_count()
                        best_mask = candidate
        else:
            num_samples = min(300, len(elems) * (len(elems) - 1) // 2)
            for _ in range(num_samples):
                if time.time() > deadline:
                    return best_mask
                a, b = random.sample(elems, 2)
                reduced = mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
                candidate = _greedy_fill(reduced)
                if candidate.bit_count() > best_size:
                    best_size = candidate.bit_count()
                    best_mask = candidate
    
    # Try 3-exchange (sampled)
    if len(elems) >= 3:
        num_samples = min(200, len(elems) * (len(elems) - 1) * (len(elems) - 2) // 6)
        for _ in range(num_samples):
            if time.time() > deadline:
                return best_mask
            a, b, c = random.sample(elems, 3)
            reduced = mask & ~(1 << (a - 1)) & ~(1 << (b - 1)) & ~(1 << (c - 1))
            candidate = _greedy_fill(reduced)
            if candidate.bit_count() > best_size:
                best_size = candidate.bit_count()
                best_mask = candidate
    
    return best_mask

def construct():
    start_time = time.time()
    # Leave 10 seconds margin for safety (evaluator kills at 90s)
    deadline = start_time + 78.0
    
    # 1. Start with strong baseline
    best_mask = _construct_baseline()
    best_size = best_mask.bit_count()
    
    # 2. Greedy fill with multiple orderings
    if time.time() < deadline - 15:
        # Try ascending order
        filled_mask = _greedy_fill(best_mask, list(range(1, N + 1)))
        if filled_mask.bit_count() > best_size:
            best_mask = filled_mask
            best_size = best_mask.bit_count()
        
        # Try descending order
        if time.time() < deadline - 12:
            filled_desc = _greedy_fill(best_mask, list(range(N, 0, -1)))
            if filled_desc.bit_count() > best_size:
                best_mask = filled_desc
                best_size = best_mask.bit_count()
        
        # Try a few random orderings
        if time.time() < deadline - 10:
            for seed in range(5):
                if time.time() > deadline - 10:
                    break
                random.seed(seed)
                perm = list(range(1, N + 1))
                random.shuffle(perm)
                filled_r = _greedy_fill(best_mask, perm)
                if filled_r.bit_count() > best_size:
                    best_mask = filled_r
                    best_size = best_mask.bit_count()
    
    # 3. Local search
    if time.time() < deadline - 8:
        searched_mask = _local_search(best_mask, deadline - 8)
        if searched_mask.bit_count() > best_size:
            best_mask = searched_mask
            best_size = best_mask.bit_count()
            
    # 4. B&B / exchange search
    if time.time() < deadline - 4:
        bb_mask = _branch_and_bound(best_mask, deadline=deadline - 4)
        if bb_mask.bit_count() > best_size:
            best_mask = bb_mask
            best_size = best_mask.bit_count()
            
    # 5. Random restarts with perturbation
    if time.time() < deadline - 4:
        for restart in range(20):
            if time.time() > deadline - 4:
                break
            random.seed(1000 + restart)
            
            # Perturbation: remove random number of elements (2-6)
            elems = [i + 1 for i in range(N) if (best_mask >> i) & 1]
            if len(elems) < 3:
                break
            num_remove = random.randint(2, min(6, len(elems)))
            to_remove = random.sample(elems, num_remove)
            perturbed = best_mask
            for r in to_remove:
                perturbed &= ~(1 << (r - 1))
            
            # Greedy fill with random ordering
            perm = list(range(1, N + 1))
            random.shuffle(perm)
            filled_p = _greedy_fill(perturbed, perm)
            
            # Local search
            if time.time() < deadline - 3:
                searched_p = _local_search(filled_p, deadline - 3)
                if searched_p.bit_count() > best_size:
                    best_mask = searched_p
                    best_size = best_mask.bit_count()
            
            # B&B on perturbed result if time allows
            if time.time() < deadline - 2:
                bb_p = _branch_and_bound(best_mask, deadline=deadline - 2)
                if bb_p.bit_count() > best_size:
                    best_mask = bb_p
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
