"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _compute_tails(values, x):
    """Compute all new tail sums when appending x to values."""
    tails = [x]
    total = x
    for a in reversed(values):
        total += a
        tails.append(total)
    return tails

def _is_valid_extension(values, sums, x):
    """Check if appending x creates any duplicate block sums."""
    tails = _compute_tails(values, x)
    return not sums.intersection(tails)

def _all_block_sums(values):
    """Compute all nonempty consecutive block sums for a sequence."""
    sums = set()
    n = len(values)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += values[j]
            sums.add(s)
    return sums

def _validate_sequence(values):
    """Check if a sequence is strictly increasing and has distinct block sums."""
    if not values:
        return True
    for i in range(1, len(values)):
        if values[i] <= values[i-1]:
            return False
    if len(values) > 256:
        return False
    if any(v < 1 or v > 256 for v in values):
        return False
    sums = _all_block_sums(values)
    return len(sums) == len(values) * (len(values) + 1) // 2

def _branch_and_bound(values, sums, min_val, target_len, deadline, best_found):
    """
    Branch and bound backtracking to find a sequence of target_len.
    Returns the sequence if found, else None.
    """
    if time.time() > deadline:
        return None
    
    current_len = len(values)
    if current_len == target_len:
        return values[:]
    
    # Pruning: not enough values remaining
    remaining_needed = target_len - current_len
    if 256 - min_val + 1 < remaining_needed:
        return None
    
    # Try candidates from min_val to 256
    for x in range(min_val, 257):
        if time.time() > deadline:
            return None
        
        if _is_valid_extension(values, sums, x):
            tails = _compute_tails(values, x)
            new_sums = sums | set(tails)
            new_values = values + [x]
            
            result = _branch_and_bound(new_values, new_sums, x + 1, target_len, deadline, best_found)
            if result is not None:
                return result
    
    return None

def _find_max_length(deadline):
    """
    Use iterative deepening with branch and bound to find the maximum length.
    Returns the longest valid sequence found.
    """
    # First, get a baseline with greedy
    values, sums = [], set()
    for x in range(1, 257):
        if time.time() > deadline - 5:
            break
        if _is_valid_extension(values, sums, x):
            tails = _compute_tails(values, x)
            values.append(x)
            sums.update(tails)
    
    best_result = values[:]
    base_len = len(values)
    
    # Try to find longer sequences
    for target_len in range(base_len + 1, 20):
        if time.time() > deadline - 3:
            break
        
        # Use branch and bound to find a sequence of this length
        result = _branch_and_bound([], set(), 1, target_len, deadline - 2, best_result)
        if result is not None:
            best_result = result
        else:
            # Can't find a longer sequence, stop
            break
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    # Primary strategy: iterative deepening with branch and bound
    best_result = _find_max_length(deadline)
    
    # If we have time, try some randomized approaches to potentially find longer sequences
    if time.time() < deadline - 10:
        import random
        rng = random.Random(12345)
        
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Randomized greedy with different starting points
            values, sums = [], set()
            
            # Randomly skip some small values to explore different branches
            skip_set = set()
            for _ in range(rng.randint(0, 5)):
                skip_set.add(rng.randint(1, 50))
            
            for x in range(1, 257):
                if time.time() > deadline - 5:
                    break
                if x in skip_set:
                    continue
                if _is_valid_extension(values, sums, x):
                    tails = _compute_tails(values, x)
                    values.append(x)
                    sums.update(tails)
            
            if len(values) > len(best_result):
                # Try to extend with branch and bound
                base_len = len(values)
                for target_len in range(base_len + 1, base_len + 3):
                    if time.time() > deadline - 3:
                        break
                    result = _branch_and_bound(values, _all_block_sums(values), values[-1] + 1 if values else 1, target_len, deadline - 2, best_result)
                    if result is not None:
                        best_result = result
                        break
    
    # Validate final result
    if not _validate_sequence(best_result):
        # Fallback to simple greedy
        values, sums = [], set()
        for x in range(1, 257):
            if _is_valid_extension(values, sums, x):
                tails = _compute_tails(values, x)
                values.append(x)
                sums.update(tails)
        best_result = values
    
    return best_result if best_result else [1]
# EVOLVE-BLOCK-END
