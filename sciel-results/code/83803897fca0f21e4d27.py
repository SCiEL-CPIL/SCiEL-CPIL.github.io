"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _compute_tails(values, x):
    """Compute the new tail sums if we append x to values."""
    tails = [x]
    total = x
    for a in reversed(values):
        total += a
        tails.append(total)
    return tails

def _is_valid_extension(values, sums, x):
    """Check if appending x to values creates no duplicate block sums."""
    tails = _compute_tails(values, x)
    return not sums.intersection(tails)

def _greedy_extend(values, sums, candidates, deadline, mode='smallest'):
    """Extend values greedily using the given mode."""
    ext = list(values)
    ext_sums = set(sums)
    if mode == 'smallest':
        cand_iter = list(candidates)
    else:
        cand_iter = list(reversed(candidates))
    
    for x in cand_iter:
        if time.time() > deadline:
            break
        if _is_valid_extension(ext, ext_sums, x):
            tails = _compute_tails(ext, x)
            ext.append(x)
            ext_sums.update(tails)
    return ext, ext_sums

def _beam_search(values, sums, candidates, beam_width, deadline):
    """Beam search to extend the sequence."""
    if not candidates:
        return values, sums
    
    # Each beam entry: (last_val, values, sums, length)
    beam = [(values[-1] if values else 0, list(values), set(sums), len(values))]
    best = (list(values), set(sums))
    
    while beam and time.time() < deadline - 1:
        new_beam = []
        for last_val, vals, s, length in beam:
            # Try extending this beam entry
            for x in candidates:
                if x <= last_val:
                    continue
                if time.time() > deadline - 1:
                    break
                if _is_valid_extension(vals, s, x):
                    tails = _compute_tails(vals, x)
                    new_vals = vals + [x]
                    new_sums = s | set(tails)
                    new_beam.append((x, new_vals, new_sums, length + 1))
        
        if not new_beam:
            break
        
        # Sort by length descending, then by last_val ascending (more room)
        new_beam.sort(key=lambda e: (-e[3], e[0]))
        beam = new_beam[:beam_width]
        
        # Track best
        for entry in beam:
            if entry[3] > len(best[0]):
                best = (entry[1], entry[2])
    
    return best[0], best[1]

def _randomized_backtrack(values, sums, candidates, target_len, time_budget, rng):
    """Randomized backtracking to try to reach target_len."""
    def backtrack(vals, s, min_val, depth, budget):
        if time.time() > budget:
            return None
        if len(vals) == target_len:
            return list(vals)
        
        # Pruning: can we still reach target_len?
        remaining_needed = target_len - len(vals)
        available = [c for c in candidates if c > min_val]
        if len(available) < remaining_needed:
            return None
        
        # Shuffle candidates for randomization
        shuffled = list(available)
        rng.shuffle(shuffled)
        
        for x in shuffled:
            if time.time() > budget:
                return None
            if _is_valid_extension(vals, s, x):
                tails = _compute_tails(vals, x)
                new_sums = s | set(tails)
                result = backtrack(vals + [x], new_sums, x, depth + 1, budget)
                if result is not None:
                    return result
        return None
    
    return backtrack(list(values), set(sums), values[-1] if values else 0, len(values), time_budget)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    rng = random.Random(42)
    
    best_result = []
    
    # Strategy 1: Greedy smallest
    def greedy_smallest():
        values, sums = [], set()
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if _is_valid_extension(values, sums, x):
                tails = _compute_tails(values, x)
                values.append(x)
                sums.update(tails)
        return values
    
    # Strategy 2: Greedy largest
    def greedy_largest():
        values, sums = [], set()
        for x in range(256, 0, -1):
            if time.time() > deadline:
                break
            if _is_valid_extension(values, sums, x):
                tails = _compute_tails(values, x)
                values.append(x)
                sums.update(tails)
        return sorted(values)
    
    # Strategy 3: Beam search from empty
    def beam_from_empty():
        return _beam_search([], set(), list(range(1, 257)), beam_width=8, deadline=deadline)
    
    # Strategy 4: Greedy + beam refinement
    def greedy_then_beam():
        # First get a greedy solution
        values, sums = greedy_smallest()
        if not values:
            return values
        # Then try to extend with beam search
        candidates = list(range(values[-1] + 1, 257))
        if not candidates:
            return values
        extended, _ = _beam_search(values, sums, candidates, beam_width=10, deadline=deadline)
        return extended
    
    # Strategy 5: Randomized backtracking with multiple restarts
    def randomized_restart():
        nonlocal best_result
        current_best = list(best_result) if best_result else []
        
        # Get baseline from greedy
        baseline = greedy_smallest()
        if len(baseline) > len(current_best):
            current_best = baseline
        
        target_len = len(current_best) + 1
        max_targets = 15
        
        for _ in range(max_targets):
            if time.time() > deadline - 10:
                break
            
            # Try multiple randomized backtracks
            found = None
            num_attempts = 5
            for attempt in range(num_attempts):
                if time.time() > deadline - 5:
                    break
                budget = deadline - 3
                found = _randomized_backtrack(
                    current_best, 
                    _all_block_sums(current_best) if current_best else set(),
                    list(range(1, 257)),
                    target_len,
                    budget,
                    rng
                )
                if found:
                    break
            
            if found:
                current_best = found
                target_len = len(current_best) + 1
            else:
                break
        
        return current_best
    
    def _all_block_sums(seq):
        """Return set of all nonempty consecutive block sums."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums
    
    # Run strategies in order of expected value, keeping the best
    strategies = [greedy_smallest, greedy_largest, greedy_then_beam, beam_from_empty, randomized_restart]
    
    for strategy in strategies:
        if time.time() > deadline - 5:
            break
        try:
            result = strategy()
            if result and len(result) > len(best_result):
                best_result = result
        except Exception:
            pass
    
    # Final: try to extend best_result with a quick greedy pass
    if best_result and time.time() < deadline - 2:
        try:
            sums = _all_block_sums(best_result)
            candidates = list(range(best_result[-1] + 1, 257))
            extended, _ = _greedy_extend(best_result, sums, candidates, deadline, mode='smallest')
            if len(extended) > len(best_result):
                best_result = extended
        except Exception:
            pass
    
    return best_result if best_result else greedy_smallest()
# EVOLVE-BLOCK-END
