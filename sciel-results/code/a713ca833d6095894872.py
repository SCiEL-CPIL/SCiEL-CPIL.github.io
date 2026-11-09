"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

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

def _try_insert(values, x):
    """Try to insert x into values at any position, return new values or None."""
    n = len(values)
    for pos in range(n + 1):
        # Check strictly increasing
        if pos > 0 and values[pos-1] >= x:
            continue
        if pos < n and values[pos] <= x:
            continue
        new_values = values[:pos] + [x] + values[pos:]
        sums = _all_block_sums(new_values)
        if len(sums) == len(new_values) * (len(new_values) + 1) // 2:
            return new_values
    return None

def _local_search_improve(values, deadline):
    """Try to improve a sequence by inserting elements or swapping."""
    best = list(values)
    best_sums = _all_block_sums(best)
    
    # Try inserting elements
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for x in range(1, 257):
            if x in best:
                continue
            new_vals = _try_insert(best, x)
            if new_vals is not None:
                best = new_vals
                best_sums = _all_block_sums(best)
                improved = True
                break
        if not improved:
            break
    
    # Try local swaps to enable further insertions
    for _ in range(20):
        if time.time() > deadline:
            break
        improved = False
        n = len(best)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if best[j] - best[i] <= 1:
                    continue
                # Swap best[i] and best[j] if still increasing
                if i > 0 and best[i-1] >= best[j]:
                    continue
                if j < n-1 and best[j+1] <= best[i]:
                    continue
                if i + 1 <= j - 1:
                    # Check middle elements
                    if best[i+1] > best[j] or best[j-1] < best[i]:
                        continue
                new_best = best[:]
                new_best[i], new_best[j] = best[j], best[i]
                if not _validate_sequence(new_best):
                    continue
                # Try to insert after swap
                for x in range(1, 257):
                    if x in new_best:
                        continue
                    new_vals = _try_insert(new_best, x)
                    if new_vals is not None:
                        best = new_vals
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
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
    
    # Strategy 2: Greedy with lookahead
    def greedy_with_lookahead():
        values, sums = [], set()
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if _is_valid_extension(values, sums, x):
                tails = _compute_tails(values, x)
                values.append(x)
                sums.update(tails)
        return values
    
    # Strategy 3: Beam search with multiple partial sequences
    def beam_search():
        """
        Beam search maintaining top-K partial sequences.
        """
        beam_width = 15
        beam = [(0, 0, 0, [], set())]  # (neg_len, last_val, total, values, sums)
        best_result = []
        
        while beam and time.time() < deadline - 3:
            new_beam = []
            for neg_len, last_val, total, values, sums in beam:
                for x in range(last_val + 1, 257):
                    if time.time() > deadline - 3:
                        break
                    if _is_valid_extension(values, sums, x):
                        tails = _compute_tails(values, x)
                        new_values = values + [x]
                        new_sums = sums | set(tails)
                        new_total = total + x
                        new_neg_len = -(len(new_values))
                        new_last = x
                        new_beam.append((new_neg_len, new_last, new_total, new_values, new_sums))
                new_beam.append((neg_len, last_val, total, values, sums))
            
            if not new_beam:
                break
            
            new_beam.sort(key=lambda e: (e[0], e[1]))
            beam = new_beam[:beam_width]
            
            for entry in beam:
                if len(entry[3]) > len(best_result):
                    best_result = entry[3]
        
        return best_result
    
    # Strategy 4: Randomized restarts with local search
    def randomized_search():
        best_result = []
        rng = random.Random(42)
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Start with a random seed sequence
            values = []
            sums = set()
            used = set()
            
            # Pick a few initial values randomly
            for _ in range(min(3, 256)):
                candidates = [x for x in range(1, 257) if x not in used]
                if not candidates:
                    break
                x = rng.choice(candidates)
                if _is_valid_extension(values, sums, x):
                    tails = _compute_tails(values, x)
                    values.append(x)
                    sums.update(tails)
                    used.add(x)
            
            # Extend greedily
            for x in range(1, 257):
                if x in used:
                    continue
                if _is_valid_extension(values, sums, x):
                    tails = _compute_tails(values, x)
                    values.append(x)
                    sums.update(tails)
                    used.add(x)
            
            # Local search improvement
            values = _local_search_improve(values, deadline - 3)
            
            if len(values) > len(best_result):
                best_result = values
        
        return best_result
    
    # Strategy 5: Iterative deepening with pruning
    def iterative_deepening():
        best_result = []
        baseline = greedy_smallest()
        best_result = baseline
        target_len = len(baseline) + 1
        
        while target_len <= 18 and time.time() < deadline - 10:
            found = _find_sequence_of_length(target_len, deadline - 5)
            if found:
                best_result = found
                target_len += 1
            else:
                break
        
        return best_result
    
    def _find_sequence_of_length(target_len, time_budget):
        def backtrack(values, sums, min_val):
            if time.time() > time_budget:
                return None
            if len(values) == target_len:
                return values[:]
            
            if 256 - min_val + 1 < target_len - len(values):
                return None
            
            for x in range(min_val, 257):
                if time.time() > time_budget:
                    return None
                if _is_valid_extension(values, sums, x):
                    tails = _compute_tails(values, x)
                    new_sums = sums | set(tails)
                    result = backtrack(values + [x], new_sums, x + 1)
                    if result is not None:
                        return result
            return None
        
        return backtrack([], set(), 1)
    
    # Run all strategies, keep the best
    best_result = []
    
    strategies = [greedy_smallest, beam_search, randomized_search, iterative_deepening]
    
    for strategy in strategies:
        if time.time() > deadline - 5:
            break
        try:
            result = strategy()
            if result and len(result) > len(best_result):
                # Validate before accepting
                if _validate_sequence(result):
                    best_result = result
        except Exception:
            pass
    
    # Final local search improvement on best result
    if best_result and time.time() < deadline - 2:
        improved = _local_search_improve(best_result, deadline - 1)
        if _validate_sequence(improved) and len(improved) >= len(best_result):
            best_result = improved
    
    return best_result if best_result else greedy_smallest()
# EVOLVE-BLOCK-END
