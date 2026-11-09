"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s kill (90s wall, but be conservative)

    # First, get a baseline greedy solution to have something valid
    def greedy_construction():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    baseline = greedy_construction()
    best = baseline
    best_len = len(baseline)

    # Now try to do better with randomized/backtracking approaches
    # Key insight: the greedy approach is suboptimal because it always picks the smallest
    # available number. Sometimes skipping a small number allows more numbers later.
    
    # Strategy 1: Randomized greedy with restarts
    # Strategy 2: Backtracking with pruning for longer sequences
    
    def try_randomized_greedy(seed_val, max_attempts=2000):
        """Try randomized greedy: at each step, consider candidate values and pick randomly."""
        rng = random.Random(seed_val)
        values, sums = [], set()
        for _ in range(max_attempts):
            # Get all valid candidates
            candidates = []
            for x in range(1, 257):
                if values and x <= values[-1]:
                    continue
                tails, total = [x], x
                for a in reversed(values):
                    total += a
                    tails.append(total)
                if not sums.intersection(tails):
                    candidates.append(x)
            if not candidates:
                break
            # Pick randomly, with slight bias toward smaller values
            if rng.random() < 0.7:
                # Pick from the first 30% of candidates
                k = max(1, len(candidates) // 3)
                x = rng.choice(candidates[:k])
            else:
                x = rng.choice(candidates)
            values.append(x)
            tails, total = [x], x
            for a in reversed(values[:-1]):
                total += a
                tails.append(total)
            sums.update(tails)
            if len(values) > best_len:
                return values[:]
        return values if len(values) > best_len else None

    # Strategy 2: Backtracking with limited depth
    def backtrack_search(current_values, current_sums, start_val, depth_limit=12, time_limit=5.0):
        """Backtracking search to extend the sequence."""
        if len(current_values) >= depth_limit:
            return current_values[:]
        
        found_longer = [None]
        
        def bt(vals, sums, sv):
            if time.time() > time_limit:
                return
            if len(vals) > best_len and (found_longer[0] is None or len(vals) > len(found_longer[0])):
                found_longer[0] = vals[:]
            
            for x in range(sv, 257):
                tails, total = [x], x
                for a in reversed(vals):
                    total += a
                    tails.append(total)
                if sums.intersection(tails):
                    continue
                new_sums = sums | set(tails)
                bt(vals + [x], new_sums, x + 1)
                if found_longer[0] and len(found_longer[0]) > len(vals) + 1:
                    return  # Found something better, prune
        
        bt(current_values, current_sums, start_val)
        return found_longer[0]

    # Try multiple randomized greedy attempts
    for attempt in range(50):
        if time.time() > deadline:
            break
        result = try_randomized_greedy(attempt * 137 + 42)
        if result and len(result) > best_len:
            best = result
            best_len = len(best)
    
    # Try backtracking from the baseline or best found so far
    if time.time() < deadline - 10:
        # Try to extend the best sequence found so far
        bt_result = backtrack_search(best, None, best[-1] + 1 if best else 1, 
                                      depth_limit=best_len + 5, time_limit=8.0)
        if bt_result and len(bt_result) > best_len:
            best = bt_result
            best_len = len(best)

    # Try a different approach: start from scratch with backtracking
    if time.time() < deadline - 15:
        def full_backtrack(max_len=14, time_limit=12.0):
            best_found = [None]
            start_time = time.time()
            
            def bt(vals, sums, sv):
                if time.time() > start_time + time_limit:
                    return
                if len(vals) > (best_found[0] is None and 0 or len(best_found[0])):
                    best_found[0] = vals[:]
                
                if len(vals) >= max_len:
                    return
                
                for x in range(sv, 257):
                    tails, total = [x], x
                    for a in reversed(vals):
                        total += a
                        tails.append(total)
                    if sums.intersection(tails):
                        continue
                    new_sums = sums | set(tails)
                    bt(vals + [x], new_sums, x + 1)
                    if time.time() > start_time + time_limit:
                        return
            
            bt([], set(), 1)
            return best_found[0]
        
        bt_full = full_backtrack(max_len=best_len + 4, time_limit=10.0)
        if bt_full and len(bt_full) > best_len:
            best = bt_full
            best_len = len(best)

    return best

# EVOLVE-BLOCK-END
