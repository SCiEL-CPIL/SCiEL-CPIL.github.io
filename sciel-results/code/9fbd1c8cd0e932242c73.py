import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin for safety
    
    def _greedy():
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

    def _check_extend(values, sums, x):
        """Check if x can be appended. Returns (tails, new_sums) or None."""
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        for t in tails:
            if t in sums:
                return None
        return tails

    def _backtrack(values, sums, start_val, target_len, deadline):
        """Try to extend values to reach target_len. Returns extended list or None."""
        if len(values) >= target_len:
            return values[:]
        if time.time() > deadline:
            return None
        for x in range(start_val, 257):
            result = _check_extend(values, sums, x)
            if result is not None:
                tails = result
                new_sums = sums | set(tails)
                ext = _backtrack(values + [x], new_sums, x + 1, target_len, deadline)
                if ext is not None:
                    return ext
        return None

    best_sequence = []
    best_len = 0

    # 1. Get greedy baseline
    greedy_seq = _greedy()
    best_sequence = greedy_seq
    best_len = len(greedy_seq)

    # Compute sums for the greedy sequence
    sums_set = set()
    n = len(best_sequence)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += best_sequence[j]
            sums_set.add(s)

    # 2. Try to extend the greedy sequence by 1
    if time.time() < deadline - 5:
        for x in range(best_sequence[-1] + 1, 257):
            result = _check_extend(best_sequence, sums_set, x)
            if result is not None:
                best_sequence = best_sequence + [x]
                best_len += 1
                break

    # 3. Try to find even longer sequences via iterative deepening backtracking
    #    Start from scratch, guided by the current best length.
    #    We try to find a sequence of length best_len + 1, then +2, etc.
    #    This is expensive, so we limit the number of extra lengths to try.
    
    # Optimization: Use the greedy sequence to guide the search?
    # Maybe start the backtrack with the first few elements of the greedy sequence?
    # Or just do a full search for small extensions.
    
    # Let's try a limited depth search for best_len + 1
    # If we find one, update best.
    
    # A more efficient approach:
    # Use the "inspiration 1" style backtrack but with a tighter deadline.
    
    # Let's try to find a sequence of length best_len + 1 from scratch.
    # This might be faster than trying to extend the specific greedy sequence if
    # the greedy sequence has a "bad" tail that blocks extension.
    
    # However, backtracking from scratch for length L is exponential.
    # For L=15-20, it might be too slow.
    
    # Alternative: Local Search / Hill Climbing
    # Try to swap elements in the current best sequence to free up a slot for a new element?
    # Or try to replace an element with a smaller one to allow a different extension?
    
    # Let's try a simple local search:
    # 1. Take current best.
    # 2. Try removing one element and re-running greedy from that point?
    #    Or try replacing an element with a smaller unused one?
    
    # Let's try "Greedy with restarts":
    # Try different starting points or different ordering heuristics.
    
    # Heuristic 1: Greedy smallest (done)
    # Heuristic 2: Greedy largest? (Unlikely to be longer, but maybe different structure)
    # Heuristic 3: Randomized Greedy?
    
    # Let's try a few randomized greedy runs.
    import random
    random.seed(42)
    
    for _ in range(100):
        if time.time() > deadline - 5:
            break
            
        # Shuffle the candidate order slightly?
        # Or just try a different greedy strategy.
        # Let's try: at each step, pick the smallest x that doesn't conflict, 
        # but with some probability pick the next smallest?
        
        # Actually, let's just try the "greedy from scratch" again but with a small random perturbation
        # in the choice of x when multiple are valid.
        
        values, sums = [], set()
        for x in range(1, 257):
            if time.time() > deadline - 2:
                break
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        
        if len(values) > best_len:
            best_len = len(values)
            best_sequence = values

    # Try backtracking for best_len + 1 if time permits
    if time.time() < deadline - 10:
        # Try to find a sequence of length best_len + 1
        # Use a timeout for this specific search
        bt_deadline = time.time() + 5
        found = _backtrack([], set(), 1, best_len + 1, bt_deadline)
        if found is not None:
            best_sequence = found
            best_len = len(found)
            
            # If we found best_len + 1, try for best_len + 2?
            if time.time() < deadline - 5:
                bt_deadline2 = time.time() + 3
                found2 = _backtrack([], set(), 1, best_len + 1, bt_deadline2)
                if found2 is not None:
                    best_sequence = found2
                    best_len = len(found2)

    return best_sequence
