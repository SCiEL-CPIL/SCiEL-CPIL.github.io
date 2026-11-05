import time
import random

def _check_and_add(values, sums, x):
    """Check if x can be appended to values. Returns (ok, new_tails)."""
    if values and x <= values[-1]:
        return False, None
    tails = [x]
    total = x
    for a in reversed(values):
        total += a
        tails.append(total)
    if sums.intersection(tails):
        return False, None
    return True, tails

def _greedy_order(order, values_init=None, sums_init=None):
    """Run greedy with a given order of candidates."""
    if values_init is None:
        values = []
        sums = set()
    else:
        values = list(values_init)
        sums = set(sums_init)
    
    for x in order:
        ok, tails = _check_and_add(values, sums, x)
        if ok:
            values.append(x)
            sums.update(tails)
    return values, sums

def _try_local_improve(values, sums, max_iterations=5000):
    """Try to improve by removing an element and re-inserting elsewhere."""
    best_len = len(values)
    current_values = list(values)
    current_sums = set(sums)
    
    for _ in range(max_iterations):
        # Try removing a random element
        idx = random.randint(0, len(current_values) - 1)
        removed = current_values[idx]
        
        # Remove it and rebuild sums
        new_values = current_values[:idx] + current_values[idx+1:]
        # Rebuild sums for new_values
        new_sums = set()
        for i in range(len(new_values)):
            s = 0
            for j in range(i, len(new_values)):
                s += new_values[j]
                new_sums.add(s)
        
        # Try re-inserting removed at various positions
        best_pos = -1
        best_new_values = None
        best_new_sums = None
        for pos in range(len(new_values) + 1):
            candidate = new_values[:pos] + [removed] + new_values[pos:]
            # Check if valid
            cand_sums = set()
            valid = True
            for i in range(len(candidate)):
                s = 0
                for j in range(i, len(candidate)):
                    s += candidate[j]
                    if s in cand_sums:
                        valid = False
                        break
                    cand_sums.add(s)
                if not valid:
                    break
            if valid:
                # Check strictly increasing
                if all(candidate[i] < candidate[i+1] for i in range(len(candidate)-1)):
                    if len(candidate) > best_len or (best_new_values is None):
                        best_pos = pos
                        best_new_values = candidate
                        best_new_sums = cand_sums
        
        if best_new_values is not None and len(best_new_values) >= len(current_values):
            current_values = best_new_values
            current_sums = best_new_sums
            if len(current_values) > best_len:
                best_len = len(current_values)
    
    return current_values

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Phase 1: Natural order greedy
    values, sums = _greedy_order(range(1, 257))
    best_values = list(values)
    best_sums = set(sums)
    best_len = len(best_values)
    
    # Phase 2: Randomized greedy restarts
    rng = random.Random(42)
    candidates = list(range(1, 257))
    
    while time.time() < deadline:
        rng.shuffle(candidates)
        vals, sums = _greedy_order(candidates)
        if len(vals) > best_len:
            best_len = len(vals)
            best_values = vals
            best_sums = sums
        elif len(vals) == best_len:
            # Could try to merge or keep as alternative
            pass
    
    # Phase 3: Local improvement on best
    if time.time() < deadline - 5:
        improved = _try_local_improve(best_values, best_sums, max_iterations=2000)
        if len(improved) > best_len:
            best_values = improved
    
    return best_values
