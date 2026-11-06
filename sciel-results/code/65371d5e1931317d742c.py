"""B2[2] set construction via multi-restart randomized greedy + local polish."""
import random
import time

DEADLINE = 140  # safety margin before 150s kill

def _try_add(chosen, counts, x, max_count=2):
    """Try to add x to chosen; return (success, new_counts) if success."""
    # sums involving x: x + y for y in chosen with y <= x, plus 2*x
    # We need to check all sums s = x + y where y in chosen and y <= x (since a<=b)
    # Actually for B2: representations a+b with a<=b. When adding x:
    # New representations: (x, y) for y in chosen where y <= x, and (x, x)
    # But we must count ALL representations, so we track counts of each sum.
    new_sums = []
    for y in chosen:
        if y <= x:
            new_sums.append(x + y)
    new_sums.append(2 * x)
    
    trial = dict(counts)
    ok = True
    for s in new_sums:
        trial[s] = trial.get(s, 0) + 1
        if trial[s] > max_count:
            ok = False
            break
    if ok:
        return True, trial
    return False, None

def _greedy_with_order(order, max_count=2):
    """Greedy construction following the given order of candidates."""
    chosen = []
    counts = {}
    for x in order:
        ok, new_counts = _try_add(chosen, counts, x, max_count)
        if ok:
            chosen.append(x)
            counts = new_counts
    return chosen

def _local_polish(chosen, counts, max_count=2, time_limit=5.0):
    """Try to improve the set by swapping elements or adding missing ones."""
    chosen_set = set(chosen)
    best_size = len(chosen)
    
    # Try to add any element from 1..1000 not in chosen
    for x in range(1, 1001):
        if x in chosen_set:
            continue
        ok, _ = _try_add(chosen, counts, x, max_count)
        if ok:
            chosen.append(x)
            chosen_set.add(x)
            # Update counts
            for y in chosen:
                if y <= x:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
            best_size = len(chosen)
    
    return chosen, counts, best_size

def construct():
    start_time = time.time()
    
    # Phase 1: Try several randomized greedy orderings
    best_set = []
    best_counts = {}
    best_size = 0
    
    # First, try the natural order (1, 2, ..., 1000) as baseline
    order = list(range(1, 1001))
    result = _greedy_with_order(order)
    if len(result) > best_size:
        best_set = result
        best_size = len(result)
        # Rebuild counts
        best_counts = {}
        for i, a in enumerate(best_set):
            for b in best_set[i:]:
                s = a + b
                best_counts[s] = best_counts.get(s, 0) + 1
    
    # Try reverse order
    order = list(range(1000, 0, -1))
    result = _greedy_with_order(order)
    if len(result) > best_size:
        best_set = result
        best_size = len(result)
        best_counts = {}
        for i, a in enumerate(best_set):
            for b in best_set[i:]:
                s = a + b
                best_counts[s] = best_counts.get(s, 0) + 1
    
    # Randomized restarts with different seeds
    rng = random.Random(42)
    attempts = 0
    while time.time() - start_time < DEADLINE - 20:  # reserve 20s for polish
        order = list(range(1, 1001))
        rng.shuffle(order)
        result = _greedy_with_order(order)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
            best_counts = {}
            for i, a in enumerate(best_set):
                for b in best_set[i:]:
                    s = a + b
                    best_counts[s] = best_counts.get(s, 0) + 1
        attempts += 1
        # Check time more frequently
        if attempts % 10 == 0 and time.time() - start_time > DEADLINE - 20:
            break
    
    # Phase 2: Local polish - try to add any remaining elements
    if time.time() - start_time < DEADLINE - 5:
        best_set, best_counts, best_size = _local_polish(best_set, best_counts)
    
    # Phase 3: Another round of randomized greedy starting from best_set
    # Try to extend best_set with random order of remaining elements
    if time.time() - start_time < DEADLINE - 3:
        chosen_set = set(best_set)
        remaining = [x for x in range(1, 1001) if x not in chosen_set]
        if remaining:
            rng2 = random.Random(123)
            rng2.shuffle(remaining)
            # Try to add from remaining in shuffled order
            extended = list(best_set)
            ext_counts = dict(best_counts)
            for x in remaining:
                if time.time() - start_time > DEADLINE - 3:
                    break
                ok, new_counts = _try_add(extended, ext_counts, x)
                if ok:
                    extended.append(x)
                    ext_counts = new_counts
            if len(extended) > best_size:
                best_set = extended
                best_size = len(extended)
                best_counts = ext_counts
    
    return best_set

# EVOLVE-BLOCK-END
