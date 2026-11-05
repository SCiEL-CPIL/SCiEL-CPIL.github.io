import time
import random
import bisect

def _is_valid_candidate(candidate):
    sums = 0
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            if (sums >> s) & 1:
                return False
            sums |= (1 << s)
    return True

def _greedy_build(order):
    selected = []
    for x in order:
        candidate = selected + [x]
        candidate.sort()
        if _is_valid_candidate(candidate):
            selected = candidate
    return selected

def _try_insert(selected, x):
    """Try to insert x into sorted selected and check validity."""
    # Find insertion point
    pos = bisect.bisect_left(selected, x)
    if pos < len(selected) and selected[pos] == x:
        return None # Duplicate
    new_candidate = selected[:pos] + [x] + selected[pos:]
    if _is_valid_candidate(new_candidate):
        return new_candidate
    return None

def _try_swap(selected, u, v):
    """Try to swap u in selected with v (not in selected)."""
    if u not in selected or v in selected:
        return None
    new_candidate = [x for x in selected if x != u]
    new_candidate.append(v)
    new_candidate.sort()
    if _is_valid_candidate(new_candidate):
        return new_candidate
    return None

def _local_search(selected, available, deadline):
    """Try to improve the sequence by inserting or swapping."""
    current = selected[:]
    available_set = set(available)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to insert each available number
        avail_list = list(available_set)
        random.shuffle(avail_list)
        for x in avail_list:
            if x in current:
                continue
            new_seq = _try_insert(current, x)
            if new_seq:
                current = new_seq
                available_set.remove(x)
                improved = True
                break
        
        # Try swaps if no insertion was found
        if not improved and current and available_set:
            for _ in range(20):
                u = random.choice(current)
                v = random.choice(list(available_set))
                new_seq = _try_swap(current, u, v)
                if new_seq:
                    current = new_seq
                    available_set.remove(v)
                    available_set.add(u)
                    improved = True
                    break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_seq = []
    
    # Baselines
    seq = _greedy_build(list(range(1, 257)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    seq = _greedy_build(list(range(256, 0, -1)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    # Random Restarts with Local Search
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Generate a random order, biased towards larger numbers sometimes
        # Strategy 1: Pure random
        order = list(range(1, 257))
        rng.shuffle(order)
        
        # Strategy 2: Sorted descending (often better for sums)
        # We already did this, but maybe mix?
        # Let's stick to random shuffle for diversity
        
        seq = _greedy_build(order)
        
        # Local Search Phase
        if len(seq) > len(best_seq):
            best_seq = seq
            
        # Try to improve the current sequence (or best_seq?)
        # It's better to improve the current sequence and then compare
        available = set(range(1, 257)) - set(seq)
        improved_seq = _local_search(seq, available, deadline)
        
        if len(improved_seq) > len(best_seq):
            best_seq = improved_seq
            
        # Early termination if we found a very long sequence?
        # Max possible length is 256, but constraints are tight.
        
    return best_seq
