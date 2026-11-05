import time

def _is_valid(seq):
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_forward_optimized():
    selected = []
    sums = set()
    for x in range(1, 257):
        # x is new last element
        # New sums are suffixes of selected + x
        valid = True
        new_sums = []
        current_sum = 0
        for a in reversed(selected):
            current_sum += a
            new_sum = current_sum + x
            if new_sum in sums:
                valid = False
                break
            new_sums.append(new_sum)
        
        if not valid:
            continue
        if x in sums:
            continue
        new_sums.append(x)
        
        selected.append(x)
        sums.update(new_sums)
    return selected

def _greedy_reverse_optimized():
    selected = []
    sums = set()
    for x in range(256, 0, -1):
        # x is new first element
        valid = True
        new_sums = []
        current_sum = 0
        for a in selected:
            current_sum += a
            new_sum = x + current_sum
            if new_sum in sums:
                valid = False
                break
            new_sums.append(new_sum)
        
        if not valid:
            continue
        if x in sums:
            continue
        new_sums.append(x)
        
        selected.insert(0, x)
        sums.update(new_sums)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_seq = []
    
    # 1. Optimized Greedies
    seq_fwd = _greedy_forward_optimized()
    if len(seq_fwd) > len(best_seq):
        best_seq = seq_fwd
        
    seq_rev = _greedy_reverse_optimized()
    if len(seq_rev) > len(best_seq):
        best_seq = seq_rev
        
    # 2. Randomized Restarts
    import random
    random.seed(123)
    
    # We can run more random restarts now because the first two were instant.
    # Let's run until deadline.
    # Each random restart is O(N * K^2).
    # Estimate: 256 * 400 = 100k ops.
    # Python does ~10M simple ops/sec? So ~0.01s per restart.
    # 50s / 0.01s = 5000 restarts.
    
    for i in range(10000):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Use a generic greedy that handles arbitrary insertion
        # But to be fast, we can just use the _is_valid check on the sorted list
        # We build the list by inserting x in sorted position and checking.
        
        selected = []
        # We can't easily optimize random order incrementally.
        # So we just do the naive check for each candidate in the shuffled list.
        
        for x in candidates:
            # Find insertion point
            idx = 0
            for j in range(len(selected)):
                if selected[j] < x:
                    idx = j + 1
                else:
                    break
            new_seq = selected[:idx] + [x] + selected[idx:]
            if _is_valid(new_seq):
                selected = new_seq
                
        if len(selected) > len(best_seq):
            best_seq = selected
            
    return best_seq
