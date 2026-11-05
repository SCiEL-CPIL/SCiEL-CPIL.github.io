"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _check_conflict(new_value, current_seq):
    """Check if adding new_value to current_seq creates duplicate consecutive block sums.
    Returns (is_valid, new_sums) where new_sums is the set of all consecutive block sums
    after adding new_value (if valid).
    """
    n = len(current_seq)
    # The new consecutive blocks ending at the new position:
    # [new_value], [current_seq[-1], new_value], [current_seq[-2], current_seq[-1], new_value], ...
    # Their sums are: new_value, new_value + current_seq[-1], new_value + current_seq[-1] + current_seq[-2], ...
    
    # To check efficiently, we need all existing consecutive block sums.
    # But recomputing all of them each time is O(n^2). Let's precompute or track them.
    
    # Actually, for a clean implementation, let's just track the set of all sums.
    pass

def _all_consecutive_sums(seq):
    """Compute set of all consecutive block sums for a sequence."""
    sums = set()
    n = len(seq)
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += seq[j]
            sums.add(total)
    return sums

def _greedy_build(candidate_order, deadline=None):
    """Build a valid sequence by greedily selecting from candidate_order."""
    values = []
    sums = set()
    
    for x in candidate_order:
        if deadline and time.time() > deadline:
            break
        
        # Compute all new consecutive block sums if we append x
        # New blocks are: [x], [values[-1], x], [values[-2], values[-1], x], ...
        new_sums = set()
        total = x
        new_sums.add(total)
        for a in reversed(values):
            total += a
            new_sums.add(total)
        
        # Check for conflicts with existing sums
        if sums.isdisjoint(new_sums):
            values.append(x)
            sums.update(new_sums)
    
    return values

def _greedy_forward():
    """Standard greedy: try candidates in ascending order."""
    return _greedy_build(range(1, 257))

def _greedy_backward():
    """Try candidates in descending order first."""
    return _greedy_build(range(256, 0, -1))

def _greedy_spaced():
    """Try candidates with gaps to spread out sums."""
    # Pick roughly every 3rd number, then fill in
    candidates = list(range(1, 257, 3))
    # Add some in between for more options
    candidates2 = list(range(2, 257, 3))
    candidates3 = list(range(3, 257, 3))
    
    best = _greedy_build(candidates)
    best_len = len(best)
    
    for c in [candidates2, candidates3]:
        result = _greedy_build(c)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    return best

def _beam_search(deadline):
    """Beam search over candidate selections."""
    import random
    
    # State: (sequence, sum_set)
    # We'll maintain a beam of partial sequences
    beam_width = 20
    beam = [([], set())]
    candidates = list(range(1, 257))
    
    while time.time() < deadline and len(beam) > 0:
        next_beam = []
        for seq, sums in beam:
            # Try each candidate
            for x in candidates:
                if x in seq:
                    continue
                if not (seq == [] or x > seq[-1]):  # Must be strictly increasing
                    continue
                
                # Compute new sums
                new_sums = set()
                total = x
                new_sums.add(total)
                for a in reversed(seq):
                    total += a
                    new_sums.add(total)
                
                if sums.isdisjoint(new_sums):
                    next_seq = seq + [x]
                    next_sums = sums.union(new_sums)
                    next_beam.append((next_seq, next_sums))
        
        if not next_beam:
            break
        
        # Keep only the top beam_width by length (and diversity)
        next_beam.sort(key=lambda t: (-len(t[0]), len(t[1])))
        beam = next_beam[:beam_width]
        
        if not beam:
            break
    
    # Return the longest sequence found
    if beam:
        return max(beam, key=lambda t: len(t[0]))[0]
    return []

def _local_search(start_seq, deadline):
    """Try to improve a sequence by swapping or removing elements."""
    import random
    
    best_seq = list(start_seq)
    best_len = len(best_seq)
    
    while time.time() < deadline:
        # Randomly try to modify the sequence
        seq = list(best_seq)
        random.shuffle(seq)  # This won't work well for strict ordering
        
        # Better: try removing one element and re-adding different ones
        if len(seq) < 2:
            break
        
        remove_idx = random.randint(0, len(seq) - 1)
        candidate_seq = seq[:remove_idx] + seq[remove_idx+1:]
        
        # Now try to extend candidate_seq with new values
        sums = _all_consecutive_sums(candidate_seq) if candidate_seq else set()
        improved = False
        
        for x in range(candidate_seq[-1] + 1 if candidate_seq else 1, 257):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(candidate_seq):
                total += a
                new_sums.add(total)
            
            if sums.isdisjoint(new_sums):
                candidate_seq.append(x)
                sums.update(new_sums)
                improved = True
                if len(candidate_seq) > best_len:
                    best_seq = list(candidate_seq)
                    best_len = len(candidate_seq)
        
        if not improved:
            # Try a different removal
            continue
    
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Start with greedy approaches
    best = _greedy_forward()
    best_len = len(best)
    
    result = _greedy_backward()
    if len(result) > best_len:
        best = result
        best_len = len(result)
    
    result = _greedy_spaced()
    if len(result) > best_len:
        best = result
        best_len = len(result)
    
    # Try beam search
    if time.time() < deadline:
        result = _beam_search(deadline)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Try local search improvements
    if time.time() < deadline:
        result = _local_search(best, deadline)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    return best
# EVOLVE-BLOCK-END
