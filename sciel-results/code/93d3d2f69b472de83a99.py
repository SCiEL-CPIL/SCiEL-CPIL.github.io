import time
import random

def _check_extension(seq, sums, x):
    """
    Check if adding x to the end of seq is valid.
    Returns the new sums to add if valid, or None if invalid.
    """
    if not seq:
        if x in sums:
            return None
        return [x]
    
    tails = []
    total = x
    tails.append(total)
    for a in reversed(seq):
        total += a
        tails.append(total)
        
    for s in tails:
        if s in sums:
            return None
            
    return tails

def _greedy_randomized(k=1, seed=None, max_val=256):
    """
    Greedy construction with randomization.
    If k=1, it's standard greedy (smallest valid).
    If k>1, it picks randomly from the last k valid candidates seen.
    """
    if seed is not None:
        random.seed(seed)
        
    seq = []
    sums = set()
    candidates_buffer = []
    
    for x in range(1, max_val + 1):
        new_tails = _check_extension(seq, sums, x)
        
        if new_tails is not None:
            candidates_buffer.append((x, new_tails))
            
            if len(candidates_buffer) >= k:
                idx = random.randint(0, len(candidates_buffer) - 1)
                x_val, tails = candidates_buffer[idx]
                
                seq.append(x_val)
                sums.update(tails)
                
                candidates_buffer = []
                
    if candidates_buffer:
        idx = random.randint(0, len(candidates_buffer) - 1)
        x_val, tails = candidates_buffer[idx]
        seq.append(x_val)
        sums.update(tails)

    return seq

def _random_skip_greedy(seed=None, max_val=256, skip_prob=0.1):
    """
    Randomized greedy that sometimes skips a valid candidate to explore different paths.
    """
    if seed is not None:
        random.seed(seed)
        
    seq = []
    sums = set()
    
    for x in range(1, max_val + 1):
        new_tails = _check_extension(seq, sums, x)
        
        if new_tails is not None:
            # With some probability, skip this candidate to try later ones
            if random.random() > skip_prob:
                seq.append(x)
                sums.update(new_tails)
            else:
                # Skip, but we need to be careful: if we skip, we might not find a valid one later
                # For now, just skip. If no more valid ones, we stop.
                pass
                
    return seq

def _beam_search(seed=None, beam_width=10, max_val=256):
    """
    Beam search: keep top 'beam_width' partial sequences at each step.
    """
    if seed is not None:
        random.seed(seed)
        
    # Each beam is (seq, sums)
    beams = [([], set())]
    
    for step in range(max_val):
        new_beams = []
        for seq, sums in beams:
            # Try to extend this beam
            for x in range(1, max_val + 1):
                if seq and x <= seq[-1]:
                    continue
                new_tails = _check_extension(seq, sums, x)
                if new_tails is not None:
                    new_seq = seq + [x]
                    new_sums = sums.union(new_tails)
                    new_beams.append((new_seq, new_sums))
        
        if not new_beams:
            break
            
        # Keep top 'beam_width' beams based on some heuristic (e.g., length, then sum)
        # Sort by length descending, then by last element ascending (to leave room)
        new_beams.sort(key=lambda b: (-len(b[0]), b[0][-1] if b[0] else 0))
        beams = new_beams[:beam_width]
        
        # Check if any beam is complete (can't be extended)
        # If all beams are complete, we can stop
        # But for now, we just keep going until no new beams are generated
        
    # Return the best beam (longest)
    best = max(beams, key=lambda b: len(b[0]))
    return best[0]

def _is_valid_candidate(candidate):
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build(order):
    selected = []
    sums = set()
    # We need to maintain the set of sums for the current 'selected' sequence.
    # However, since we are inserting into the middle (sorting), the set of sums changes non-locally.
    # It is safer to just recompute validity or use the O(N^2) check if N is small.
    # Given max length is likely < 20, O(N^2) is fine.
    for x in order:
        candidate = selected + [x]
        candidate.sort()
        if _is_valid_candidate(candidate):
            selected = candidate
    return selected

def _local_search(initial_seq, deadline, best_len, best_seq):
    """
    Perform local search on the sequence to maximize length.
    """
    current_seq = list(initial_seq)
    
    # Helper to get available numbers
    def get_available(seq):
        s = set(seq)
        return [x for x in range(1, 257) if x not in s]
    
    # Initial check
    if len(current_seq) > best_len:
        best_len = len(current_seq)
        best_seq = current_seq[:]
        
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.7: # Add
            available = get_available(current_seq)
            if not available:
                break
            x = random.choice(available)
            new_seq = current_seq + [x]
            new_seq.sort()
            if _is_valid_candidate(new_seq):
                current_seq = new_seq
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    
        elif r < 0.85: # Remove
            if current_seq:
                x = random.choice(current_seq)
                current_seq.remove(x)
                
        else: # Swap
            if current_seq:
                available = get_available(current_seq)
                if available:
                    x_rem = random.choice(current_seq)
                    x_add = random.choice(available)
                    new_seq = [x for x in current_seq if x != x_rem]
                    new_seq.append(x_add)
                    new_seq.sort()
                    if _is_valid_candidate(new_seq):
                        current_seq = new_seq
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = current_seq[:]
                            
    return best_len, best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_seq = []
    best_len = 0
    
    # 1. Deterministic Greedy (k=1)
    seq_greedy = _greedy_randomized(k=1, seed=0)
    if len(seq_greedy) > best_len:
        best_len = len(seq_greedy)
        best_seq = seq_greedy[:]
        
    # 2. Randomized Greedy with various k and seeds
    for k in [2, 3, 5, 10, 20]:
        for seed in range(20):
            if time.time() > deadline:
                break
            seq = _greedy_randomized(k=k, seed=seed)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
                
        if time.time() > deadline:
            break
            
    # 3. Random Skip Greedy
    for skip_prob in [0.05, 0.1, 0.2, 0.3]:
        for seed in range(10):
            if time.time() > deadline:
                break
            seq = _random_skip_greedy(seed=seed, skip_prob=skip_prob)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
                
        if time.time() > deadline:
            break
            
    # 4. Beam Search
    for bw in [5, 10, 20, 50]:
        if time.time() > deadline:
            break
        seq = _beam_search(seed=0, beam_width=bw)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # 5. Random Order Greedy (from inspiration 2)
    for seed in range(20):
        if time.time() > deadline:
            break
        random.seed(seed)
        order = list(range(1, 257))
        random.shuffle(order)
        seq = _greedy_build(order)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # 6. Local Search (from inspiration 1)
    # Run on the best sequence found so far
    if time.time() < deadline:
        best_len, best_seq = _local_search(best_seq, deadline, best_len, best_seq)
        
    # Also run local search on a few other good candidates if time permits
    # We need to keep track of top candidates. For simplicity, let's just run on the current best.
    # To be more robust, we could run on the result of a few different greedy runs.
    # But local search is expensive, so we stick to the best one or a couple of diverse ones.
    
    # Let's try running local search on the result of a standard greedy and a random greedy
    candidates_for_ls = [best_seq]
    
    # Add a couple of diverse seeds
    if time.time() < deadline - 5:
        seq_div1 = _greedy_randomized(k=5, seed=100)
        if len(seq_div1) > len(best_seq) - 2: # Only if competitive
            candidates_for_ls.append(seq_div1)
            
        seq_div2 = _random_skip_greedy(seed=50, skip_prob=0.2)
        if len(seq_div2) > len(best_seq) - 2:
            candidates_for_ls.append(seq_div2)
            
    for cand in candidates_for_ls:
        if time.time() < deadline - 1:
            # Run a short burst of local search
            sub_deadline = min(deadline - 1, time.time() + 5)
            # We need to be careful not to update global best if we start from a worse state
            # and don't improve it.
            # _local_search returns best_len, best_seq relative to the input best_len/best_seq
            # So we pass the global best_len/best_seq
            best_len, best_seq = _local_search(cand, sub_deadline, best_len, best_seq)
            
    return best_seq
