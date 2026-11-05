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
                pass
                
    return seq

def _try_extend_from_prefix(prefix, prefix_sums, max_val=256, deadline=None, max_tries=15):
    """
    Try to extend a given prefix using greedy with limited backtracking.
    Returns the longest valid extension found.
    """
    if deadline is not None and time.time() > deadline:
        return prefix[:]
    
    best = prefix[:]
    best_sums = set(prefix_sums)
    
    def _dfs(seq, sums, start_val, depth):
        nonlocal best, best_sums
        if deadline is not None and time.time() > deadline:
            return
        if len(seq) > len(best):
            best = seq[:]
            best_sums = set(sums)
        
        tried = 0
        for x in range(start_val, max_val + 1):
            if tried >= max_tries:
                break
            if deadline is not None and time.time() > deadline:
                break
            new_tails = _check_extension(seq, sums, x)
            if new_tails is not None:
                tried += 1
                _dfs(seq + [x], sums.union(new_tails), x + 1, depth + 1)
    
    _dfs(prefix, prefix_sums, (prefix[-1] + 1) if prefix else 1, 0)
    return best

def _local_search_improve(seq, sums, max_val=256, deadline=None, max_iters=50):
    """
    Local search: try to remove an element and re-extend to get a longer sequence.
    """
    if deadline is not None and time.time() > deadline:
        return seq[:]
    
    best = seq[:]
    best_len = len(seq)
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if len(best) >= max_val:
            break
            
        # Try removing each element and re-extending
        for i in range(len(best)):
            if deadline is not None and time.time() > deadline:
                break
                
            # Remove element i
            candidate = best[:i] + best[i+1:]
            if not candidate:
                cand_sums = set()
            else:
                # Recompute sums for candidate
                cand_sums = set()
                for j in range(len(candidate)):
                    s = 0
                    for k in range(j, len(candidate)):
                        s += candidate[k]
                        cand_sums.add(s)
            
            # Try to extend from this shorter prefix
            extended = _try_extend_from_prefix(candidate, cand_sums, max_val, deadline, max_tries=10)
            
            if len(extended) > best_len:
                best = extended
                best_len = len(extended)
                break  # Restart the outer loop with the improved sequence
    
    return best

def _beam_search(seed=None, beam_width=10, max_val=256, deadline=None):
    """
    Beam search: keep top 'beam_width' partial sequences at each step.
    """
    if seed is not None:
        random.seed(seed)
        
    # Each beam is (seq, sums)
    beams = [([], set())]
    
    for step in range(max_val):
        if deadline is not None and time.time() > deadline:
            break
            
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
        
    # Return the best beam (longest)
    best = max(beams, key=lambda b: len(b[0]))
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    best_seq = []
    best_len = 0
    
    # Phase 1: Quick greedy initialization with various strategies
    # 1. Deterministic Greedy (k=1)
    seq_greedy = _greedy_randomized(k=1, seed=0)
    if len(seq_greedy) > best_len:
        best_len = len(seq_greedy)
        best_seq = seq_greedy[:]
        
    # 2. Randomized Greedy with various k and seeds
    for k in [2, 3, 5, 10, 20]:
        for seed in range(30):
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
        for seed in range(20):
            if time.time() > deadline:
                break
            seq = _random_skip_greedy(seed=seed, skip_prob=skip_prob)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
                
        if time.time() > deadline:
            break
            
    # Phase 2: Beam search for better initial solutions
    for bw in [5, 10, 20]:
        if time.time() > deadline:
            break
        seq = _beam_search(seed=0, beam_width=bw, deadline=deadline)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
    # Phase 3: Local search improvement on the best solution found
    if time.time() < deadline - 10:
        # Compute sums for best_seq
        best_sums = set()
        for i in range(len(best_seq)):
            s = 0
            for j in range(i, len(best_seq)):
                s += best_seq[j]
                best_sums.add(s)
        
        improved = _local_search_improve(best_seq, best_sums, deadline=deadline, max_iters=30)
        if len(improved) > best_len:
            best_len = len(improved)
            best_seq = improved[:]
            
    # Phase 4: Targeted backtracking from prefixes of best solution
    if time.time() < deadline - 5 and best_len > 0:
        # Try extending from various prefixes of the best solution
        for prefix_len in range(min(5, best_len), 0, -1):
            if time.time() > deadline - 2:
                break
            prefix = best_seq[:prefix_len]
            # Compute sums for prefix
            prefix_sums = set()
            for i in range(len(prefix)):
                s = 0
                for j in range(i, len(prefix)):
                    s += prefix[j]
                    prefix_sums.add(s)
            
            extended = _try_extend_from_prefix(prefix, prefix_sums, deadline=deadline - 2, max_tries=20)
            if len(extended) > best_len:
                best_len = len(extended)
                best_seq = extended[:]
                
    return best_seq
