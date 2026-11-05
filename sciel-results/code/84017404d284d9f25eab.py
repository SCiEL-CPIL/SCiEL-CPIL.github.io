import time
import random

def _check_extension(seq, sums, x):
    """
    Check if adding x to seq is valid.
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

def _greedy_randomized(k=1, seed=None):
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
    
    for x in range(1, 257):
        new_tails = _check_extension(seq, sums, x)
        
        if new_tails is not None:
            candidates_buffer.append((x, new_tails))
            
            if len(candidates_buffer) >= k:
                idx = random.randint(0, len(candidates_buffer) - 1)
                x_val, tails = candidates_buffer[idx]
                
                seq.append(x_val)
                sums.update(tails)
                
                candidates_buffer = []
                
    # If there are remaining candidates, pick one?
    # Usually we want to maximize length, so if there's any valid candidate left, take the best?
    # Or just stop. The loop ends at 256.
    # If buffer is not empty, we can add one more element.
    if candidates_buffer:
        # Pick the largest? Smallest? Random?
        # Let's pick the largest to leave room? No, we are at the end.
        # Just pick one.
        idx = random.randint(0, len(candidates_buffer) - 1)
        x_val, tails = candidates_buffer[idx]
        seq.append(x_val)
        sums.update(tails)

    return seq

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
        for seed in range(50):
            if time.time() > deadline:
                break
                
            seq = _greedy_randomized(k=k, seed=seed)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
                
        if time.time() > deadline:
            break
            
    # 3. Random Skip Strategy
    # Skip valid numbers with probability p
    for p in [0.05, 0.1, 0.2, 0.3]:
        for seed in range(50):
            if time.time() > deadline:
                break
            random.seed(seed)
            seq = []
            sums = set()
            for x in range(1, 257):
                if random.random() < p:
                    continue
                    
                new_tails = _check_extension(seq, sums, x)
                if new_tails is not None:
                    seq.append(x)
                    sums.update(new_tails)
                    
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
                
        if time.time() > deadline:
            break
            
    return best_seq
