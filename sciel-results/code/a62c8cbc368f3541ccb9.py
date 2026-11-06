import time
import random

def _greedy_build(candidates):
    seq = []
    sums = set()
    last = 0
    
    for x in candidates:
        if x <= last:
            continue
            
        # Check if x can be added
        # New sums: x, x+seq[-1], x+seq[-1]+seq[-2], ...
        
        # Check x
        if x in sums:
            continue
            
        acc = 0
        valid = True
        for a in reversed(seq):
            acc += a
            if (x + acc) in sums:
                valid = False
                break
                
        if valid:
            seq.append(x)
            last = x
            sums.add(x)
            acc = 0
            for a in reversed(seq[:-1]):
                acc += a
                sums.add(x + acc)
                
    return seq

def construct():
    start_time = time.time()
    time_limit = 80.0
    
    best_seq = []
    best_len = 0
    
    # 1. Deterministic Greedy (Ascending)
    seq_asc = _greedy_build(range(1, 257))
    if len(seq_asc) > best_len:
        best_len = len(seq_asc)
        best_seq = seq_asc
        
    # 2. Randomized Greedy
    # Use a random generator for better distribution
    rng = random.Random()
    
    while time.time() - start_time < time_limit:
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        
        seq = _greedy_build(candidates)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            
    return best_seq
