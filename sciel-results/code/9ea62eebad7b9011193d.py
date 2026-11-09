import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0 # Safety margin
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET
    
    # Pre-compute all masks with Hamming weight < MIN_DIST
    # These are the "forbidden" offsets
    bad_masks = []
    for m in range(1 << N):
        if m.bit_count() < MIN_DIST:
            bad_masks.append(m)
    
    best_words = []
    max_restarts = 100 # Cap on restarts, but time is the real limit
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        words = []
        blocked = set()
        
        # Generate candidate order
        if i == 0:
            candidates = range(1 << N)
        else:
            candidates = list(range(1 << N))
            random.shuffle(candidates)
            
        # Greedy construction with blocked set
        for c in candidates:
            # Early exit if time is running out
            if time.time() > deadline:
                break
                
            if c in blocked:
                continue
            
            # Accept c
            words.append(c)
            
            # Block all neighbors
            for m in bad_masks:
                blocked.add(c ^ m)
                
        # Update best
        if len(words) > len(best_words):
            best_words = words[:]
            
        # If we hit a high bound, we might stop, but let's rely on time
        # A_2(18,6) is known to be at least 2048? No, upper bound is ~1500?
        # Actually, for n=18, d=6, the bound is lower.
        # Let's just run until time out.
        
    return best_words
