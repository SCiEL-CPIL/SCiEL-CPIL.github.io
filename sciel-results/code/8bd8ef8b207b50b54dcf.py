import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET
    
    # Pre-compute all masks with Hamming weight < MIN_DIST
    bad_masks = []
    for m in range(1 << N):
        if m.bit_count() < MIN_DIST:
            bad_masks.append(m)
    
    best_words = []
    max_restarts = 5000
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        words = []
        blocked = bytearray(1 << N)
        
        # Generate candidate order
        if i == 0:
            candidates = list(range(1 << N))
            candidates.sort(key=lambda x: abs(x.bit_count() - N//2))
        else:
            candidates = list(range(1 << N))
            random.shuffle(candidates)
        
        # Greedy construction
        for c in candidates:
            if time.time() > deadline:
                break
            if blocked[c]:
                continue
            words.append(c)
            for m in bad_masks:
                blocked[c ^ m] = 1
                
        # Local improvement: try to improve by swapping
        if time.time() < deadline - 2.0 and len(words) > 10:
            for pos in range(min(len(words), 30)):
                if time.time() > deadline - 2.0:
                    break
                w_remove = words[pos]
                for m in bad_masks:
                    blocked[w_remove ^ m] = 0
                words_temp = words[:pos] + words[pos+1:]
                
                added = False
                for c in candidates[:2000]:
                    if blocked[c] or c in words_temp:
                        continue
                    valid = True
                    for w in words_temp:
                        if (c ^ w).bit_count() < MIN_DIST:
                            valid = False
                            break
                    if valid:
                        words_temp.append(c)
                        for m in bad_masks:
                            blocked[c ^ m] = 1
                        added = True
                        break
                
                if added and len(words_temp) > len(words):
                    words = words_temp
                elif not added:
                    words = words_temp + [w_remove]
                    for m in bad_masks:
                        blocked[w_remove ^ m] = 1
        
        if len(words) > len(best_words):
            best_words = words[:]
            
    return best_words
