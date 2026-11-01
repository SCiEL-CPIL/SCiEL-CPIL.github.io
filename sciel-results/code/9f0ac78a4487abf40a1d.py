import itertools
import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Fallback for non-specific cases
    if not (q == 3 and n == 6 and radius == 1):
        return _fallback_greedy(q, n, radius, start_time, deadline)
        
    total_words = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = []
    for i in range(total_words):
        w = all_words[i]
        nbs = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w2 = w.copy()
                    w2[pos] = new_val
                    nbs.append(word_to_idx[tuple(w2)])
        neighbors.append(np.array(nbs, dtype=np.int32))
        
    target_size = 73
    best_code = None
    
    # Try to find a 73-word covering code
    for restart in range(20):
        if time.time() > deadline:
            break
            
        random.seed(restart) # Deterministic restarts
        
        # Initialize with 73 random words
        code = set(np.random.choice(total_words, size=target_size, replace=False).tolist())
        
        # Calculate initial coverage
        covered = np.zeros(total_words, dtype=bool)
        for idx in code:
            covered[neighbors[idx]] = True
            
        uncovered = np.sum(~covered)
        
        if uncovered == 0:
            best_code = list(code)
            break
            
        # Simulated Annealing
        T = 10.0
        cool_rate = 0.95
        
        while T > 0.1 and time.time() < deadline:
            T *= cool_rate
            
            # Pick a word to swap out
            out_idx = random.choice(list(code))
            
            # Pick a word to swap in
            while True:
                in_idx = np.random.randint(total_words)
                if in_idx not in code:
                    break
                    
            # Calculate new uncovered
            # Remove out_idx, add in_idx
            # We need to know which words become uncovered and which become covered.
            
            # Words covered by out_idx but not by any other word in code?
            # This is expensive to check for each swap.
            
            # Approximation: Just re-calculate uncovered for the whole set?
            # 73 words * 19 neighbors = 1387 ops. Very fast.
            
            new_code = set(code)
            new_code.remove(out_idx)
            new_code.add(in_idx)
            
            new_covered = np.zeros(total_words, dtype=bool)
            for idx in new_code:
                new_covered[neighbors[idx]] = True
            new_uncovered = np.sum(~new_covered)
            
            delta = new_uncovered - uncovered
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                code = new_code
                covered = new_covered
                uncovered = new_uncovered
                
                if uncovered == 0:
                    best_code = list(code)
                    break
        
        if best_code:
            break
            
    if best_code:
        return [list(all_words[i]) for i in best_code]
        
    return _fallback_greedy(q, n, radius, start_time, deadline)

def _fallback_greedy(q, n, radius, start_time, deadline):
    # Copy of seed logic
    total_words = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    neighbors = []
    for i in range(total_words):
        w = all_words[i]
        nbs = [i]
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w2 = w.copy()
                        w2[pos] = new_val
                        nbs.append(word_to_idx[tuple(w2)])
        neighbors.append(np.array(nbs, dtype=np.int32))
        
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    # Greedy
    while np.sum(~covered) > 0:
        if time.time() > deadline:
            break
        best_center = -1
        best_count = -1
        for candidate in range(total_words):
            nbs = neighbors[candidate]
            count = np.sum(~covered[nbs])
            if count > best_count:
                best_count = count
                best_center = candidate
        if best_center == -1:
            break
        codewords.append(best_center)
        covered[neighbors[best_center]] = True
        
    # Remove redundant
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            c_idx = codewords[i]
            redundant = True
            for w_idx in neighbors[c_idx]:
                covered_by_other = False
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    if w_idx in neighbors[codewords[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            if redundant:
                codewords.pop(i)
                changed = True
                break
                
    return [list(all_words[c]) for c in codewords]
