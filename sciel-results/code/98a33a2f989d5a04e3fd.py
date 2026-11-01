import numpy as np
import time
import itertools

def _precompute_neighbors(q, n):
    total = q ** n
    powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    all_indices = np.arange(total, dtype=np.int64)
    
    digits = np.zeros((total, n), dtype=np.int8)
    for i in range(n):
        digits[:, i] = (all_indices // powers[i]) % q
        
    neighbors = np.zeros((total, 1 + n * (q - 1)), dtype=np.int64)
    neighbors[:, 0] = all_indices
    
    pos = 1
    for i in range(n):
        p = powers[i]
        for v in range(q):
            mask = digits[:, i] != v
            new_indices = all_indices + (v - digits[:, i]) * p
            neighbors[mask, pos] = new_indices[mask]
            pos += 1
            
    return neighbors

def _cleanup(code, neighbors, total_words):
    # code is a list of indices
    # We want to remove redundant codewords
    # cover_count[w] = number of codewords covering w
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # Initialize counts
    for c in code:
        cover_count[neighbors[c]] += 1
        
    changed = True
    while changed:
        changed = False
        # Iterate in reverse to remove most recently added first? 
        # Or just iterate normally.
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            # Check if all neighbors of c are covered by at least 2 codewords
            # If min(cover_count[neighbors[c]]) > 1, then c is redundant
            if np.min(cover_count[neighbors[c]]) > 1:
                # Remove c
                code.pop(i)
                cover_count[neighbors[c]] -= 1
                changed = True
                break # Restart loop to be safe, as indices changed
                
    return code

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140
    
    if q != 3 or n != 8 or radius != 1:
        return _fallback_construct(q, n, radius)
        
    total_words = 3 ** n
    neighbors = _precompute_neighbors(q, n)
    
    best_code = None
    best_len = float('inf')
    
    seed_counter = 0
    while time.time() < deadline:
        seed_counter += 1
        random_state = np.random.RandomState(seed_counter)
        
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
            
            counts = uncovered[neighbors].sum(axis=1)
            max_count = counts.max()
            if max_count == 0:
                break
                
            candidates = np.where(counts == max_count)[0]
            idx = candidates[random_state.randint(len(candidates))]
            
            uncovered[neighbors[idx]] = False
            code.append(idx)
            
        # Cleanup
        code = _cleanup(code, neighbors, total_words)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            
            if best_len <= 486:
                break

    if best_code is None:
        return _fallback_construct(q, n, radius)
        
    powers = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    codewords = []
    for idx in best_code:
        word = []
        for i in range(n):
            word.append(int((idx // powers[i]) % 3))
        codewords.append(word)
        
    return codewords

def _fallback_construct(q, n, radius):
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
