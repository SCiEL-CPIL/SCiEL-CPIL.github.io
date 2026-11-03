import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140
    
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=int)
    N = len(all_words)
    
    # Map word to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    nbrs = []
    for i in range(N):
        w = tuple(all_words[i])
        neighbors = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                new_w_tuple = tuple(new_w)
                neighbors.append(word_to_idx[new_w_tuple])
        nbrs.append(neighbors)
        
    # Convert to numpy arrays for faster slicing
    nbrs_arr = [np.array(nbr, dtype=int) for nbr in nbrs]

    # Seed solution
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        seed_code_indices = [word_to_idx[tuple(w)] for w in seed_code_words]
    else:
        # Fallback for small n
        seed_code_indices = list(range(N))

    best_code = seed_code_indices
    best_len = len(best_code)

    def greedy_search(seed):
        random.seed(seed)
        uncovered = np.ones(N, dtype=bool)
        code = []
        
        while uncovered.any():
            best_score = -1
            best_c = -1
            
            for c in range(N):
                score = uncovered[nbrs_arr[c]].sum()
                if score > best_score:
                    best_score = score
                    best_c = c
                elif score == best_score and score > 0:
                    if random.random() < 0.5:
                        best_c = c
            
            if best_c == -1:
                break
            
            code.append(best_c)
            uncovered[nbrs_arr[best_c]] = False
            
        return code

    def remove_redundant(code):
        if not code:
            return code
        
        cover_counts = np.zeros(N, dtype=int)
        for c in code:
            cover_counts[nbrs_arr[c]] += 1
            
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                if np.min(cover_counts[nbrs_arr[c]]) >= 2:
                    cover_counts[nbrs_arr[c]] -= 1
                    changed = True
                else:
                    new_code.append(c)
            code = new_code
        return code

    restarts = 0
    # Try a few restarts
    while time.time() < deadline:
        restarts += 1
        code = greedy_search(restarts)
        code = remove_redundant(code)
        
        if len(code) < best_len:
            best_code = code
            best_len = len(code)
            
    return [list(all_words[c]) for c in best_code]
