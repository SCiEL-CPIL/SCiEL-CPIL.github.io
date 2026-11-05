import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Generate the initial 729-word code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert to tuples for hashing
    code = [tuple(w) for w in initial_code]
    
    # Precompute neighbors function
    def neighbors(w):
        yield w
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (w[i] + d) % q
                yield tuple(new_w)
    
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for d in range(n):
            idx = idx * q + w[d]
        return idx
    
    # Build the count array
    space_size = q ** n
    count = [0] * space_size
    for w in code:
        for nb in neighbors(w):
            count[word_to_idx(nb)] += 1
    
    # Pruning function
    def prune(code, count):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                w = code[i]
                # Check if w is redundant
                redundant = True
                for nb in neighbors(w):
                    if count[word_to_idx(nb)] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove w
                    code.pop(i)
                    for nb in neighbors(w):
                        count[word_to_idx(nb)] -= 1
                    improved = True
                    break
        return code, count
    
    # Initial pruning
    code, count = prune(code, count)
    
    # Try random shuffling to escape local optima
    best_code = code
    best_count = count
    best_size = len(code)
    
    for _ in range(50):  # 50 restarts
        if time.time() > deadline - 10:
            break
        # Shuffle the code
        shuffled_code = code.copy()
        random.shuffle(shuffled_code)
        
        # Rebuild count array
        count2 = [0] * space_size
        for w in shuffled_code:
            for nb in neighbors(w):
                count2[word_to_idx(nb)] += 1
        
        # Prune
        pruned_code, pruned_count = prune(shuffled_code, count2)
        
        if len(pruned_code) < best_size:
            best_code = pruned_code
            best_count = pruned_count
            best_size = len(pruned_code)
    
    return [list(w) for w in best_code]
