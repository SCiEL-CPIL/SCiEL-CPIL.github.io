import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # The task specifies q=3, n=6, radius=1 for the main evaluation, 
    # but we should handle general cases if possible or at least not crash.
    # However, the specific optimization is for 3, 6, 1.
    # If parameters differ, fall back to seed logic or simple greedy.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            # Direct sum for others
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Small n, just return all words or simple set
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=6, radius=1
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Generate Seed (81 codewords)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # 2. Helper: Check validity
    def is_valid(code):
        covered = np.zeros(729, dtype=bool)
        for c in code:
            # Mark c
            idx = c[0]*81 + c[1]*27 + c[2]*9 + c[3]*3 + c[4]*1 + c[5]*1 # Wait, base 3 conversion
            # Better to use a helper or just linear index
            # idx = sum(c[i] * (3 ** (5-i)) for i in range(6))
            # Let's precompute neighbors for speed?
            # Or just compute on the fly.
            
            # Efficient index calculation
            idx = 0
            for i in range(6):
                idx = idx * 3 + c[i]
            covered[idx] = True
            
            # Mark neighbors
            for i in range(6):
                original_val = c[i]
                for new_val in range(3):
                    if new_val == original_val:
                        continue
                    # Create new codeword
                    nc = c.copy()
                    nc[i] = new_val
                    nidx = 0
                    for j in range(6):
                        nidx = nidx * 3 + nc[j]
                    covered[nidx] = True
        return np.all(covered)

    # Optimization: Precompute all 729 words and their neighbors?
    # Total words: 729.
    # We can represent a codeword by its index 0..728.
    # Precompute neighbors for all 729 indices.
    
    all_words = [list(map(int, np.base_repr(i, 3).zfill(6))) for i in range(729)]
    # Wait, base_repr returns string.
    # Let's do it manually or use product.
    all_words = [list(w) for w in itertools.product(range(3), repeat=6)]
    
    # Map word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each index
    neighbors = []
    for i in range(729):
        w = all_words[i]
        nbs = [i]
        for pos in range(6):
            orig = w[pos]
            for val in range(3):
                if val == orig:
                    continue
                nw = w.copy()
                nw[pos] = val
                nbs.append(word_to_idx[tuple(nw)])
        neighbors.append(nbs)
        
    def is_valid_indices(code_indices):
        covered = np.zeros(729, dtype=bool)
        for idx in code_indices:
            covered[neighbors[idx]] = True
        return np.all(covered)

    # Convert initial code to indices
    def to_indices(code):
        return [word_to_idx[tuple(c)] for c in code]
    
    def from_indices(indices):
        return [all_words[i] for i in indices]

    current_indices = to_indices(initial_code)
    best_indices = list(current_indices)
    best_len = len(best_indices)
    
    # Greedy Removal
    def greedy_remove(indices):
        changed = True
        while changed:
            changed = False
            for i in range(len(indices)):
                # Check if removing i keeps validity
                temp = indices[:i] + indices[i+1:]
                if is_valid_indices(temp):
                    indices = temp
                    changed = True
                    break # Restart loop
        return indices

    # Initial Optimization
    current_indices = greedy_remove(current_indices)
    best_indices = list(current_indices)
    best_len = len(best_indices)
    
    # Local Search with Perturbation
    random.seed(42)
    
    while time.time() < deadline:
        # Perturbation
        # Strategy: Remove a few, add a few random
        num_perturb = random.randint(1, 3)
        current_indices = list(best_indices)
        
        # Remove random elements
        for _ in range(num_perturb):
            if len(current_indices) > 1:
                idx_to_remove = random.randint(0, len(current_indices)-1)
                current_indices.pop(idx_to_remove)
        
        # Add random elements
        for _ in range(num_perturb):
            random_idx = random.randint(0, 728)
            # Avoid duplicates
            while random_idx in current_indices:
                random_idx = random.randint(0, 728)
            current_indices.append(random_idx)
        
        # Optimize
        current_indices = greedy_remove(current_indices)
        
        if len(current_indices) < best_len:
            best_len = len(current_indices)
            best_indices = list(current_indices)
            
            # Try to do more greedy removal immediately
            best_indices = greedy_remove(best_indices)
            best_len = len(best_indices)

    return from_indices(best_indices)
