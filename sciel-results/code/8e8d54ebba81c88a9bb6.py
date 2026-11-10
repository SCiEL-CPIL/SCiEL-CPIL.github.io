import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    # Only optimized for the specific task instance q=3, n=7, radius=1
    # But we should handle general inputs gracefully or at least not crash.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies q=3, n=7, radius=1.
    
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple construction for other parameters to avoid crashing
        # Direct sum of tetracode if n>=4
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # Extend to n
            code = []
            for h in tetracode:
                for tail in itertools.product(range(q), repeat=n-4):
                    code.append(h + list(tail))
            return code
        else:
            # Brute force for small n
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    total_words = 3**7
    # Pre-compute neighbors for all 2187 words
    # Index representation: word is a base-3 number of length 7
    # Or just use tuples. Tuples are hashable and fast enough.
    # Let's use integer indices 0..2186 for speed in counting arrays.
    
    # Map index to tuple
    idx_to_tuple = [None] * total_words
    for i in range(total_words):
        t = [0]*n
        val = i
        for j in range(n-1, -1, -1):
            t[j] = val % 3
            val //= 3
        idx_to_tuple[i] = tuple(t)
        
    # Pre-compute neighbors (indices) for each index
    # neighbors[i] = list of indices within distance 1
    neighbors = [None] * total_words
    for i in range(total_words):
        w = idx_to_tuple[i]
        neigh = [i]
        for pos in range(n):
            for new_val in range(3):
                if new_val == w[pos]:
                    continue
                # Create new word
                new_w = list(w)
                new_w[pos] = new_val
                # Convert to index
                idx = 0
                for k in range(n):
                    idx = idx * 3 + new_w[k]
                neigh.append(idx)
        neighbors[i] = neigh

    # Helper to compute counts
    def compute_counts(code_indices):
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code_indices:
            for w in neighbors[c]:
                counts[w] += 1
        return counts

    # Helper to check validity (all counts > 0)
    def is_valid(counts):
        return np.all(counts > 0)

    # Initialization 1: Tetracode Sum (243 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init1 = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=3):
            w = h + list(tail)
            idx = 0
            for k in range(7):
                idx = idx * 3 + w[k]
            init1.append(idx)
            
    # Initialization 2: Greedy Set Cover
    def greedy_init():
        uncovered = set(range(total_words))
        code = []
        while uncovered:
            best_c = -1
            best_covered = 0
            # Optimization: only check candidates that cover at least one uncovered point?
            # For now, check all.
            for c in range(total_words):
                covered = 0
                for w in neighbors[c]:
                    if w in uncovered:
                        covered += 1
                if covered > best_covered:
                    best_covered = covered
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            for w in neighbors[best_c]:
                uncovered.discard(w)
            if time.time() > deadline - 5:
                break
        return code

    init2 = greedy_init()
    
    # Pick the smaller init
    if len(init1) < len(init2):
        current_code = init1
    else:
        current_code = init2
        
    current_counts = compute_counts(current_code)
    best_code = current_code.copy()
    best_size = len(best_code)
    
    # Local Search
    # We will use a simple loop: Delete, then Perturb.
    
    while time.time() < deadline - 5:
        # 1. Deletion Phase
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow safe deletion
            for i in range(len(current_code) - 1, -1, -1):
                c = current_code[i]
                # Check if c can be removed
                # c is removable if no word w in neighbors[c] has count 1
                removable = True
                for w in neighbors[c]:
                    if current_counts[w] == 1:
                        removable = False
                        break
                if removable:
                    # Remove c
                    for w in neighbors[c]:
                        current_counts[w] -= 1
                    current_code.pop(i)
                    changed = True
                    break # Restart loop
                    
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code.copy()
            
        # 2. Perturbation Phase
        if len(current_code) == 0:
            break
            
        # Pick a random word to replace
        idx_to_change = random.randint(0, len(current_code) - 1)
        old_c = current_code[idx_to_change]
        
        # Pick a new random word
        new_c = random.randint(0, total_words - 1)
        
        if new_c == old_c:
            continue
            
        # Apply change
        # Remove old
        for w in neighbors[old_c]:
            current_counts[w] -= 1
        # Add new
        for w in neighbors[new_c]:
            current_counts[w] += 1
            
        current_code[idx_to_change] = new_c
        
        # Run deletion phase again
        changed = True
        while changed:
            changed = False
            for i in range(len(current_code) - 1, -1, -1):
                c = current_code[i]
                removable = True
                for w in neighbors[c]:
                    if current_counts[w] == 1:
                        removable = False
                        break
                if removable:
                    for w in neighbors[c]:
                        current_counts[w] -= 1
                    current_code.pop(i)
                    changed = True
                    break

        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code.copy()
        else:
            # Revert to best state to avoid drifting
            # Restore current_code and counts to best_code state
            # This is expensive to re-compute counts from scratch every time.
            # Optimization: Store best_counts too?
            # Or just accept the drift if it's not too bad?
            # Given the short time, reverting is safer to keep the search focused.
            # But re-computing counts is O(M*15). M~200. 3000 ops. Fast.
            current_code = best_code.copy()
            current_counts = compute_counts(current_code)

    # Convert best_code indices back to list of lists
    result = [list(idx_to_tuple[idx]) for idx in best_code]
    return result
