import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors for each word
    neighbors = []
    for i, w in enumerate(words):
        neigh = set([w])
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                new_w = list(w)
                new_w[pos] = val
                neigh.add(tuple(new_w))
        neigh_indices = [word_to_idx[ng] for ng in neigh]
        neighbors.append(neigh_indices)
    
    # 3. Greedy Set Cover
    uncovered = set(range(N))
    code = []
    
    while uncovered:
        best_idx = -1
        best_count = 0
        # Find word that covers most uncovered points
        for j in range(N):
            count = 0
            for nb in neighbors[j]:
                if nb in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = j
        code.append(best_idx)
        # Remove covered
        for nb in neighbors[best_idx]:
            uncovered.discard(nb)
            
        if time.time() > deadline:
            break

    # 4. Local Search: Try to remove redundant words
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Re-initialize coverage counts for current code
        counts = np.zeros(N, dtype=int)
        for idx in code:
            for nb in neighbors[idx]:
                counts[nb] += 1
        
        # Try to remove words
        removals = []
        for i in range(len(code)):
            idx = code[i]
            # Check if removing idx makes any count 0
            redundant = True
            for nb in neighbors[idx]:
                if counts[nb] == 1:
                    redundant = False
                    break
            
            if redundant:
                removals.append(idx)
        
        if removals:
            changed = True
            for idx in removals:
                code.remove(idx)
                for nb in neighbors[idx]:
                    counts[nb] -= 1
    
    # 5. Swap-based local search to try to reduce size further
    # Try to replace a codeword with another that might allow removing others
    for _ in range(10):  # Limited iterations due to time
        if time.time() > deadline:
            break
            
        # Pick a random codeword to try replacing
        if not code:
            break
        idx_to_replace = random.choice(code)
        
        # Try replacing with a random word not in code
        replacement = random.randint(0, N - 1)
        if replacement in code:
            continue
            
        # Tentatively replace
        new_code = [i for i in code if i != idx_to_replace]
        new_code.append(replacement)
        
        # Check if new_code is valid
        if is_valid(words, new_code, N, neighbors):
            code = new_code
            # Try to remove redundant words again
            changed = True
            while changed and time.time() < deadline:
                changed = False
                counts = np.zeros(N, dtype=int)
                for idx in code:
                    for nb in neighbors[idx]:
                        counts[nb] += 1
                
                removals = []
                for i in range(len(code)):
                    idx = code[i]
                    redundant = True
                    for nb in neighbors[idx]:
                        if counts[nb] == 1:
                            redundant = False
                            break
                    if redundant:
                        removals.append(idx)
                
                if removals:
                    changed = True
                    for idx in removals:
                        code.remove(idx)
                        for nb in neighbors[idx]:
                            counts[nb] -= 1
    
    return [list(words[i]) for i in code]

def is_valid(words, code_indices, N, neighbors):
    if not code_indices:
        return N == 0
    covered = np.zeros(N, dtype=bool)
    for idx in code_indices:
        for nb in neighbors[idx]:
            covered[nb] = True
    return np.all(covered)
