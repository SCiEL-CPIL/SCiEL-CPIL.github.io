import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 120.0  # Leave 30s margin for safety
    
    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors for each word
    neighbors = []
    for i, w in enumerate(words):
        neigh = [i]  # Include itself
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors.append(neigh)
    
    # 3. Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    while np.any(uncovered):
        best_idx = -1
        best_count = 0
        # Find word that covers most uncovered points
        for j in range(N):
            count = np.sum([uncovered[nb] for nb in neighbors[j]])
            if count > best_count:
                best_count = count
                best_idx = j
        code.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered[nb] = False
            
        if time.time() > deadline - 5:
            break
    
    # 4. Local Search: Remove redundant words
    def compute_counts(code_list):
        counts = np.zeros(N, dtype=int)
        for idx in code_list:
            for nb in neighbors[idx]:
                counts[nb] += 1
        return counts
    
    counts = compute_counts(code)
    
    # Remove redundant words iteratively
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        i = 0
        while i < len(code):
            idx = code[i]
            # Check if idx is redundant: all its neighbors have count > 1
            redundant = all(counts[nb] > 1 for nb in neighbors[idx])
            if redundant:
                code.pop(i)
                for nb in neighbors[idx]:
                    counts[nb] -= 1
                changed = True
            else:
                i += 1
    
    # 5. Local Search: Try swaps to reduce size
    # For each word in code, try replacing it with a word not in code
    # This is expensive, so we do it sparingly
    if time.time() < deadline - 20:
        for _ in range(50):  # Limit number of swap attempts
            if time.time() > deadline - 15:
                break
            if not code:
                break
            # Pick a random word from code to try to remove
            remove_idx = random.randint(0, len(code) - 1)
            removed_word = code[remove_idx]
            code.remove(removed_word)
            for nb in neighbors[removed_word]:
                counts[nb] -= 1
            
            # Check if still valid
            if np.any(counts == 0):
                # Invalid, try to fix by adding a word that covers the uncovered
                uncovered_indices = np.where(counts == 0)[0]
                # Find a word that covers the most uncovered
                best_add = -1
                best_cov = 0
                for j in range(N):
                    if j in code:
                        continue
                    cov = sum(1 for nb in neighbors[j] if counts[nb] == 0)
                    if cov > best_cov:
                        best_cov = cov
                        best_add = j
                if best_add != -1:
                    code.append(best_add)
                    for nb in neighbors[best_add]:
                        counts[nb] += 1
                    # If still invalid after adding, revert
                    if np.any(counts == 0):
                        code.remove(best_add)
                        for nb in neighbors[best_add]:
                            counts[nb] -= 1
                        code.insert(remove_idx, removed_word)
                        for nb in neighbors[removed_word]:
                            counts[nb] += 1
                else:
                    # Can't fix, revert
                    code.insert(remove_idx, removed_word)
                    for nb in neighbors[removed_word]:
                        counts[nb] += 1
            else:
                # Valid! We successfully reduced size by 1
                pass
    
    return [list(words[i]) for i in code]
