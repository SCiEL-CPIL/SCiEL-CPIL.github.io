"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    total_words = q ** n
    
    # Precompute all words as a 2D array: shape (total_words, n)
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute neighbor indices for each word (Hamming distance <= radius)
    # For radius=1, each word has 1 + n*(q-1) neighbors
    # Build a mapping from word to index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
    
    # Precompute neighbor lists
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
        neighbors.append(nbs)
    
    # Convert to numpy array for fast operations
    neighbors_arr = np.array(neighbors, dtype=np.int32)
    
    # Initialize coverage
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    codeword_set = set()
    
    # Seed with tetracode direct sum construction
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for tail in itertools.product(range(3), repeat=n - 4):
            for h in tetracode:
                word = h + list(tail)
                idx = word_to_idx[tuple(word)]
                if idx not in codeword_set:
                    codewords.append(idx)
                    codeword_set.add(idx)
                    covered[neighbors_arr[idx]] = True
    
    # If not fully covered, use greedy set-cover to add more codewords
    uncovered_count = np.sum(~covered)
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find the codeword that covers the most uncovered words
        # Vectorized: for each candidate, count uncovered neighbors
        # uncovered_mask = ~covered
        # For each candidate i, count = sum(~covered[neighbors_arr[i]])
        
        # Efficient approach: use matrix multiplication or direct indexing
        # Since total_words is small (729), we can do this directly
        best_center = -1
        best_count = -1
        
        for candidate in range(total_words):
            nbs = neighbors_arr[candidate]
            count = np.sum(~covered[nbs])
            if count > best_count:
                best_count = count
                best_center = candidate
        
        if best_center == -1 or best_count == 0:
            break
        
        codewords.append(best_center)
        codeword_set.add(best_center)
        covered[neighbors_arr[best_center]] = True
        uncovered_count = np.sum(~covered)
    
    # Post-processing: remove redundant codewords
    # Iterate until no more can be removed
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            
            c_idx = codewords[i]
            # Check if removing c_idx leaves any word uncovered
            # A word w in neighbors[c_idx] is still covered if it's in neighbors of some other codeword
            redundant = True
            for w_idx in neighbors_arr[c_idx]:
                # Check if w_idx is covered by any other codeword
                covered_by_other = False
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    # Check if w_idx is in neighbors of codewords[j]
                    # Use set membership for speed
                    if w_idx in neighbors_arr[codewords[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            
            if redundant:
                codewords.pop(i)
                codeword_set.remove(c_idx)
                changed = True
                break  # Restart as indices changed
    
    # Convert back to list of lists
    result = [list(all_words[c]) for c in codewords]
    
    return result
# EVOLVE-BLOCK-END
