import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors and coverers
    neighbors = [[] for _ in range(total_words)]
    coverers = [[] for _ in range(total_words)]
    
    for i, word in enumerate(all_words):
        sphere = set()
        if radius == 1:
            sphere.add(i)
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_word_list = list(word)
                    new_word_list[pos] = val
                    new_word = tuple(new_word_list)
                    j = word_to_idx[new_word]
                    sphere.add(j)
        else:
            # Generic radius
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for offsets in itertools.product(range(q), repeat=dist):
                        if all(word[p] == o for p, o in zip(positions, offsets)):
                            continue
                        new_word_list = list(word)
                        for pos, off in zip(positions, offsets):
                            new_word_list[pos] = off
                        new_word = tuple(new_word_list)
                        j = word_to_idx[new_word]
                        sphere.add(j)
                        
        for j in sphere:
            neighbors[i].append(j)
            coverers[j].append(i)
            
    best_solution = None
    best_size = total_words + 1
    
    start_time = time.time()
    max_time = 115
    
    while time.time() - start_time < max_time:
        # Global Greedy Construction
        covered = np.zeros(total_words, dtype=np.bool_)
        gain = np.array([len(neighbors[i]) for i in range(total_words)], dtype=np.int32)
        selected = []
        
        while not covered.all():
            # Find candidate with max gain
            if np.max(gain) == 0:
                break # Should not happen if covered.all() is false and graph is connected, but safety
            
            c_star = np.argmax(gain)
            
            # If gain is 0, we might be stuck or done. 
            # If covered.all() is False but max gain is 0, it means no single codeword covers any uncovered word? 
            # This implies uncovered words are isolated? No, every word is in its own sphere.
            # So gain should be > 0 for any codeword that covers an uncovered word.
            
            selected.append(c_star)
            
            # Update covered and gains
            for nb in neighbors[c_star]:
                if not covered[nb]:
                    covered[nb] = True
                    # Decrement gain for all coverers of nb
                    for c in coverers[nb]:
                        gain[c] -= 1
                        
        # Local Search: Redundancy Removal
        cov_count = np.zeros(total_words, dtype=np.int32)
        for c in selected:
            for nb in neighbors[c]:
                cov_count[nb] += 1
        
        changed = True
        while changed:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is redundant
                # If min coverage of its neighbors is > 1, c is redundant
                min_cov = min(cov_count[nb] for nb in neighbors[c])
                if min_cov > 1:
                    for nb in neighbors[c]:
                        cov_count[nb] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = list(selected)
            
    return [all_words[i] for i in best_solution]
