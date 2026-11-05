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
        covered = np.zeros(total_words, dtype=np.bool_)
        selected = []
        
        while not covered.all():
            # Pick a random uncovered word
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            target_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            
            # Get candidates that cover target_idx
            candidates = coverers[target_idx]
            
            best_candidate = -1
            best_score = -1
            best_ties = []
            
            for c in candidates:
                # Score: number of new words covered by c
                score = 0
                for nb in neighbors[c]:
                    if not covered[nb]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_ties = [c]
                elif score == best_score:
                    best_ties.append(c)
                    
            best_candidate = random.choice(best_ties)
            selected.append(best_candidate)
            
            # Update covered
            for nb in neighbors[best_candidate]:
                covered[nb] = True
                
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
