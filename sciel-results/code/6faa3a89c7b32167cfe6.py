import numpy as np
import itertools
import time

def construct(q, n, radius):
    # Parameters for the specific task instance
    # q=3, n=8, radius=1
    total_words = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Precompute balls
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    # Compute balls for each word
    # balls[i] contains the indices of words within distance 1 of all_words[i]
    # Distance 1 means 0 or 1 position differs.
    # We can compute this by generating all neighbors.
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    
    # Helper to convert base-q number to vector? 
    # Or just use the vector representation.
    # Since we have all_words as vectors, we can compute neighbors.
    
    # For each word, generate its ball
    for i in range(total_words):
        w = all_words[i]
        neighbors = [w]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                # Find index of new_w
                # This is slow. Better to use a lookup table.
                pass
        
    # Faster approach for precomputation:
    # Map each word to its index.
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
        
    for i in range(total_words):
        w = all_words[i]
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors.append(idx)
        balls[i] = neighbors
        
    # 2. Greedy Set Cover
    is_covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    # Precompute newly_covered matrix for speed? 
    # We can just update is_covered.
    # scores = (~is_covered)[balls].sum(axis=1)
    
    while not np.all(is_covered):
        if time.time() > deadline - 5:
            break
            
        newly_covered = (~is_covered)[balls]
        scores = newly_covered.sum(axis=1)
        best_idx = np.argmax(scores)
        
        if scores[best_idx] == 0:
            break # Should not happen
            
        codewords.append(best_idx)
        is_covered[balls[best_idx]] = True
        
    # 3. Local Search
    # Initialize cover counts
    cover_counts = np.zeros(total_words, dtype=np.int32)
    for c in codewords:
        cover_counts[balls[c]] += 1
        
    # Precompute covers[w] for fast candidate finding in swaps
    covers = [[] for _ in range(total_words)]
    for i in range(total_words):
        for b_idx in balls[i]:
            covers[b_idx].append(i)
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant codewords
        i = 0
        while i < len(codewords):
            c = codewords[i]
            cover_counts[balls[c]] -= 1
            if not np.any(cover_counts == 0):
                codewords.pop(i)
                improved = True
            else:
                cover_counts[balls[c]] += 1
                i += 1
                
        # Phase 2: Swaps to enable removal
        if not improved:
            # Try to find a swap that allows a removal
            swap_found = False
            for i in range(len(codewords)):
                if swap_found:
                    break
                c = codewords[i]
                
                # Find words uniquely covered by c
                # A word w is uniquely covered by c if cover_counts[w] == 1 and c is in covers[w]
                # We can check this by:
                # unique_mask = (cover_counts == 1)
                # unique_in_ball = unique_mask[balls[c]]
                # unique_words = balls[c][unique_in_ball]
                
                unique_mask = (cover_counts == 1)
                unique_in_ball = unique_mask[balls[c]]
                unique_words = balls[c][unique_in_ball]
                
                if len(unique_words) == 0:
                    continue # Should have been removed in Phase 1
                
                # Find candidates b that cover all unique_words
                # Start with candidates covering the first unique word
                candidates = set(covers[unique_words[0]])
                for w in unique_words[1:]:
                    candidates &= set(covers[w])
                    
                # Exclude current codewords (we want to swap with an outside word? 
                # Actually, swapping with an inside word is just a permutation. 
                # But if we swap c with b (inside), the set is the same. 
                # We want to replace c with b (outside) to change the set structure.
                # However, if b is inside, it's useless.
                current_set = set(codewords)
                candidates -= current_set
                
                for b in candidates:
                    # Tentative swap
                    cover_counts[balls[c]] -= 1
                    cover_counts[balls[b]] += 1
                    
                    if not np.any(cover_counts == 0):
                        # Valid swap. Check if it enables a removal.
                        # Try removing any other codeword j
                        removal_found = False
                        for j in range(len(codewords)):
                            if j == i:
                                continue
                            c_j = codewords[j]
                            cover_counts[balls[c_j]] -= 1
                            if not np.any(cover_counts == 0):
                                # We can remove c_j
                                removal_found = True
                                cover_counts[balls[c_j]] += 1 # Revert check
                                break
                            else:
                                cover_counts[balls[c_j]] += 1
                                
                        if removal_found:
                            # Accept swap
                            codewords[i] = b
                            improved = True
                            swap_found = True
                            break # Restart main while loop
                        else:
                            # Revert swap
                            cover_counts[balls[b]] -= 1
                            cover_counts[balls[c]] += 1
                    else:
                        # Invalid swap
                        cover_counts[balls[b]] -= 1
                        cover_counts[balls[c]] += 1
                        
    return [list(all_words[c]) for c in codewords]
