"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Generate all words in the space
    total_words = q ** n
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute balls for each word as sets of tuples for fast intersection
    # For efficiency, we map tuple -> index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    balls = []
    for idx in range(total_words):
        word = all_words[idx]
        ball = set([idx])
        for pos in range(n):
            for new_val in range(q):
                if new_val != word[pos]:
                    modified = word.copy()
                    modified[pos] = new_val
                    ball.add(word_to_idx[tuple(modified)])
        balls.append(ball)
    
    # Greedy Set Cover
    uncovered = set(range(total_words))
    codewords = []
    
    while uncovered:
        if time.time() > deadline - 10:
            break
            
        best_idx = None
        max_cover = -1
        
        for idx in range(total_words):
            count = len(balls[idx] & uncovered)
            if count > max_cover:
                max_cover = count
                best_idx = idx
                
        if best_idx is None:
            break
            
        codewords.append(all_words[best_idx])
        uncovered -= balls[best_idx]
        
    # Local Search: Remove redundant codewords
    # A codeword is redundant if all words in its ball are covered by other codewords.
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 5:
                break
            
            cw = tuple(codewords[i])
            cw_idx = word_to_idx[cw]
            
            can_remove = True
            for w_idx in balls[cw_idx]:
                # Check if w_idx is covered by any codeword j != i
                is_covered_by_other = False
                for j, other_cw in enumerate(codewords):
                    if j == i:
                        continue
                    if w_idx in balls[word_to_idx[tuple(other_cw)]]:
                        is_covered_by_other = True
                        break
                if not is_covered_by_other:
                    can_remove = False
                    break
            
            if can_remove:
                codewords.pop(i)
                changed = True
            else:
                i += 1
                
    # Additional local search: try swapping codewords to reduce count
    # We can try to replace pairs of codewords with one, or find better individual replacements
    # First, let's try a more aggressive removal with random ordering
    if time.time() < deadline - 10:
        # Try removing codewords in random order multiple times
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(codewords):
                    continue
                if time.time() > deadline - 5:
                    break
                cw = tuple(codewords[i])
                cw_idx = word_to_idx[cw]
                
                can_remove = True
                for w_idx in balls[cw_idx]:
                    is_covered_by_other = False
                    for j, other_cw in enumerate(codewords):
                        if j == i:
                            continue
                        if w_idx in balls[word_to_idx[tuple(other_cw)]]:
                            is_covered_by_other = True
                            break
                    if not is_covered_by_other:
                        can_remove = False
                        break
                
                if can_remove:
                    codewords.pop(i)
                    
    return codewords
# EVOLVE-BLOCK-END
