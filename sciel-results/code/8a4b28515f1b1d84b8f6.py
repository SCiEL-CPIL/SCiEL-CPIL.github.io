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
        
        # Iterate over candidates. To speed up, we can just pick any from uncovered first?
        # No, greedy picks the one covering most.
        # Optimization: Only consider codewords that are in uncovered?
        # Actually, a codeword must cover itself, so if it's not in uncovered, it's already covered.
        # But a covered codeword might cover other uncovered words? No, if it's covered, it's in the ball of some other codeword.
        # However, the ball of a covered word might extend to uncovered words.
        # So we must check all words? Or just those in uncovered?
        # If a word is covered, it is in the ball of an existing codeword.
        # Does picking a covered word as a NEW codeword help?
        # If I pick word W which is already covered by C, I add W to codewords.
        # The new coverage is ball(W) \ union(balls of existing).
        # Since W is in ball(C), ball(W) is not necessarily subset of ball(C).
        # So yes, we should consider all words.
        
        for idx in range(total_words):
            # If this word is already a codeword, skip? No, we haven't added it yet.
            # We can skip if we already have this word in codewords (shouldn't happen in greedy set cover if we remove covered).
            # But we can skip if the intersection with uncovered is small?
            # Let's just calculate the count.
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
            
            # Check if all words in ball(cw_idx) are covered by codewords other than i
            # We need to check if for every w in ball(cw_idx), there exists j != i such that w in ball(codewords[j])
            
            # Optimization: If we remove codeword i, are all words in ball(i) still covered?
            # We can check this by seeing if the union of balls of other codewords covers ball(i).
            # This is expensive.
            # Alternative: Just try to remove it and verify coverage of the whole space?
            # Verification is O(N * K * N) which is too slow.
            
            # Better: For each word in ball(i), check if it's covered by others.
            # If any word in ball(i) is ONLY covered by i, we cannot remove i.
            
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
                # Do not increment i, check same index again
            else:
                i += 1
                
    return codewords
# EVOLVE-BLOCK-END
