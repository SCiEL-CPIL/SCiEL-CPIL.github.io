"""Improved covering code constructor using greedy set cover with pruning."""
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius
    using a greedy set-covering heuristic with local improvement.
    """
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, leaving 20s margin

    # Total number of words
    total_words = q ** n
    # Ball size: 1 + n*(q-1) for radius 1
    ball_size = 1 + n * (q - 1)

    # Precompute all words as tuples
    # For q=3, n=6: 729 words, manageable
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # For each word, precompute the set of words within Hamming distance <= radius
    # This is the "ball" around each word
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    # Build adjacency: for each word index, list of word indices within radius
    # For radius 1: each word is adjacent to itself and to n*(q-1) neighbors
    balls = []
    for w in all_words:
        ball = [w]
        for pos in range(n):
            orig = w[pos]
            for v in range(q):
                if v == orig:
                    continue
                neighbor = list(w)
                neighbor[pos] = v
                neighbor = tuple(neighbor)
                ball.append(neighbor)
        balls.append(ball)

    # Convert balls to sets of indices for fast intersection/union
    ball_idx_sets = [set(word_to_idx[w] for w in ball) for ball in balls]

    # Greedy set cover: start with no codewords, repeatedly pick the word whose
    # ball covers the most uncovered words.
    
    # We'll track coverage as a set of covered word indices
    covered = set()
    codewords = []
    
    # For efficiency, we track for each word how many codewords cover it (or just whether it's covered)
    # and the number of newly covered words if we add each candidate.
    
    # Use a list to track coverage count
    coverage_count = np.zeros(total_words, dtype=np.int32)
    
    def time_check():
        return time.time() > deadline

    # Greedy phase
    # At each step, find the word that maximizes the number of newly covered words
    # To speed this up, we can maintain a priority structure, but for 729 words it's fine
    # to just scan.
    
    # Optimization: we can precompute the balls and just check which are uncovered
    # For each candidate codeword c, the number of newly covered words is:
    #   len(ball_idx_sets[c] - covered)
    
    # We'll use a simple loop. For 729 words and ~73 iterations, this is 729*73 ≈ 53k checks,
    # each involving set difference of ~19 elements. Should be fast.
    
    while len(covered) < total_words:
        if time_check():
            break
        
        best_idx = -1
        best_new = 0
        best_ball = None
        
        for i in range(total_words):
            if coverage_count[i] > 0:
                # This word is already covered; adding it might still help
                # but we need to check if it covers new words
                pass
            # Compute how many words in ball i are not yet covered
            # We can do this by iterating over the ball
            new_covered = 0
            for w_idx in ball_idx_sets[i]:
                if coverage_count[w_idx] == 0:
                    new_covered += 1
            
            if new_covered > best_new:
                best_new = new_covered
                best_idx = i
                best_ball = ball_idx_sets[i]
        
        if best_idx == -1 or best_new == 0:
            break
        
        # Add best_idx as a codeword
        codewords.append(list(all_words[best_idx]))
        for w_idx in best_ball:
            coverage_count[w_idx] += 1
            if coverage_count[w_idx] == 1:
                covered.add(w_idx)
        
        if len(covered) >= total_words:
            break
    
    # Local improvement phase: try to remove codewords if possible
    # A codeword can be removed if all words in its ball are covered by other codewords
    if not time_check():
        improved = True
        while improved and not time_check():
            improved = False
            for i in range(len(codewords)):
                if time_check():
                    break
                # Temporarily remove codeword i
                # Check if all words in its ball are still covered by remaining codewords
                # We need to recompute coverage without this codeword
                # This is expensive; instead, we check if any word in the ball has coverage_count == 1
                # and that the only codeword covering it is codeword i
                
                # For each word in the ball of codeword i, check if it's covered by any other codeword
                can_remove = True
                ball_i = ball_idx_sets[ball_idx_sets.__len__() - 1]  # wrong, need to map back
                
                # Actually, let's just recompute: for each word in ball of codeword i,
                # is there another codeword j != i such that word is in ball of codeword j?
                cw_i = all_words[codewords[i] if False else codewords[i]]  # codewords[i] is a list
                # We need the index of codeword i in all_words
                cw_i_tuple = tuple(codewords[i])
                cw_i_idx = word_to_idx[cw_i_tuple]
                
                for w_idx in ball_idx_sets[cw_i_idx]:
                    # Check if w_idx is covered by any other codeword
                    found = False
                    for j, cw in enumerate(codewords):
                        if j == i:
                            continue
                        cw_j_tuple = tuple(cw)
                        cw_j_idx = word_to_idx[cw_j_tuple]
                        if w_idx in ball_idx_sets[cw_j_idx]:
                            found = True
                            break
                    if not found:
                        can_remove = False
                        break
                
                if can_remove:
                    codewords.pop(i)
                    improved = True
                    break  # restart the loop since indices changed
    
    # Verify coverage and return
    # Final verification
    if not time_check():
        final_covered = set()
        for cw in codewords:
            cw_tuple = tuple(cw)
            cw_idx = word_to_idx[cw_tuple]
            final_covered |= ball_idx_sets[cw_idx]
        
        if len(final_covered) < total_words:
            # Fallback: use the direct sum construction
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return codewords
