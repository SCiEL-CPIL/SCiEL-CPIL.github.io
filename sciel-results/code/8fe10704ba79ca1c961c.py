import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    total_words = q ** n
    # Precompute all words
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # For each word, precompute its ball (words within distance <= radius)
    # For radius 1, q=3, n=6: ball size = 1 + n*(q-1) = 1 + 6*2 = 13... wait
    # Actually for ternary: 1 + 6*2 = 13? No: 1 + n*(q-1) = 1 + 6*2 = 13. But we need to verify.
    # Hamming distance <= 1 means: same word, or differ in exactly 1 position.
    # In 1 position, there are (q-1) = 2 alternatives. So ball size = 1 + n*(q-1) = 1 + 12 = 13.
    
    # Precompute balls for all words
    balls = []
    for word in all_words:
        ball = [word]
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = word[:]
                    new_word[pos] = val
                    ball.append(new_word)
        balls.append(ball)
    
    # Convert balls to sets of indices for fast lookup
    ball_indices = []
    for ball in balls:
        idx_set = set()
        for w in ball:
            idx_set.add(word_to_idx[tuple(w)])
        ball_indices.append(idx_set)
    
    # Greedy initialization: pick words that cover the most uncovered words
    uncovered = set(range(total_words))
    codewords = []
    
    while uncovered and time.time() < deadline - 30:
        best_word = -1
        best_count = -1
        # Sample some candidates if too many to check
        if len(uncovered) > 100:
            # Check a random sample plus the best so far
            candidates = list(uncovered)
            if len(candidates) > 200:
                random.shuffle(candidates)
                candidates = candidates[:200]
            # Also check some random words
            for _ in range(50):
                candidates.append(random.randint(0, total_words - 1))
        else:
            candidates = list(uncovered)
        
        for idx in candidates:
            covered = len(ball_indices[idx] & uncovered)
            if covered > best_count:
                best_count = covered
                best_word = idx
        
        if best_word < 0:
            break
        
        codewords.append(best_word)
        uncovered -= ball_indices[best_word]
        
        if time.time() > deadline - 30:
            break
    
    # Local improvement: try to replace codewords to reduce count or improve coverage
    # First, ensure full coverage
    while uncovered:
        # Pick word that covers most uncovered
        best_word = -1
        best_count = -1
        for idx in uncovered:
            covered = len(ball_indices[idx] & uncovered)
            if covered > best_count:
                best_count = covered
                best_word = idx
        if best_word < 0:
            break
        codewords.append(best_word)
        uncovered -= ball_indices[best_word]
        if time.time() > deadline - 10:
            break
    
    # Now try to remove redundant codewords
    # A codeword is redundant if removing it still leaves full coverage
    def is_redundant(cw_idx, codeword_list):
        # Check if all words covered by this codeword are also covered by others
        cw_ball = ball_indices[cw_idx]
        for other_idx in codeword_list:
            if other_idx == cw_idx:
                continue
            other_ball = ball_indices[other_idx]
            # If cw_ball is subset of union of other balls
            # Simplified: check if every word in cw_ball is in some other ball
            for w in cw_ball:
                covered_by_other = False
                for o in codeword_list:
                    if o != cw_idx and w in ball_indices[o]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    return False
        return True
    
    # Better: use a coverage count approach
    def try_remove_redundant(codeword_list):
        if not codeword_list:
            return codeword_list
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Count how many codewords cover each word
            coverage_count = [0] * total_words
            for cw in codeword_list:
                for w in ball_indices[cw]:
                    coverage_count[w] += 1
            
            for i, cw in enumerate(codeword_list):
                if time.time() > deadline - 5:
                    break
                # Check if removing cw leaves all words covered
                can_remove = True
                for w in ball_indices[cw]:
                    if coverage_count[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove it
                    del codeword_list[i]
                    changed = True
                    break  # Restart the loop since indices changed
        
        return codeword_list
    
    codewords = try_remove_redundant(codewords)
    
    # Try swapping: replace one codeword with another that allows further removal
    # This is more complex, so let's try a simpler approach:
    # Try adding a new codeword and see if we can remove multiple old ones
    
    # Convert to list of actual codewords
    result = [all_words[cw] for cw in codewords]
    
    # Verify coverage
    covered_set = set()
    for cw in codewords:
        covered_set |= ball_indices[cw]
    
    if len(covered_set) < total_words:
        # Add more codewords to cover remaining
        uncovered = set(range(total_words)) - covered_set
        while uncovered and time.time() < deadline - 2:
            best_word = -1
            best_count = -1
            for idx in uncovered:
                covered = len(ball_indices[idx] & uncovered)
                if covered > best_count:
                    best_count = covered
                    best_word = idx
            if best_word < 0:
                break
            codewords.append(best_word)
            uncovered -= ball_indices[best_word]
            covered_set |= ball_indices[best_word]
        
        result = [all_words[cw] for cw in codewords]
    
    return result
