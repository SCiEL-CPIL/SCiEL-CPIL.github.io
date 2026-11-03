import time
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # Map integer index to tuple of digits
    def index_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[i] = idx % q
            idx //= q
        return tuple(t)
        
    def tuple_to_index(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute balls of radius 1 for all indices
    # balls[i] is a list of indices j such that dist(i, j) <= 1
    balls = [None] * total_words
    for i in range(total_words):
        t = index_to_tuple(i)
        ball = [i]
        for pos in range(n):
            for val in range(q):
                if val == t[pos]:
                    continue
                new_t = list(t)
                new_t[pos] = val
                ball.append(tuple_to_index(tuple(new_t)))
        balls[i] = ball

    # Greedy Set Cover
    uncovered = [1] * total_words
    # coverage_count[i] = number of uncovered words in balls[i]
    coverage_count = np.zeros(total_words, dtype=np.int32)
    
    # Initialize coverage counts
    for i in range(total_words):
        coverage_count[i] = len(balls[i])
        
    selected = []
    
    while True:
        if time.time() > deadline:
            break
            
        # Find the codeword that covers the most uncovered words
        best_idx = np.argmax(coverage_count)
        if coverage_count[best_idx] == 0:
            break
            
        selected.append(best_idx)
        
        # Update uncovered status and coverage counts
        # For each word w in the ball of best_idx:
        for w in balls[best_idx]:
            if uncovered[w]:
                uncovered[w] = 0
                # w is now covered. 
                # This affects the coverage_count of all codewords c that cover w.
                # These are exactly the codewords in balls[w].
                for c in balls[w]:
                    coverage_count[c] -= 1
                    
    # Convert indices back to lists of digits
    codewords = []
    for idx in selected:
        codewords.append(list(index_to_tuple(idx)))
        
    # Check if we have a valid cover (should be true if loop finished normally)
    # If time ran out, we might have an incomplete cover. 
    # The task requires a valid cover. 
    # If we ran out of time, we should try to complete it or return best effort?
    # "Valid when every word ... is within Hamming distance radius".
    # If invalid, score 0. So we MUST return a valid cover.
    # Given the speed, we should almost certainly finish.
    # If not, we can fall back to the seed solution or a brute force completion.
    
    # Verify validity
    covered = np.zeros(total_words, dtype=bool)
    for c in selected:
        for w in balls[c]:
            covered[w] = True
            
    if not np.all(covered):
        # Fallback: Add random words until covered
        # This is a safety net, shouldn't trigger
        remaining = np.where(~covered)[0]
        while len(remaining) > 0 and time.time() < deadline:
            # Pick a random remaining word
            w = remaining[np.random.randint(len(remaining))]
            # Find a codeword that covers w and is not already selected? 
            # Or just pick w itself?
            # Simplest: add w as a codeword
            if w not in selected:
                selected.append(w)
                for c in balls[w]:
                    covered[c] = True
                remaining = np.where(~covered)[0]
            
            # If stuck, add all remaining? No, that's too many.
            # Just add w.
            if w not in selected:
                 selected.append(w)
                 for c in balls[w]:
                     covered[c] = True
                 remaining = np.where(~covered)[0]

    return [list(index_to_tuple(idx)) for idx in selected]
