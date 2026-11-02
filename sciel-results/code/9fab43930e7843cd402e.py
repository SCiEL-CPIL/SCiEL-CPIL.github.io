import random
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate all words
    all_words = []
    for i in range(q ** n):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        all_words.append(tuple(w))
    
    N = len(all_words)
    
    # Precompute neighbors (Hamming ball of radius 1)
    # neighbors[i] is a list of indices of words within distance 1 of all_words[i]
    neighbors = [[] for _ in range(N)]
    
    # To speed up neighbor lookup, we can use the fact that distance 1 means changing one coordinate.
    # For each word index i, generate all words at distance 1.
    # We need a mapping from word tuple to index.
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    for i, w in enumerate(all_words):
        neighbors[i].append(i) # distance 0
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                j = word_to_idx[new_w]
                neighbors[i].append(j)
                
    # Greedy Set Cover
    best_code = None
    best_size = float('inf')
    
    # We can run multiple restarts with different random seeds for tie-breaking
    # Since the problem is small, let's do a few restarts.
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        covered = [False] * N
        gains = [0] * N
        # Initially, all words are uncovered, so gain for each center is the size of its ball
        # Ball size is 1 + n * (q - 1)
        ball_size = 1 + n * (q - 1)
        for i in range(N):
            gains[i] = ball_size
        
        code = []
        uncovered_count = N
        
        # To break ties randomly, we can shuffle indices or add small random noise to gains?
        # Better: when finding max gain, if there are ties, pick one randomly.
        # Or, we can just pick the first max. Randomizing the order of iteration might help.
        
        while uncovered_count > 0:
            # Find index with max gain
            max_gain = -1
            best_idx = -1
            # Optimization: if we just need *a* max, we can scan.
            # To introduce randomness in ties, we could collect all maxes and pick one.
            # But scanning is O(N). N=729 is small.
            
            # Let's collect candidates with max gain to allow random selection
            candidates = []
            for i in range(N):
                if gains[i] > max_gain:
                    max_gain = gains[i]
                    candidates = [i]
                elif gains[i] == max_gain:
                    candidates.append(i)
            
            if max_gain == 0:
                break # Should not happen if uncovered_count > 0 and logic is correct, but safety
                
            # Pick one candidate randomly
            chosen = random.choice(candidates)
            
            code.append(chosen)
            
            # Update covered and gains
            for v in neighbors[chosen]:
                if not covered[v]:
                    covered[v] = True
                    uncovered_count -= 1
                    # For each neighbor u of v, decrement gain[u]
                    # Because v is now covered, it no longer contributes to the gain of u
                    for u in neighbors[v]:
                        gains[u] -= 1
                        
        if len(code) < best_size:
            best_size = len(code)
            best_code = [list(all_words[i]) for i in code]
            
    # If we didn't find any code (should not happen), return empty or seed
    if best_code is None:
        return []
        
    return best_code
