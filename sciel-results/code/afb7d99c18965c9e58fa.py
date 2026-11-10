import numpy as np
import time
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to be safe
    
    # For the specific task, q=3, n=8, radius=1
    # But the function should be general enough or at least handle the specific case well.
    # The prompt says "the evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies we might receive different parameters, but the task is specifically about K_3(8,1).
    # However, the seed program uses a specific construction for n=8, q=3.
    # If q, n, radius are different, we should probably fall back to a generic method or the seed's logic if applicable.
    # But the seed's logic is specific to n=8, q=3.
    # Let's assume the evaluator always passes q=3, n=8, radius=1 for this task.
    # If not, we can return a trivial cover (all words) or use the seed if n>4.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed logic if possible, else trivial
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            import itertools
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=8, radius=1
    total_words = 3**8
    # Map word to index
    # word is list of 8 ints
    # index = w[0] + w[1]*3 + ... + w[7]*3^7
    
    # Precompute neighbors for all words
    # neighbors[i] is a list of indices of words within distance 1 of word i
    neighbors = [[] for _ in range(total_words)]
    
    # Helper to convert index to word
    def index_to_word(idx):
        w = [0]*n
        for i in range(n):
            w[i] = idx % 3
            idx //= 3
        return w
        
    # Helper to convert word to index
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx += w[i] * (3**i)
        return idx

    # Precompute neighbors
    for i in range(total_words):
        w = index_to_word(i)
        neighbors[i].append(i) # distance 0
        for pos in range(n):
            for delta in [1, 2]: # equivalent to -1, +1 mod 3
                new_val = (w[pos] + delta) % 3
                w[pos] = new_val
                j = word_to_index(w)
                neighbors[i].append(j)
        # Restore w not needed since we recompute or use local var
    
    # Greedy set cover
    covered = [False] * total_words
    codewords = []
    
    # To speed up, we can maintain a count of uncovered neighbors for each candidate
    # uncovered_count[i] = number of neighbors of i that are not yet covered
    uncovered_count = [len(neighbors[i]) for i in range(total_words)]
    
    # We need to pick the word with max uncovered_count
    # Then mark its neighbors as covered, and update uncovered_count for all other words
    
    # This is O(N^2) in the worst case if we scan all N words each time.
    # N = 6561. N^2 = 43 million.
    # Each step involves scanning 6561 words to find max, then updating counts for neighbors of the chosen word.
    # Updating counts: for each neighbor j of chosen word c, for each neighbor k of j, if covered[j] was just set to True, decrement uncovered_count[k]?
    # No, uncovered_count[k] is the number of uncovered neighbors of k.
    # When we cover a word j, we need to decrement uncovered_count[k] for all k such that j is a neighbor of k.
    # Since the neighbor relation is symmetric, k is a neighbor of j iff j is a neighbor of k.
    # So for each j in neighbors[c], we iterate through neighbors[j] and decrement uncovered_count[k] if j was uncovered.
    
    # Complexity per step:
    # 1. Find max uncovered_count: O(N)
    # 2. For each j in neighbors[c] (17 items):
    #    If not covered[j]:
    #       covered[j] = True
    #       For each k in neighbors[j] (17 items):
    #           uncovered_count[k] -= 1
    # Total updates per step: 17 * 17 = 289.
    # Total steps: ~500.
    # Total operations: 500 * (6561 + 289) ~ 3.4 million.
    # This is very fast in Python.
    
    while True:
        if time.time() > deadline:
            break
            
        # Find word with max uncovered_count
        max_count = -1
        max_idx = -1
        for i in range(total_words):
            if uncovered_count[i] > max_count:
                max_count = uncovered_count[i]
                max_idx = i
                if max_count == 0:
                    break
        
        if max_count == 0:
            break
        
        codewords.append(max_idx)
        
        # Update covered status and counts
        for j in neighbors[max_idx]:
            if not covered[j]:
                covered[j] = True
                for k in neighbors[j]:
                    uncovered_count[k] -= 1
                    
    # Convert indices back to words
    result = [index_to_word(i) for i in codewords]
    
    return result
