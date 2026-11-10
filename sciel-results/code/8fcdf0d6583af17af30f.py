import time
import numpy as np

def construct(q, n, radius):
    # Deadline guard: 150s limit, aim for 110s safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Specific parameters for the task: q=3, n=7, radius=1
    # Total space size: 3^7 = 2187
    # We represent words as integers 0..2186 in base-q
    
    # Precompute the "ball" of each word: which words are within radius?
    # For q=3, n=7, radius=1, each word covers 1 + n*(q-1) = 1 + 7*2 = 15 words.
    
    # 1. Convert all words to base-q digits for distance calculation?
    # Actually, it's faster to precompute the neighbors for each integer in 0..q^n-1.
    # Since q^n is small (2187), we can afford a lookup table.
    
    total_words = q ** n
    # neighbors[i] is a numpy array of indices of words within radius of i
    neighbors = []
    
    # Helper to convert int to base-q tuple
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t

    # Helper to convert base-q tuple to int
    def tuple_to_int(t, q):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * q + t[i]
        return x

    # Precompute neighbors
    # This might take a few seconds but is worth it for the speed of the search.
    # 2187 * 15 checks is trivial.
    # Actually, let's just generate the ball for each center.
    # Ball size is 1 + n*(q-1).
    ball_size = 1 + n * (q - 1)
    
    # Optimization: Instead of storing lists of lists, store a single flat array?
    # Or just a list of numpy arrays.
    # Let's try to be efficient.
    
    # We will maintain a coverage count array: cover_count[j] = number of codewords covering j
    # And a list of codewords.
    
    # Phase 1: Greedy Set Cover
    # Start with empty set.
    # At each step, pick the codeword that covers the most *currently uncovered* words.
    
    # To speed this up, we can maintain an array of uncovered words.
    # However, standard greedy is O(N * M) where N is steps, M is space size.
    # N is ~150-200, M is 2187. 200 * 2187 * 15 (ball size) ~ 6.5M ops. Very fast.
    
    # Let's precompute the balls.
    # balls[i] = set of indices covered by i
    # Using numpy sets is slow. Using boolean masks or lists?
    # Let's use a list of lists for simplicity, then convert to numpy arrays for counting?
    
    # Actually, for greedy, we need to find argmax of sum(cover[j] == 0 for j in ball[i]).
    
    # Let's precompute balls as lists of ints.
    balls = [None] * total_words
    for i in range(total_words):
        t_i = int_to_tuple(i, n, q)
        ball = [i]
        # Generate all words at distance 1
        for pos in range(n):
            orig_val = t_i[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                t_new = t_i[:]
                t_new[pos] = new_val
                ball.append(tuple_to_int(t_new, q))
        balls[i] = ball
        
    # Greedy Selection
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    # Optimization: Keep track of uncovered count per candidate?
    # Or just recompute each step? 
    # Steps ~ 150. Recomputing score for 2187 candidates each step:
    # 150 * 2187 * 15 (ball size) = ~5M operations. Python loops are slow.
    # Vectorization is key.
    
    # We can represent the coverage state as a boolean array `covered`.
    # For each candidate i, score = sum(~covered[j] for j in balls[i])
    
    # Let's try a simple loop first, see if it's fast enough.
    # If not, we can use numpy.
    
    # Better approach:
    # Maintain an array `uncovered_count` for each candidate?
    # When we add a codeword c, for every j in balls[c], if covered[j] was False, 
    # then for every candidate i such that j in balls[i], uncovered_count[i] -= 1.
    # This is complex to precompute (inverse balls).
    
    # Let's stick to simple greedy with numpy acceleration for the scoring step.
    # We can't easily vectorize "sum over variable length balls" without a matrix.
    # A 2187x2187 boolean matrix is 4.7M booleans = ~5MB. This is feasible!
    # Let's create a coverage matrix M where M[i,j] = 1 if j is in balls[i].
    
    # 2187 * 2187 is ~4.8 million entries.
    # Memory: 4.8M * 1 byte (bool) = 4.8 MB. Very safe.
    
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        for j in balls[i]:
            M[i, j] = 1
            
    # Now greedy step:
    # covered is a boolean array of shape (total_words,)
    # uncovered is ~covered
    # score[i] = np.sum(M[i] * uncovered)
    # We want argmax score.
    
    # We can compute all scores at once:
    # scores = M @ uncovered (since M is 0/1 and uncovered is 0/1)
    # Wait, M @ uncovered gives the count of uncovered words covered by each i.
    
    # Let's implement the loop.
    while True:
        if time.time() > deadline - 10:
            break
            
        # Compute scores
        # uncovered is boolean, cast to float/int for dot product?
        # M is uint8. uncovered is bool.
        # scores = M.dot(uncovered.astype(np.uint8))
        
        uncovered_mask = ~covered
        if not np.any(uncovered_mask):
            break
            
        scores = M.dot(uncovered_mask.astype(np.uint8))
        best_idx = np.argmax(scores)
        
        if scores[best_idx] == 0:
            # No improvement possible? Should not happen if not fully covered.
            # But if some words are isolated? No, radius 1 covers everything.
            break
            
        codewords.append(best_idx)
        # Update covered
        covered[balls[best_idx]] = True
        
        # Check time
        if time.time() > deadline - 10:
            break
            
    # Phase 2: Local Search (Pruning and Swapping)
    # Current size: len(codewords)
    # Goal: Reduce size.
    
    # Strategy:
    # 1. Try to remove a codeword c if all words it uniquely covers are covered by others.
    #    i.e., if for all j in balls[c], there exists c' in codewords \ {c} such that j in balls[c'].
    #    Equivalent: if (sum of M[c'] for c' in others)[j] >= 1 for all j in balls[c].
    #    Or simpler: coverage_count[j] > 1 for all j in balls[c].
    
    # Let's maintain coverage counts.
    # cover_count[j] = number of codewords covering j.
    
    cover_count = np.zeros(total_words, dtype=np.uint8)
    for c in codewords:
        cover_count[balls[c]] += 1
        
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try to remove
        # For each c in codewords, check if min(cover_count[balls[c]]) > 1
        # If so, remove c.
        
        # To be efficient, we can check candidates.
        # Randomize order? Or iterate?
        # Iterating 150-200 items is fast.
        
        # We need to be careful: removing c changes cover_count for balls[c].
        
        # Let's try a random removal attempt loop.
        # Actually, deterministic removal of "redundant" codewords is safe and good.
        
        # Find a redundant codeword.
        # redundant if np.min(cover_count[balls[c]]) > 1
        
        # We can vectorize this check?
        # min_cover = np.min(cover_count[balls[c]] for c in codewords) ... hard to vectorize easily without list comprehension.
        
        # List comprehension is fine for ~200 items.
        
        redundant_idx = -1
        for idx, c in enumerate(codewords):
            if np.min(cover_count[balls[c]]) > 1:
                redundant_idx = idx
                break
                
        if redundant_idx != -1:
            c_to_remove = codewords[redundant_idx]
            # Update cover_count
            cover_count[balls[c_to_remove]] -= 1
            # Remove from list
            codewords.pop(redundant_idx)
            improved = True
            continue # Restart check from beginning as state changed
            
        # If no removal possible, try swaps?
        # Swap: Replace c with c' such that size doesn't change but we might enable future removals?
        # Or just try to find a better codeword?
        # Given time constraints, simple removal is the most robust improvement over pure greedy.
        # Pure greedy often leaves redundant codewords.
        
        # If no removal, we can try a "kick":
        # Randomly remove a codeword, then greedily fill the gap?
        # This is risky for validity.
        
        # Let's try a limited swap search:
        # For a random subset of codewords, try replacing one with a new one that covers more "tight" spots?
        # Actually, let's just try to remove one, then re-greedy the uncovered spots.
        
        # Alternative: 
        # If no direct removal, try to swap c with c_new such that c_new covers the "weakly covered" words better.
        
        # Let's implement a simple "kick" restart if stuck?
        # No, let's stick to removal. It's safe.
        
        # If we are stuck, we can try to swap two codewords?
        # Or just break.
        
        # Let's add a simple swap heuristic:
        # Find c with min "redundancy" (max min_cover_count is low).
        # Find c_new not in codewords that maximizes coverage of words with cover_count == 1?
        
        # This is getting complex. Let's rely on the removal loop. 
        # Greedy usually produces ~180-190. Removal can get it to ~170-180.
        # The target is 186.
        
        # If no removal found, break.
        break
        
    # Convert back to list of lists
    result = []
    for c in codewords:
        result.append(int_to_tuple(c, n, q))
        
    return result

# EVOLVE-BLOCK-END
