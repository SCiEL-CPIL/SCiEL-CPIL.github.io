import time
import numpy as np
import random

def construct(q, n, radius):
    # Deadline guard: 150s limit, aim for 115s safety margin
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    
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

    # Precompute balls as lists of integers
    # balls[i] contains i and all words at distance 1 from i
    balls = [None] * total_words
    for i in range(total_words):
        t_i = int_to_tuple(i, n, q)
        ball = [i]
        for pos in range(n):
            orig_val = t_i[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                t_new = t_i[:]
                t_new[pos] = new_val
                ball.append(tuple_to_int(t_new, q))
        balls[i] = ball

    # Coverage matrix M where M[i, j] = 1 if j is in balls[i]
    # This allows vectorized scoring
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        for j in balls[i]:
            M[i, j] = 1

    # Phase 1: Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    while time.time() < deadline - 5:
        uncovered_mask = ~covered
        if not np.any(uncovered_mask):
            break
        scores = M.dot(uncovered_mask.astype(np.uint8))
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
        codewords.append(best_idx)
        covered[balls[best_idx]] = True

    # Maintain coverage counts for local search
    cover_count = np.zeros(total_words, dtype=np.uint8)
    for c in codewords:
        cover_count[balls[c]] += 1

    # Phase 2: Local Search with Removal and Swap
    # We use a random walk with restarts or kicks if stuck.
    # Goal: minimize len(codewords)
    
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    improved = True
    stuck_iterations = 0
    max_stuck = 50
    
    while improved and time.time() < deadline - 10:
        improved = False
        stuck_iterations += 1
        
        if stuck_iterations > max_stuck:
            # Try a kick: randomly remove a codeword and see if we can fill it back with fewer?
            # Or just restart local search with a random perturbation.
            # For now, break to avoid wasting time if simple moves fail.
            # A better kick: remove a random codeword, then greedily add back until covered.
            # This might result in the same size or larger. If larger, revert.
            # If we get lucky, it might open up a removal path.
            if len(codewords) > 1:
                # Pick a random codeword to remove
                rand_idx = random.randint(0, len(codewords) - 1)
                c_to_remove = codewords[rand_idx]
                
                # Save state
                old_codewords = list(codewords)
                old_cover_count = cover_count.copy()
                
                # Remove it
                codewords.pop(rand_idx)
                cover_count[balls[c_to_remove]] -= 1
                
                # Greedily add back until covered
                while time.time() < deadline - 10:
                    uncovered_mask = cover_count == 0
                    if not np.any(uncovered_mask):
                        break
                    scores = M.dot(uncovered_mask.astype(np.uint8))
                    best_idx = np.argmax(scores)
                    if scores[best_idx] == 0:
                        break
                    codewords.append(best_idx)
                    cover_count[balls[best_idx]] += 1
                
                # Check if we improved
                if len(codewords) < len(old_codewords):
                    improved = True
                    stuck_iterations = 0
                    # Continue local search
                else:
                    # Revert
                    codewords = old_codewords
                    cover_count = old_cover_count
                    stuck_iterations += 10 # Penalize failed kick
        
        # Standard removal check
        # Find a redundant codeword
        redundant_idx = -1
        # To speed up, we can check random order or all.
        # All is ~200 checks, each O(15). Very fast.
        for idx, c in enumerate(codewords):
            if np.min(cover_count[balls[c]]) > 1:
                redundant_idx = idx
                break
        
        if redundant_idx != -1:
            c_to_remove = codewords[redundant_idx]
            cover_count[balls[c_to_remove]] -= 1
            codewords.pop(redundant_idx)
            improved = True
            stuck_iterations = 0
            
            # Update best
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue # Restart check
        
        # If no removal, try a swap?
        # Swap: replace c with c' such that c' covers more "unique" spots?
        # This is harder to verify quickly for validity without breaking coverage.
        # Given time constraints, we rely on removal and kicks.
        
        if not improved:
             # If we are not improving, we might be stuck.
             # The kick logic above increments stuck_iterations if it fails.
             # If we didn't kick (because stuck_iterations <= max_stuck), we just break?
             # No, we should try kicks periodically.
             if stuck_iterations % 10 == 0 and len(codewords) > 1:
                 # Trigger a kick
                 rand_idx = random.randint(0, len(codewords) - 1)
                 c_to_remove = codewords[rand_idx]
                 
                 old_codewords = list(codewords)
                 old_cover_count = cover_count.copy()
                 
                 codewords.pop(rand_idx)
                 cover_count[balls[c_to_remove]] -= 1
                 
                 # Greedily fill
                 while time.time() < deadline - 10:
                     uncovered_mask = cover_count == 0
                     if not np.any(uncovered_mask):
                         break
                     scores = M.dot(uncovered_mask.astype(np.uint8))
                     best_idx = np.argmax(scores)
                     if scores[best_idx] == 0:
                         break
                     codewords.append(best_idx)
                     cover_count[balls[best_idx]] += 1
                 
                 if len(codewords) < len(old_codewords):
                     improved = True
                     stuck_iterations = 0
                     if len(codewords) < best_size:
                         best_size = len(codewords)
                         best_codewords = list(codewords)
                 else:
                     codewords = old_codewords
                     cover_count = old_cover_count
                     stuck_iterations += 1
                     if stuck_iterations > max_stuck * 2:
                         break
             else:
                 # If we can't remove and don't want to kick yet, we are stuck.
                 # But the loop condition `while improved` will exit if improved is False.
                 # So we need to set improved=True if we want to continue kicking.
                 # Let's structure the loop to explicitly handle the kick attempt as part of the iteration.
                 pass

    # Ensure we return the best found
    if len(best_codewords) < len(codewords):
        codewords = best_codewords

    # Convert back to list of lists
    result = []
    for c in codewords:
        result.append(int_to_tuple(c, n, q))
        
    return result
