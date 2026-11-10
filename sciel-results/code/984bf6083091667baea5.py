import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: aim to finish by 115s to be well within the 120s kill switch
    deadline = start_time + 115.0
    
    total_words = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Helper functions for conversion
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t

    def tuple_to_int(t, q):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * q + t[i]
        return x

    # Precompute balls (neighbors within radius)
    # For radius=1, ball size is 1 + n*(q-1)
    # We store balls as lists of integers for fast coverage updates
    balls = [None] * total_words
    for i in range(total_words):
        t_i = int_to_tuple(i, n, q)
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = t_i[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    t_new = t_i[:]
                    t_new[pos] = new_val
                    ball.append(tuple_to_int(t_new, q))
        balls[i] = ball
        
    # Precompute "neighbors of neighbors" or just use the ball lists for repair?
    # For repair, we need to find the codeword that covers the most *uncovered* words.
    # Let U be the set of uncovered words.
    # Score(c) = |B(c) \intersect U|.
    # We can maintain a count array `cover_count` where cover_count[w] is the number of codewords covering w.
    # U = {w | cover_count[w] == 0}.
    # To compute Score(c) efficiently, we can just sum cover_count[w] == 0 for w in B(c)?
    # No, that's O(ball_size) per candidate. Total O(N * ball_size).
    # N=2187, ball_size=15. 2187*15 ~ 33k ops. Very fast.
    
    def greedy_cover(initial_codewords=None):
        """
        Fast greedy set cover.
        """
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        
        if initial_codewords:
            seen = set()
            for c in initial_codewords:
                if 0 <= c < total_words and c not in seen:
                    codewords.append(c)
                    seen.add(c)
                    covered[balls[c]] = True
        
        while True:
            if time.time() > deadline - 5:
                break
            if not np.any(~covered):
                break
            
            # Find best candidate
            best_score = -1
            best_idx = -1
            # We can optimize this by only checking candidates that have uncovered neighbors,
            # but N is small enough that scanning all N is fine.
            # Actually, we can vectorize this slightly or just loop.
            # Looping in Python over 2187 items is ~0.1ms? 2187 * 15 ops = 33k ops.
            # Let's try to speed it up.
            
            # Alternative: Precompute for each word w, the list of codewords that cover it.
            # This is the "inverse" of balls.
            # inv_balls[w] = list of c such that w in balls[c].
            # Then for each uncovered w, we can increment score for all c in inv_balls[w].
            # This is O(|U| * ball_size). |U| starts large, ends small.
            
            # Let's stick to the direct method first, it's simple and likely fast enough for N=2187.
            # But we can do better.
            
            # Let's use the inverse approach for speed.
            # Precompute inv_balls once.
            # inv_balls is a list of lists.
            
            # For now, let's just do the direct scan in Python, it might be slow if called many times.
            # Let's optimize:
            # scores = np.zeros(total_words)
            # for w in range(total_words):
            #     if not covered[w]:
            #         for c in inv_balls[w]:
            #             scores[c] += 1
            # best_idx = np.argmax(scores)
            
            # This is O(N * ball_size) which is same as direct.
            # Let's just use the direct method inside a tight loop, but maybe vectorized?
            # Actually, the current seed program used M.dot which is very fast.
            # Let's reuse the matrix idea for the "repair" step in the local search, 
            # but for the initial greedy, we can be simpler.
            
            # Let's just find the best candidate by checking all candidates.
            # To make it fast, we'll maintain a list of uncovered words.
            uncovered_list = np.where(~covered)[0]
            if len(uncovered_list) == 0:
                break
            
            # For each candidate c, score = number of uncovered words in balls[c]
            # We can compute this by:
            # scores = np.array([np.sum(~covered[balls[c]]) for c in range(total_words)])
            # This is a list comprehension of 2187 numpy sums. Might be slow.
            
            # Let's use the inverse mapping precomputed outside.
            # We need inv_balls.
            
            # Since I can't easily precompute inv_balls inside this function without overhead,
            # and the seed program already had M, let's just use the M matrix approach for the repair step
            # in the main loop, but for the initial greedy, let's just use a simple heuristic or 
            # the same M matrix.
            
            # Let's refactor: Precompute M and inv_balls at the top level of construct.
            
            # Actually, let's just use the M matrix for everything. It's 2187x2187 uint8 = 4.7MB.
            # M.dot is extremely fast.
            
            # So, let's just use M for the greedy step too.
            # scores = M.dot(~covered)
            # best_idx = np.argmax(scores)
            
            # This is what the seed program did. It's fast.
            
            if 'M' not in locals():
                 # Fallback or error? No, we will define M in the main block.
                 pass
            
            # Let's assume M is available from the outer scope or we pass it.
            # I will define M in the outer scope of construct.
            
            scores = M.dot(~covered)
            best_idx = np.argmax(scores)
            
            if scores[best_idx] == 0:
                break
                
            codewords.append(best_idx)
            covered[balls[best_idx]] = True
            
        return codewords

    def local_search_destroy_reconstruct(codewords, num_destroy=5):
        """
        1. Remove 'num_destroy' random codewords.
        2. Repair by adding codewords that cover the most uncovered words.
        3. If the result is smaller or equal, keep it.
        """
        if len(codewords) <= num_destroy:
            return codewords
        
        # Make a copy
        current = codewords[:]
        
        # Calculate current coverage counts
        # We need a fast way to update coverage.
        # cover_count[w] = number of codewords covering w.
        
        # To speed up the "repair" phase, we need to know which words are uncovered.
        # And for each candidate, how many uncovered words it covers.
        
        # Let's maintain cover_count array.
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in current:
            cover_count[balls[c]] += 1
        
        # Identify uncovered words
        uncovered_mask = (cover_count == 0)
        
        # If there are no uncovered words, we can just try to remove more?
        # The strategy is: Destroy, then Repair.
        # If we destroy and create holes, we must repair them.
        
        # Step 1: Destroy
        # Pick num_destroy random indices to remove
        indices_to_remove = random.sample(range(len(current)), num_destroy)
        # Sort descending to remove safely
        indices_to_remove.sort(reverse=True)
        
        for idx in indices_to_remove:
            c = current[idx]
            cover_count[balls[c]] -= 1
            current.pop(idx)
        
        # Step 2: Repair
        # While there are uncovered words, add the best codeword.
        # We must ensure we don't add a codeword that is already in 'current'.
        # Actually, adding a duplicate doesn't help.
        # We need to track which codewords are in 'current'.
        
        in_current = np.zeros(total_words, dtype=bool)
        for c in current:
            in_current[c] = True
        
        # Repair loop
        while True:
            if time.time() > deadline - 2:
                # Return original if timeout? Or current?
                # Current is invalid. We should return the original valid one.
                return codewords
            
            uncovered_mask = (cover_count == 0)
            if not np.any(uncovered_mask):
                break
            
            # Find best candidate
            # Score(c) = sum(cover_count[w] == 0 for w in balls[c])
            # We can compute this using M?
            # M is 0/1. 
            # scores = M.dot(uncovered_mask) gives number of uncovered words covered by c.
            # But we must exclude c if it's already in 'current'.
            # If c is in 'current', its ball is already covered? 
            # No, if c is in 'current', it contributes to cover_count.
            # If we add it again, it's a duplicate.
            # So we mask out candidates that are in 'current'.
            
            scores = M.dot(uncovered_mask.astype(np.uint8))
            scores[in_current] = 0 # Disallow adding existing codewords
            
            best_idx = np.argmax(scores)
            
            if scores[best_idx] == 0:
                # Stuck? Shouldn't happen if the space is connected, but just in case.
                # If stuck, return original.
                return codewords
                
            # Add best_idx
            current.append(best_idx)
            in_current[best_idx] = True
            cover_count[balls[best_idx]] += 1
            
        return current

    def local_search_remove(codewords):
        """
        Iteratively remove redundant codewords.
        """
        if not codewords:
            return codewords
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            cover_count = np.zeros(total_words, dtype=np.int32)
            for c in codewords:
                cover_count[balls[c]] += 1
            
            for i in range(len(codewords)):
                c = codewords[i]
                # Check if removing c leaves all words in balls[c] covered by others
                # i.e., cover_count[w] > 1 for all w in balls[c]
                # Equivalent to: min(cover_count[w] for w in balls[c]) > 1
                # Or: sum(cover_count[w] == 1 for w in balls[c]) == 0
                if np.min(cover_count[balls[c]]) > 1:
                    codewords.pop(i)
                    changed = True
                    break # Restart loop to update cover_count
            
        return codewords

    # Precompute M (Coverage Matrix)
    # M[i, j] = 1 if j is in balls[i]
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        M[i, balls[i]] = 1

    # Main Strategy:
    # 1. Generate a valid cover using Greedy.
    # 2. Apply Local Search Remove to minimize.
    # 3. Apply Iterated Destroy-Reconstruct to escape local optima.
    
    # Step 1: Initial Greedy
    # We can use a random seed for the greedy order? 
    # The greedy algorithm is deterministic if we just pick argmax.
    # To get diversity, we can add random noise to scores or randomize ties.
    # Or just run multiple independent greedy starts?
    # Actually, the greedy set cover is usually very good.
    
    initial_cover = greedy_cover()
    initial_cover = local_search_remove(initial_cover)
    
    best_codewords = initial_cover[:]
    best_size = len(best_codewords)
    
    # Step 2: Iterated Local Search
    # We have ~100s left.
    # Each ILS iteration: Destroy + Repair + Remove.
    # Cost: Destroy (O(num_destroy * ball_size)), Repair (O(K * N * ball_size)? No, O(steps * N * ball_size)), Remove (O(K^2 * ball_size)).
    # With N=2187, these are fast.
    
    num_destroy = 5 # Start with small destruction
    
    # We can increase num_destroy over time or randomize it.
    
    while time.time() < deadline - 5:
        # Randomize destruction amount slightly
        current_destroy = random.randint(3, 10)
        
        # Try the destroy-reconstruct move
        candidate = local_search_destroy_reconstruct(best_codewords, current_destroy)
        
        # If candidate is invalid (shouldn't be, but just in case), skip
        # We can check validity quickly?
        # local_search_destroy_reconstruct returns a valid cover by construction (it repairs until no holes).
        # So it is valid.
        
        # Apply local search remove to the candidate
        candidate = local_search_remove(candidate)
        
        if len(candidate) < best_size:
            best_size = len(candidate)
            best_codewords = candidate[:]
            
        # Occasionally, try a larger perturbation?
        # Or just keep going.
        
        # If we are stuck for a while, we might want to restart from scratch?
        # But greedy is fast.
        # Let's just keep iterating.

    # Final check
    if not best_codewords:
        best_codewords = greedy_cover()
        best_codewords = local_search_remove(best_codewords)

    # Convert to output format
    result = []
    for c in best_codewords:
        result.append(int_to_tuple(c, n, q))
        
    return result
