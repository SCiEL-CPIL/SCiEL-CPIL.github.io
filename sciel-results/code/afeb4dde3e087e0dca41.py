import time
import random
import sys

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # But the function signature is generic. We assume the evaluator passes these.
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # If parameters don't match the specific hard problem, fallback to simple greedy
    # But the task is specifically K_3(7,1).
    
    # 1. Generate all words and their balls
    # Map word tuple to index 0..3^n-1
    # For q=3, n=7, total words = 2187
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: simple greedy for general case
        return _generic_greedy(q, n, radius, deadline)

    total_words = 3 ** n
    # Precompute balls for all words
    # A ball of radius 1 around a word w includes w and all words at dist 1.
    balls = [0] * total_words
    
    # Helper to convert index to tuple
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for speed
    # For each word, generate the 15 words in its ball
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 1 << i
        balls[i] = mask
        # Generate neighbors at distance 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                # Efficiently compute index of neighbor
                # Neighbor differs at 'pos'
                # index = ... 
                # It's faster to just use the word tuple and a mapping if we had one, 
                # but we don't have a global map from tuple to idx in O(1) without a dict.
                # Let's use a dict for word->idx for this precomputation phase.
                pass
                
    # Better precomputation using a dictionary
    word_to_idx = {}
    for i in range(total_words):
        w = idx_to_word(i)
        word_to_idx[tuple(w)] = i
        
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 1 << i
        balls[i] = mask
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_list[pos] = new_val
                nidx = word_to_idx[tuple(w_list)]
                balls[i] |= (1 << nidx)
                w_list[pos] = orig_val

    # 2. Greedy Set Cover with Multi-Start
    best_code = None
    best_size = total_words + 1
    
    # Number of restarts to try
    # Each restart is fast.
    # Let's try 50 restarts.
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        uncovered = (1 << total_words) - 1
        selected = []
        
        while uncovered:
            best_candidate = -1
            best_new_cov = -1
            best_dist_sum = 0 # For tie breaking
            
            # Find candidate with max new coverage
            # To allow for randomness in tie-breaking, we can collect all max candidates
            # or just add noise.
            
            # Optimization: We can stop early if we find a candidate covering all remaining?
            # Unlikely.
            
            for c in range(total_words):
                new_cov = (balls[c] & uncovered).bit_count()
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_candidate = c
                    # Calculate distance sum for tie-breaking later?
                    # Or just store candidate
                elif new_cov == best_new_cov and new_cov > 0:
                    # Tie: prefer one that is "far" from existing selected centers
                    # This helps in packing, which might reduce total count needed?
                    # Actually, for covering, it doesn't strictly help, but it explores different basins.
                    pass
            
            # If we want to be faster, we can iterate only over candidates that might be good?
            # No, 2187 is small.
            
            # Re-evaluate: The loop above finds the max.
            # To introduce randomness for multi-start, we can pick randomly among the top K candidates?
            # Or just randomize the order of iteration?
            
            # Let's implement a randomized greedy:
            # At each step, pick from the top 10 candidates with max coverage uniformly at random.
            
            # But the loop above just finds THE max.
            # Let's modify to collect top candidates.
            
            # Actually, for simplicity and speed, let's just use the standard greedy 
            # but with random tie-breaking by shuffling the candidate list if we detect ties?
            # No, simpler:
            # In the selection loop, if new_cov == best_new_cov, we can keep the first one, 
            # but to vary the path, we can randomize the candidate order in the outer loop?
            # No, the candidate order is fixed 0..N.
            
            # Let's use a "noisy" score:
            # score = new_cov + random.uniform(0, epsilon)
            
            best_candidate = -1
            best_score = -1
            for c in range(total_words):
                new_cov = (balls[c] & uncovered).bit_count()
                if new_cov == 0:
                    continue
                score = new_cov + random.uniform(0, 0.1)
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    
            if best_candidate == -1:
                # Should not happen if uncovered is not 0
                break
                
            selected.append(best_candidate)
            uncovered &= ~balls[best_candidate]
            
        # Post-processing: Remove redundant centers
        # Iterate until no change
        changed = True
        while changed:
            changed = False
            for i in range(len(selected)):
                # Check if selected[i] is redundant
                # i.e., ball[selected[i]] is subset of union of balls of all others
                # union_others = 0
                # for j in range(len(selected)):
                #     if i != j:
                #         union_others |= balls[selected[j]]
                # if balls[selected[i]] & ~union_others == 0:
                #     selected.pop(i)
                #     changed = True
                #     break
                
                # Optimization: 
                # We can maintain a "coverage count" for each word.
                # But re-computing is O(K^2 * BallSize). K is ~200. 200^2 * 15 = 600k ops.
                # This is fine.
                
                union_others = 0
                for j in range(len(selected)):
                    if i != j:
                        union_others |= balls[selected[j]]
                        
                if balls[selected[i]] & ~union_others == 0:
                    selected.pop(i)
                    changed = True
                    break
                    
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected[:]
            
    # Convert indices back to codewords
    result = []
    for idx in best_code:
        w = idx_to_word(idx)
        result.append(w)
        
    return result

def _generic_greedy(q, n, radius, deadline):
    # Fallback for other parameters
    total_words = q ** n
    if total_words > 100000:
         # Too big for bitset, use list of sets or similar
         # But for this task, it's small.
         pass
         
    # Simple greedy
    uncovered = set(range(total_words))
    selected = []
    
    # Precompute balls
    balls = []
    for i in range(total_words):
        w = [0]*n
        tmp = i
        for k in range(n):
            w[k] = tmp % q
            tmp //= q
        w = tuple(w)
        ball = set()
        ball.add(i)
        for pos in range(n):
            for val in range(q):
                if w[pos] == val: continue
                nw = list(w)
                nw[pos] = val
                nidx = 0
                for k in range(n):
                    nidx = nidx * q + nw[k]
                ball.add(nidx)
        balls.append(ball)
        
    while uncovered:
        best_c = -1
        best_cov = -1
        for c in range(total_words):
            cov = len(balls[c] & uncovered)
            if cov > best_cov:
                best_cov = cov
                best_c = c
        if best_c == -1: break
        selected.append(best_c)
        uncovered -= balls[best_c]
        
    result = []
    for idx in selected:
        w = [0]*n
        tmp = idx
        for k in range(n):
            w[k] = tmp % q
            tmp //= q
        result.append(w)
    return result
