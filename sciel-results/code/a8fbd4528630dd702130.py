import itertools
import time
import random

def construct(q, n, radius):
    # This implementation is optimized for the specific case q=3, n=6, radius=1
    # but will fall back to a general greedy approach for other parameters if needed.
    # For q=3, n=6, radius=1, we aim for 73.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: Simple Greedy
        # Generate all words
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute balls
        balls = []
        for w in all_words:
            ball = set()
            ball.add(w)
            for i in range(n):
                for val in range(q):
                    if val == w[i]: continue
                    new_w = list(w)
                    new_w[i] = val
                    ball.add(tuple(new_w))
            balls.append(ball)
            
        # Greedy Set Cover
        uncovered = set(all_words)
        code = []
        candidates = list(range(len(all_words)))
        
        while uncovered:
            best_idx = -1
            best_cov = 0
            # Randomize candidates for diversity in restarts
            random.shuffle(candidates)
            for idx in candidates:
                w = all_words[idx]
                # Check if w is already in code? 
                # In greedy set cover for covering codes, we don't usually exclude previously picked 
                # unless we want distinct codewords, which we do. But picking the same one again 
                # gives 0 new coverage.
                cov = len(uncovered & balls[idx])
                if cov > best_cov:
                    best_cov = cov
                    best_idx = idx
            if best_cov == 0:
                break
            code.append(all_words[best_idx])
            uncovered -= balls[best_idx]
            
        return code

    # Specific optimized path for q=3, n=6, radius=1
    # Total words = 3^6 = 729
    N = 729
    
    # Precompute ID for each word
    # id = w0*3^5 + w1*3^4 + w2*3^3 + w3*3^2 + w4*3 + w5
    id_to_word = []
    for w in itertools.product(range(3), repeat=6):
        id_to_word.append(w)
        
    # Precompute ball masks
    # A ball of radius 1 around (w0,...,w5) contains the word itself and words differing in exactly one position.
    # Total size = 1 + 6*2 = 13 words.
    ball_masks = [0] * N
    
    # Helper to get ID from word tuple
    def get_id(w):
        return w[0]*3**5 + w[1]*3**4 + w[2]*3**3 + w[3]*3**2 + w[4]*3 + w[5]
        
    # Precompute powers of 3 for speed
    p3 = [1, 3, 9, 27, 81, 243, 729]
    
    for i in range(N):
        w = id_to_word[i]
        mask = 1 << i
        for pos in range(6):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = get_id(tuple(new_w))
                mask |= (1 << j)
        ball_masks[i] = mask
        
    # Greedy Set Cover with Bitmasks
    # We want to minimize the number of codewords.
    # We will run multiple random restarts of the greedy algorithm.
    # In each restart, we pick the candidate with max coverage. 
    # To introduce randomness, we can shuffle the list of candidates before evaluating, 
    # so that in case of ties, different candidates are picked.
    
    best_code_size = float('inf')
    best_code_ids = []
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Number of restarts is dynamic based on time
    restarts = 0
    
    while time.time() < deadline:
        uncovered = (1 << N) - 1
        code_ids = []
        
        # We iterate until uncovered is 0
        # Optimization: Keep a list of active candidates? 
        # Since N=729, scanning all 729 each step is 729 * (steps) operations.
        # Steps will be around 73. 729 * 73 ~ 53k operations per restart.
        # This is very fast. We can do many restarts.
        
        # To break ties randomly, we can just shuffle the range once at the start of the restart?
        # No, the "best" changes as uncovered changes.
        # Better: When finding max, if there are ties, pick randomly.
        
        while uncovered:
            max_cov = 0
            best_id = -1
            tied_ids = []
            
            # Iterate through all possible codewords
            for i in range(N):
                # If we already picked i, coverage is 0 (since bits are cleared)
                # Actually, if we picked i, ball_masks[i] bits are cleared from uncovered.
                # So coverage will be 0.
                cov = (uncovered & ball_masks[i]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_id = i
                    tied_ids = [i]
                elif cov == max_cov and cov > 0:
                    tied_ids.append(i)
            
            if max_cov == 0:
                # Should not happen if we started with full coverage and balls cover everything
                break
                
            # Pick randomly from tied_ids to diversify
            chosen = random.choice(tied_ids)
            code_ids.append(chosen)
            uncovered &= ~ball_masks[chosen]
            
        # End while uncovered
        
        if len(code_ids) < best_code_size:
            best_code_size = len(code_ids)
            best_code_ids = code_ids[:]
            # If we hit the theoretical lower bound or known optimum, we can stop early?
            # Lower bound is 71, known is 73. If we find 71 or 72, it's a breakthrough.
            # If we find 73, it's the best known.
            if best_code_size <= 73:
                # We can try to run a few more to see if we get 71/72, 
                # but 73 is very stable.
                # Let's continue for a bit to see if we can improve, but 73 is likely the min.
                pass
                
        restarts += 1
        # If we have done many restarts and haven't improved, we might be stuck at 73.
        # We can break if we've run, say, 100 restarts and best is 73.
        if best_code_size == 73 and restarts > 50:
            break
        if best_code_size <= 71:
            break

    # Convert best_code_ids to list of lists of ints
    result = [list(id_to_word[i]) for i in best_code_ids]
    
    return result
