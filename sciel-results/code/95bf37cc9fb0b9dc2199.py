<<<<<<< SEARCH
    # Fast path: Precompute balls
    # This is only safe if space_size is small enough to fit in memory and time.
    # 729 * 13 ints is tiny.
    
    for i in range(space_size):
        word = decode(i, n, q)
        neighbors = {i}
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for sym in range(q):
                    if sym == orig: continue
                    new_word = word[:]
                    new_word[pos] = sym
                    neighbors.add(encode(new_word, q))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1+1, n):
                    o1, o2 = word[pos1], word[pos2]
                    for s1 in range(q):
                        if s1 == o1: continue
                        for s2 in range(q):
                            if s2 == o2: continue
                            new_word = word[:]
                            new_word[pos1] = s1
                            new_word[pos2] = s2
                            neighbors.add(encode(new_word, q))
        balls[i] = list(neighbors)
        
    # Convert balls to numpy arrays for faster iteration? 
    # Or just lists. Lists are fine for size 13.
    
    best_code = None
    best_len = float('inf')
    rng = np.random.default_rng(42)
    
    # Number of restarts
    # Each restart takes roughly:
    # Steps ~ 73.
    # Each step: scan 729 words, for each scan 13 neighbors. ~ 9500 ops.
    # Total per restart ~ 700k ops.
    # Python can do ~10M-50M simple ops/sec.
    # So ~10-20 restarts per second.
    # 110s budget -> ~1000-2000 restarts.
    # We can run until deadline.
    
    while time.time() < deadline:
        cover_count = np.zeros(space_size, dtype=np.int32)
        code = []
        
        # Greedy Construction
        # We maintain a list of active (uncovered) words to speed up?
        # No, just scan all.
        
        # Optimization: 
        # Instead of scanning all 729 words every step, we could maintain a list of "candidates".
        # But 729 is small.
        
        while np.any(cover_count == 0):
            if time.time() > deadline:
                break
            
            best_w = -1
            best_score = -1
            # Random tie-breaking
            # We can add a small random noise to break ties
            # Or just pick the first one with max score.
            # To explore different solutions, we should randomize the order of checking
            # or add noise.
            
            # Let's shuffle the range for this iteration? 
            # That's O(N log N) or O(N) extra. 
            # Better: add small random noise to the score.
            
            for w in range(space_size):
                # Count how many neighbors are uncovered (count == 0)
                # np.sum(cover_count[balls[w]] == 0)
                # This creates a temporary array. 
                # Manual loop might be faster for small lists?
                cnt = 0
                for u in balls[w]:
                    if cover_count[u] == 0:
                        cnt += 1
                # Add noise
                score = cnt + rng.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_w = w
            
            if best_w == -1:
                break
                
            code.append(best_w)
            for u in balls[best_w]:
                cover_count[u] += 1
        
        if time.time() > deadline:
            break
            
        current_len = len(code)
        
        # Pruning Phase
        # Try to remove codewords.
        # We need to be careful with cover_count.
        # If we remove c, we decrement cover_count for u in balls[c].
        # We can remove c if min(cover_count[balls[c]]) >= 2.
        
        # We should try removing in a specific order? 
        # Random order is good for restarts.
        
        # Create a list of indices to try removing
        indices = list(range(len(code)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            c = code[idx]
            # Check if c is redundant
            redundant = True
            for u in balls[c]:
                if cover_count[u] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove it
                for u in balls[c]:
                    cover_count[u] -= 1
                code[idx] = None # Mark for removal
                current_len -= 1
        
        # Clean up code list
        final_code = [c for c in code if c is not None]
        
        if len(final_code) < best_len:
            best_len = len(final_code)
            best_code = final_code
            
            # If we hit the theoretical lower bound or known best, we can stop early?
            # Best known is 73. Lower bound 71.
            # If we find 73, it's optimal (or close). 
            # We can continue to see if we find 71 or 72? 
            # Unlikely to find 71 if 73 is the record.
            # But let's keep searching a bit more to be sure, or stop if we hit 73 
            # and have spent significant time?
            # Actually, if we find 73, it's a very good result. 
            # We can break early if we want to save time for other cases? 
            # But the task is fixed. 
            # Let's keep searching to potentially find 72/71, but 73 is the target.
            
            # Optimization: If best_len == 73, we can reduce the number of restarts 
            # or just stop? 
            # The score is minimized. 73 is the known best. 
            # Getting 73 is an improvement over the seed (which likely returns >73 or fails to optimize well).
            # Wait, the seed returns the result of its pruning. 
            # If the seed returns 73, score is 1.0.
            # If we return 73, score is 1.0.
            # We need to return < 73 to beat 1.0? 
            # "Score: the number of codewords, minimized... 1.0 equals the seed program".
            # If the seed program returns 73, then 73 is the baseline.
            # If the seed program returns > 73, then 73 is an improvement.
            # Let's check the seed program behavior.
            # The seed uses a direct sum of 243 words and prunes.
            # It is unlikely to prune 243 down to 73. It might get to 100-150.
            # So 73 is likely a significant improvement (score > 1.0).
            
            # Let's continue searching until deadline to try for 72/71.
=======
    # Fast path: Precompute balls
    # This is only safe if space_size is small enough to fit in memory and time.
    # 729 * 13 ints is tiny.
    
    for i in range(space_size):
        word = decode(i, n, q)
        neighbors = {i}
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for sym in range(q):
                    if sym == orig: continue
                    new_word = word[:]
                    new_word[pos] = sym
                    neighbors.add(encode(new_word, q))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1+1, n):
                    o1, o2 = word[pos1], word[pos2]
                    for s1 in range(q):
                        if s1 == o1: continue
                        for s2 in range(q):
                            if s2 == o2: continue
                            new_word = word[:]
                            new_word[pos1] = s1
                            new_word[pos2] = s2
                            neighbors.add(encode(new_word, q))
        balls[i] = list(neighbors)
    
    # Precompute inverse balls: for each position u, which words w have u in their ball?
    inv_balls = [[] for _ in range(space_size)]
    for w in range(space_size):
        for u in balls[w]:
            inv_balls[u].append(w)
    
    best_code = None
    best_len = float('inf')
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Use a different random seed for each restart to explore different solutions
        restart_seed = int(time.time() * 1000) % (2**32)
        rng_restart = np.random.default_rng(restart_seed)
        
        cover_count = np.zeros(space_size, dtype=np.int32)
        # uncovered_neighbors[w] = number of neighbors of w that are currently uncovered
        uncovered_neighbors = np.array([len(balls[w]) for w in range(space_size)], dtype=np.int32)
        code = []
        uncovered_count = space_size
        
        # Greedy Construction
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            # Find the word with the most uncovered neighbors
            # Add small random perturbation for tie-breaking
            perturbation = rng_restart.random(size=space_size) * 0.01
            scores = uncovered_neighbors.astype(np.float64) + perturbation
            best_w = int(np.argmax(scores))
            
            code.append(best_w)
            # Update cover_count and uncovered_neighbors
            for u in balls[best_w]:
                if cover_count[u] == 0:
                    uncovered_count -= 1
                cover_count[u] += 1
                # For all words w that have u in their ball, decrement uncovered_neighbors[w]
                for w in inv_balls[u]:
                    uncovered_neighbors[w] -= 1
        
        if time.time() > deadline:
            break
            
        current_len = len(code)
        
        # Pruning Phase with iterative improvement
        indices = list(range(len(code)))
        rng_restart.shuffle(indices)
        
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            for idx in indices:
                if time.time() > deadline:
                    break
                
                c = code[idx]
                if c is None:
                    continue
                
                # Check if c is redundant: all its neighbors have cover_count >= 2
                redundant = True
                for u in balls[c]:
                    if cover_count[u] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove it
                    for u in balls[c]:
                        cover_count[u] -= 1
                    code[idx] = None
                    current_len -= 1
                    removed_any = True
            
            # Re-shuffle remaining indices for next pass
            indices = [i for i in indices if code[i] is not None]
            rng_restart.shuffle(indices)
        
        # Clean up code list
        final_code = [c for c in code if c is not None]
        
        if len(final_code) < best_len:
            best_len = len(final_code)
            best_code = final_code
>>>>>>> REPLACE
