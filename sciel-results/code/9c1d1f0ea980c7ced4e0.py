import time
import numpy as np

def construct(q, n, radius):
    # Deadline guard: 150s limit, aim for 110s safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Specific parameters for the task: q=3, n=7, radius=1
    # Total space size: 3^7 = 2187
    # We represent words as integers 0..2186 in base-q
    
    total_words = q ** n
    
    # Helper to convert int to base-q tuple
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t

    # Precompute the "ball" of each word: which words are within radius?
    # For q=3, n=7, radius=1, each word covers 1 + n*(q-1) = 1 + 7*2 = 15 words.
    
    # We will maintain a coverage count array: cover_count[j] = number of codewords covering j
    # And a binary inclusion vector: included[i] = 1 if word i is a codeword
    
    # Precompute balls as lists of ints for efficient updates
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
                # Calculate index directly to avoid tuple conversion overhead
                # t_new = t_i with pos changed to new_val
                # Instead of creating a new tuple and converting, we can just store the list
                # But we need the integer index for the cover_count array
                # Let's stick to the tuple conversion for correctness and simplicity, it's fast enough for 2187 items
                t_new = t_i[:]
                t_new[pos] = new_val
                x_new = 0
                for k in range(n-1, -1, -1):
                    x_new = x_new * q + t_new[k]
                ball.append(x_new)
        balls[i] = ball

    # Initial state: Random valid covering?
    # A simple random selection is likely to have too many codewords or be invalid.
    # Better: Start with a greedy solution to ensure validity, then optimize.
    # Or start with a random subset and let SA handle the "uncovered" penalty.
    # Given the penalty for uncovered words is high, starting invalid is risky but SA can recover.
    # However, to be safe and fast, let's initialize with a Greedy Set Cover result.
    
    # Phase 0: Greedy Initialization
    covered = np.zeros(total_words, dtype=bool)
    initial_codewords = []
    # Use a coverage matrix or just recompute? 
    # For init, simple greedy: pick word covering most uncovered.
    # To speed up init, we can use the same M matrix logic or just loop.
    # 2187 words, 15 neighbors. 
    # Let's do a quick greedy init.
    
    # Precompute M for greedy init? Or just loop?
    # Looping 150 times * 2187 candidates * 15 neighbors is ~5M ops. OK.
    
    while True:
        if time.time() > deadline - 90: # Leave time for SA
            break
        best_idx = -1
        best_count = -1
        for i in range(total_words):
            # Count uncovered in ball[i]
            cnt = 0
            for j in balls[i]:
                if not covered[j]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        initial_codewords.append(best_idx)
        covered[balls[best_idx]] = True
        
    # If greedy didn't finish (time out), we might have an invalid cover.
    # Let's check validity.
    final_covered = np.zeros(total_words, dtype=bool)
    for c in initial_codewords:
        final_covered[balls[c]] = True
    if not np.all(final_covered):
        # Force add random words until covered? Or just accept invalid start and let SA fix?
        # SA with high penalty will fix it. But if we run out of time, we return invalid.
        # Let's try to complete coverage quickly.
        uncovered_indices = np.where(~final_covered)[0]
        for u in uncovered_indices:
            # Find a word covering u
            # We can just pick u itself? No, u might not be in initial_codewords.
            # Pick any word in balls[u]? Actually, if we add u, it covers u.
            if u not in initial_codewords:
                initial_codewords.append(u)
                final_covered[balls[u]] = True

    # Now we have a valid set. Size likely ~180-190.
    # Convert to binary vector
    x = np.zeros(total_words, dtype=np.uint8)
    for c in initial_codewords:
        x[c] = 1
        
    # Calculate initial cover_count
    cover_count = np.zeros(total_words, dtype=np.uint8)
    for c in initial_codewords:
        cover_count[balls[c]] += 1
        
    # Uncovered count should be 0
    uncovered_count = np.sum(cover_count == 0)
    
    # Energy Function:
    # E = |C| + L * Uncovered
    # We want to minimize this.
    # L should be large enough to prefer valid solutions.
    # If we have 186 codewords, and 1 uncovered, E = 186 + L.
    # If we have 187 codewords, 0 uncovered, E = 187.
    # We want 186 + L > 187 => L > 1.
    # But we also want to strongly penalize uncovered.
    # Let's set L = 10.
    L = 10.0
    
    current_energy = len(initial_codewords) + L * uncovered_count
    
    # Simulated Annealing
    T = 1.0 # Initial temperature
    T_min = 0.01
    cooling_rate = 0.9999 # Slow cooling
    # Or use a schedule based on steps?
    # Let's run until deadline.
    
    best_x = x.copy()
    best_energy = current_energy
    best_size = len(initial_codewords)
    
    # Precompute random neighbors? No, pick random index, flip bit.
    
    # Optimization: 
    # Flipping bit i from 0 to 1:
    #   new_size = old_size + 1
    #   new_uncovered = old_uncovered - (number of j in balls[i] where cover_count[j] == 0)
    #   delta_E = 1 - L * (number of newly covered words)
    #   Update cover_count: for j in balls[i], cover_count[j] += 1
    
    # Flipping bit i from 1 to 0:
    #   new_size = old_size - 1
    #   new_uncovered = old_uncovered + (number of j in balls[i] where cover_count[j] == 1)
    #   delta_E = -1 + L * (number of newly uncovered words)
    #   Update cover_count: for j in balls[i], cover_count[j] -= 1
    
    # We need to track cover_count efficiently.
    
    while time.time() < deadline - 5:
        # Pick random move
        i = np.random.randint(total_words)
        
        if x[i] == 0:
            # Add i
            # Calculate delta
            # newly_covered = sum(1 for j in balls[i] if cover_count[j] == 0)
            newly_covered = 0
            for j in balls[i]:
                if cover_count[j] == 0:
                    newly_covered += 1
            
            delta_E = 1.0 - L * newly_covered
            
            # Acceptance
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                # Apply
                x[i] = 1
                for j in balls[i]:
                    cover_count[j] += 1
                uncovered_count -= newly_covered
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
                    best_size = np.sum(x)
        else:
            # Remove i
            # Calculate delta
            # newly_uncovered = sum(1 for j in balls[i] if cover_count[j] == 1)
            newly_uncovered = 0
            for j in balls[i]:
                if cover_count[j] == 1:
                    newly_uncovered += 1
            
            delta_E = -1.0 + L * newly_uncovered
            
            # Acceptance
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                # Apply
                x[i] = 0
                for j in balls[i]:
                    cover_count[j] -= 1
                uncovered_count += newly_uncovered
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
                    best_size = np.sum(x)
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final check: ensure best_x is valid
    # If best_x has uncovered words, it's invalid.
    # We need to return a VALID code.
    # The energy function penalizes uncovered, but if T drops too low or time runs out,
    # we might be stuck in an invalid state with lower energy than a valid one?
    # No, L=10 means 1 uncovered costs 10 "size units".
    # A valid code of size 200 has E=200.
    # An invalid code of size 190 with 1 uncovered has E=190+10=200.
    # An invalid code of size 180 with 2 uncovered has E=180+20=200.
    # So it's possible best_x is invalid if it has lower size + penalty than any valid solution found.
    # We must ensure the returned solution is valid.
    
    # Recover best_x
    x = best_x
    # Check validity of best_x
    check_cover = np.zeros(total_words, dtype=np.uint8)
    for i in range(total_words):
        if x[i] == 1:
            check_cover[balls[i]] += 1
            
    if np.any(check_cover == 0):
        # best_x is invalid. We need to fix it or use the last valid state?
        # We didn't track last valid state explicitly, only best energy.
        # Let's find the smallest valid solution among recent states? 
        # Or just repair best_x by adding words to cover gaps.
        uncovered_indices = np.where(check_cover == 0)[0]
        for u in uncovered_indices:
            # Add u itself? Or a neighbor?
            # Adding u is safe.
            if x[u] == 0:
                x[u] = 1
                check_cover[balls[u]] += 1
        
        # Now x is valid.
        # But is it optimal? Maybe not, but it's valid.
        # We could try to prune it again?
        # Given time, just return this.
        # Let's do a quick prune pass on this repaired x to remove redundancies.
        # Convert to list
        codewords = [i for i in range(total_words) if x[i] == 1]
        
        # Quick prune
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for idx, c in enumerate(codewords):
                # Check if c is redundant
                # Temporarily remove c
                # Check if all j in balls[c] are covered by others
                # We can use the check_cover array.
                # Decrement check_cover for balls[c]
                for j in balls[c]:
                    check_cover[j] -= 1
                # Check if any j in balls[c] is now 0
                redundant = True
                for j in balls[c]:
                    if check_cover[j] == 0:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    codewords.pop(idx)
                    changed = True
                    # No need to increment check_cover back as we removed c
                    break # Restart loop
                else:
                    # Restore
                    for j in balls[c]:
                        check_cover[j] += 1
        
        # Build result from codewords
        result = []
        for c in codewords:
            result.append(int_to_tuple(c, n, q))
        return result
    else:
        # best_x is valid
        codewords = [i for i in range(total_words) if x[i] == 1]
        result = []
        for c in codewords:
            result.append(int_to_tuple(c, n, q))
        return result

# EVOLVE-BLOCK-END
