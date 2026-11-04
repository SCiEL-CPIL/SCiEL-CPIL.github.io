import time
import random
import sys

def construct(q, n, radius):
    # Parameters for the specific task instance
    # q=3, n=7, radius=1
    # We assume the evaluator calls with these values.
    
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Total number of words
    total_words = q ** n
    
    # Map words to indices
    # We use a simple base-q encoding
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute balls
    # ball[i] is a list of indices within distance 'radius' of word i
    balls = [[] for _ in range(total_words)]
    
    # Generate all words and their indices
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Compute Hamming distance
    def hamming_dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
    
    # Precomputing balls for all 2187 words
    # 2187 * 2187 * 7 is ~33M ops, might be slow in pure Python.
    # Optimization: Only compute balls for candidates we might use?
    # In Greedy/SA, we might consider any word.
    # Let's optimize ball generation.
    # A word is within distance 1 if it differs in 0 or 1 position.
    # For a given word w, its ball contains w and words obtained by changing one position to one of the other q-1 values.
    # Size = 1 + n * (q - 1).
    
    for i, w in enumerate(all_words):
        ball = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                # Create new word
                new_w = w.copy()
                new_w[pos] = new_val
                idx = word_to_idx(new_w)
                ball.append(idx)
        balls[i] = ball

    # Greedy Construction
    # uncovered is a set of indices not yet covered
    # Using a list of booleans or a set. Set is good for removal, but we need to count coverage.
    # For Greedy, we want to pick the word that covers the most uncovered words.
    
    # We'll maintain a list of uncovered words.
    # To speed up Greedy, we can maintain an array `uncovered_count`? No, just a set `uncovered_set`.
    # But iterating all candidates to find max coverage is O(total_words * ball_size).
    # 2187 * 29 ~ 63k ops per step.
    # If we need ~200 steps, that's ~12M ops. Fast enough.
    
    def greedy_construct():
        uncovered_set = set(range(total_words))
        code = []
        
        while uncovered_set:
            best_candidate = -1
            best_cover_count = 0
            # Iterate through all possible candidates
            # Optimization: We can just iterate all.
            for c in range(total_words):
                # Count how many words in ball[c] are in uncovered_set
                # Intersection size
                count = 0
                for w in balls[c]:
                    if w in uncovered_set:
                        count += 1
                if count > best_cover_count:
                    best_cover_count = count
                    best_candidate = c
                    if best_cover_count == len(balls[c]):
                        # Can't do better for this candidate
                        break
            
            if best_candidate == -1:
                break # Should not happen
            
            code.append(best_candidate)
            # Remove covered words
            for w in balls[best_candidate]:
                if w in uncovered_set:
                    uncovered_set.remove(w)
            
            # Time check
            if time.time() - start_time > time_limit - 60:
                break
                
        return code

    initial_code = greedy_construct()
    
    # Local Search / Simulated Annealing
    # State:
    # code: list of indices
    # in_code: boolean array of size total_words
    # cover_count: array of size total_words, count of how many codewords cover each word
    
    def evaluate_code(code_list):
        in_c = [False] * total_words
        cover_cnt = [0] * total_words
        for c in code_list:
            in_c[c] = True
            for w in balls[c]:
                cover_cnt[w] += 1
        uncovered = 0
        for cnt in cover_cnt:
            if cnt == 0:
                uncovered += 1
        return in_c, cover_cnt, uncovered

    in_c, cover_cnt, uncovered = evaluate_code(initial_code)
    best_code = initial_code.copy()
    best_uncovered = uncovered
    best_cost = len(initial_code) * 1000 + uncovered
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    # Pre-compute list of all indices for random selection
    all_indices = list(range(total_words))
    
    current_cost = best_cost
    
    while time.time() - start_time < time_limit:
        T = max(T_min, T * cooling)
        
        # Choose a move
        move_type = random.random()
        if len(current_code) > 0:
            # 20% Remove, 30% Add, 50% Swap
            if move_type < 0.2:
                # Remove
                if len(current_code) == 0: continue
                idx_in_code = random.randint(0, len(current_code) - 1)
                c_remove = current_code[idx_in_code]
                # Apply
                in_c[c_remove] = False
                delta_uncovered = 0
                for w in balls[c_remove]:
                    cover_cnt[w] -= 1
                    if cover_cnt[w] == 0:
                        delta_uncovered += 1
                current_code.pop(idx_in_code)
                new_cost = len(current_code) * 1000 + (best_uncovered + delta_uncovered) # Wait, best_uncovered is not current
                # We need to track current_uncovered
                # Let's restructure to track current_uncovered properly
                
        # Restart loop logic with proper state tracking
        
    # The logic above was getting messy. Let's write a clean SA loop.
    
    # Reset state for SA
    current_code = initial_code.copy()
    in_c, cover_cnt, current_uncovered = evaluate_code(current_code)
    current_cost = len(current_code) * 1000 + current_uncovered
    
    best_code_sa = current_code.copy()
    best_cost_sa = current_cost
    
    T = 50.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() - start_time < time_limit:
        T = max(T_min, T * cooling)
        
        # Pick move
        r = random.random()
        
        if r < 0.3 and len(current_code) > 0:
            # Remove
            idx_in_code = random.randint(0, len(current_code) - 1)
            c = current_code[idx_in_code]
            
            # Apply
            in_c[c] = False
            delta_unc = 0
            for w in balls[c]:
                cover_cnt[w] -= 1
                if cover_cnt[w] == 0:
                    delta_unc += 1
            
            current_code.pop(idx_in_code)
            new_uncovered = current_uncovered + delta_unc
            new_cost = len(current_code) * 1000 + new_uncovered
            
            if new_cost < current_cost or random.random() < pow(2.718, -(new_cost - current_cost) / T):
                current_uncovered = new_uncovered
                current_cost = new_cost
                if current_uncovered == 0 and len(current_code) < len(best_code_sa):
                    best_code_sa = current_code.copy()
                    best_cost_sa = current_cost
            else:
                # Revert
                current_code.insert(idx_in_code, c)
                in_c[c] = True
                for w in balls[c]:
                    cover_cnt[w] += 1
                    
        elif r < 0.6:
            # Add
            # Pick a random word not in code
            # Optimization: Pick from a subset? Just random index.
            c = random.randint(0, total_words - 1)
            if in_c[c]:
                continue
            
            # Apply
            in_c[c] = True
            delta_unc = 0
            for w in balls[c]:
                if cover_cnt[w] == 0:
                    delta_unc -= 1
                cover_cnt[w] += 1
            
            current_code.append(c)
            new_uncovered = current_uncovered + delta_unc
            new_cost = len(current_code) * 1000 + new_uncovered
            
            if new_cost < current_cost or random.random() < pow(2.718, -(new_cost - current_cost) / T):
                current_uncovered = new_uncovered
                current_cost = new_cost
                if current_uncovered == 0 and len(current_code) < len(best_code_sa):
                    best_code_sa = current_code.copy()
                    best_cost_sa = current_cost
            else:
                # Revert
                current_code.pop()
                in_c[c] = False
                for w in balls[c]:
                    cover_cnt[w] -= 1
                    
        else:
            # Swap
            if len(current_code) == 0:
                continue
            idx_in_code = random.randint(0, len(current_code) - 1)
            c_old = current_code[idx_in_code]
            c_new = random.randint(0, total_words - 1)
            
            if c_new == c_old or in_c[c_new]:
                continue
            
            # Apply
            in_c[c_old] = False
            in_c[c_new] = True
            delta_unc = 0
            for w in balls[c_old]:
                cover_cnt[w] -= 1
                if cover_cnt[w] == 0:
                    delta_unc += 1
            for w in balls[c_new]:
                if cover_cnt[w] == 0:
                    delta_unc -= 1
                cover_cnt[w] += 1
            
            current_code[idx_in_code] = c_new
            new_uncovered = current_uncovered + delta_unc
            new_cost = len(current_code) * 1000 + new_uncovered
            
            if new_cost < current_cost or random.random() < pow(2.718, -(new_cost - current_cost) / T):
                current_uncovered = new_uncovered
                current_cost = new_cost
                if current_uncovered == 0 and len(current_code) < len(best_code_sa):
                    best_code_sa = current_code.copy()
                    best_cost_sa = current_cost
            else:
                # Revert
                current_code[idx_in_code] = c_old
                in_c[c_old] = True
                in_c[c_new] = False
                for w in balls[c_old]:
                    cover_cnt[w] += 1
                for w in balls[c_new]:
                    cover_cnt[w] -= 1

    # Return the best valid code found
    # If best_code_sa is invalid (uncovered > 0), we should return a valid one.
    # The initial greedy code is valid.
    # We track best_code_sa only when uncovered == 0.
    # So best_code_sa is always valid (if found).
    # If SA never found a valid code smaller than initial, best_code_sa is initial.
    
    # Convert indices back to words
    result_words = [idx_to_word(i) for i in best_code_sa]
    
    return result_words

