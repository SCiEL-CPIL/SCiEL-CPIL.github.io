import itertools
import random
import time
import math
import numpy as np

def construct(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls as list of numpy arrays for fast vectorized updates
    balls_np = []
    for w in all_words:
        ball = [word_to_idx[w]]
        if radius == 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    ball.append(word_to_idx[tuple(new_w)])
        balls_np.append(np.array(ball, dtype=np.int32))
        
    # Initial Greedy Code
    def get_initial_code():
        uncovered = np.ones(num_words, dtype=bool)
        code_indices = []
        # To speed up greedy, we can just pick random or use a simple heuristic
        # Given the size, a standard greedy is fine for initialization, 
        # but we want it to be fast.
        # Let's use a randomized greedy to avoid getting stuck in a specific local minimum immediately.
        # Or just use the standard "max coverage" greedy.
        # Standard greedy:
        while np.any(uncovered):
            best_c = -1
            best_cov = -1
            # Optimization: We can sample a subset of candidates if it's too slow, 
            # but for 2187 words, full scan is ~2187 * (avg ball size) operations.
            # Avg ball size is 1 + n*(q-1) = 1 + 7*2 = 15.
            # 2187 * 15 ~ 32k operations per step.
            # Greedy takes ~100-200 steps. 32k * 200 = 6.4M ops. This is fast enough in Python if optimized.
            # However, doing it in a Python loop is slow. 
            # Let's stick to a simpler initialization: Random set + local search?
            # No, let's use a vectorized greedy step if possible, or just a fast Python loop.
            
            # Fast Python Loop for Greedy:
            # We need to count how many uncovered words are in each ball.
            # We can iterate through all words and check if they are uncovered, 
            # but that's O(N^2).
            
            # Let's just use a random initialization for speed and let SA do the work,
            # or a very fast greedy.
            # Given time limits, a simple random initialization of size ~200 might work well with SA.
            # Let's try a random initialization with size estimated by lower bound * 1.2.
            # Lower bound is 156. Let's start with 200.
            # Actually, a bad start might take too long to fix.
            # Let's use a "Greedy with Random Ties" or similar.
            
            # For now, let's keep the original greedy but optimize it slightly or replace with random if it's too slow.
            # The original greedy was:
            # for c in range(num_words): ...
            # This is O(N * BallSize) per step.
            
            # Let's implement a faster greedy using numpy for the coverage count?
            # Hard to do incremental.
            
            # Let's just use a random initialization of size 190. SA is good at fixing things.
            # If we start with 190 random words, we likely have some uncovered, but SA will add/remove.
            # To ensure validity quickly, we can add random words until valid.
            
            code_set = set()
            cnt = np.zeros(num_words, dtype=np.int32)
            # Start with a random subset
            target_size = 190
            while len(code_set) < target_size:
                c = random.randint(0, num_words - 1)
                if c not in code_set:
                    code_set.add(c)
                    cnt[balls_np[c]] += 1
            
            # Add more until valid
            while np.any(cnt == 0):
                # Find an uncovered word
                unc_idx = np.where(cnt == 0)[0]
                # Pick a random uncovered word's neighbor? Or just a random word?
                # Let's pick a random word from the whole space that covers many uncovered.
                # Approximation: Pick a random word, if it covers any uncovered, add it.
                # Better: Pick the word with max coverage of uncovered.
                # Let's do a quick greedy step here to fix validity.
                best_c = -1
                best_cov = -1
                for c in range(num_words):
                    if c in code_set:
                        continue
                    # Count uncovered in ball
                    # This is still O(N * BallSize)
                    # Let's just pick a random candidate
                    c = random.randint(0, num_words - 1)
                    if c in code_set:
                        continue
                    # Check if it covers any uncovered
                    # This is inefficient.
                    pass
                
                # Let's revert to a proper greedy for initialization if random fails.
                # But for the sake of code simplicity and speed, let's use a 
                # "Greedy Add" loop.
                
                # Find word with max coverage of currently uncovered
                # We can approximate by checking a random sample of words
                best_c = -1
                best_cov = -1
                for _ in range(50): # Sample 50 random words
                    c = random.randint(0, num_words - 1)
                    if c in code_set:
                        continue
                    cov = np.sum(cnt[balls_np[c]] == 0)
                    if cov > best_cov:
                        best_cov = cov
                        best_c = c
                
                if best_c != -1 and best_cov > 0:
                    code_set.add(best_c)
                    cnt[balls_np[best_c]] += 1
                else:
                    # Fallback: add random
                    c = random.randint(0, num_words - 1)
                    while c in code_set:
                        c = random.randint(0, num_words - 1)
                    code_set.add(c)
                    cnt[balls_np[c]] += 1

            return code_set, cnt

    current_code_set, current_cnt = get_initial_code()
    current_in_code = np.zeros(num_words, dtype=bool)
    for c in current_code_set:
        current_in_code[c] = True
        
    current_size = len(current_code_set)
    current_unc = np.sum(current_cnt == 0)
    
    best_code_set = set(current_code_set)
    best_cnt = current_cnt.copy()
    best_size = current_size
    best_unc = current_unc
    
    def objective(size, unc):
        return size + 1000 * unc

    current_obj = objective(current_size, current_unc)
    best_obj = current_obj
    
    start_time = time.time()
    time_limit = 110.0
    
    # SA Parameters
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.999 # Slower cooling to explore more
    
    T = T_start
    
    while time.time() - start_time < time_limit:
        move_type = random.random()
        
        new_size = current_size
        new_unc = current_unc
        delta = 0
        accepted = False
        c_add = -1
        c_remove = -1
        
        if move_type < 0.4:
            # Remove
            if current_size == 0:
                continue
            # Pick random from current code
            code_indices = np.nonzero(current_in_code)[0]
            c_remove = code_indices[random.randint(0, len(code_indices) - 1)]
            
            # Check if valid removal
            # Valid if all neighbors are covered by at least 2 words (including this one)
            # So cnt[neighbor] >= 2 for all neighbors
            if np.all(current_cnt[balls_np[c_remove]] >= 2):
                # Apply
                current_cnt[balls_np[c_remove]] -= 1
                current_in_code[c_remove] = False
                new_size = current_size - 1
                # Unc doesn't change because all neighbors were >= 2, so they become >= 1
                new_unc = current_unc
                accepted = True
                delta = objective(new_size, new_unc) - current_obj
                
        elif move_type < 0.7:
            # Add
            # Pick random not in code
            # Efficient: pick random, if in code, retry (few retries expected if code is small)
            c_add = random.randint(0, num_words - 1)
            if current_in_code[c_add]:
                continue
            
            # Apply
            current_cnt[balls_np[c_add]] += 1
            current_in_code[c_add] = True
            new_size = current_size + 1
            # Unc decreases by number of neighbors that were 0
            new_unc = current_unc - np.sum(current_cnt[balls_np[c_add]] == 1) # Wait, if it was 0, it becomes 1.
            # Actually, we already updated cnt. So we need to know how many were 0 before.
            # Let's recalculate unc? No, that's slow.
            # Let's store old_cnt for the ball?
            # Better: 
            # old_ball_cnt = current_cnt[balls_np[c_add]]
            # new_unc = current_unc - np.sum(old_ball_cnt == 0)
            # We didn't store old. Let's just recalculate unc if it's cheap?
            # np.sum(current_cnt == 0) is O(N). N=2187. This is very fast in numpy.
            new_unc = np.sum(current_cnt == 0)
            accepted = True
            delta = objective(new_size, new_unc) - current_obj
            
        else:
            # Swap
            if current_size == 0:
                continue
            code_indices = np.nonzero(current_in_code)[0]
            c_remove = code_indices[random.randint(0, len(code_indices) - 1)]
            
            c_add = random.randint(0, num_words - 1)
            if current_in_code[c_add]:
                continue
            
            # Apply Swap
            # Remove c_remove
            current_cnt[balls_np[c_remove]] -= 1
            current_in_code[c_remove] = False
            
            # Add c_add
            current_cnt[balls_np[c_add]] += 1
            current_in_code[c_add] = True
            
            new_size = current_size # Size unchanged
            new_unc = np.sum(current_cnt == 0)
            accepted = True
            delta = objective(new_size, new_unc) - current_obj
            
        if not accepted:
            continue
            
        # Acceptance
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_obj = objective(new_size, new_unc)
            current_size = new_size
            current_unc = new_unc
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_size = current_size
                best_unc = current_unc
                best_code_set = set(np.nonzero(current_in_code)[0])
                best_cnt = current_cnt.copy()
        else:
            # Reject: Undo
            if move_type < 0.4:
                # Undo Remove (Add back)
                current_cnt[balls_np[c_remove]] += 1
                current_in_code[c_remove] = True
            elif move_type < 0.7:
                # Undo Add (Remove)
                current_cnt[balls_np[c_add]] -= 1
                current_in_code[c_add] = False
            else:
                # Undo Swap
                # Remove c_add
                current_cnt[balls_np[c_add]] -= 1
                current_in_code[c_add] = False
                # Add c_remove
                current_cnt[balls_np[c_remove]] += 1
                current_in_code[c_remove] = True
            
            # Restore current state vars (they are effectively restored by the undo ops on arrays, 
            # but we need to ensure current_size/unc/obj are consistent. 
            # Since we didn't update them on rejection, they are still the old values. 
            # But wait, we modified current_cnt and current_in_code. 
            # The undo ops restore them. 
            # So current_size, current_unc, current_obj are still the pre-move values. Correct.
            
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    return [list(all_words[c]) for c in best_code_set]
