import time
import numpy as np
import itertools

def _construct_ternary_6_1():
    start_time = time.perf_counter()
    deadline = start_time + 110.0 # Safety margin
    
    q = 3
    n = 6
    total = q ** n
    
    # Precompute balls as lists of indices
    balls = []
    # Precompute coverers: for each word x, list of codeword indices w that cover x
    coverers = [[] for _ in range(total)]
    
    for idx in range(total):
        # Convert idx to word
        w = [0] * n
        temp = idx
        for i in range(n-1, -1, -1):
            w[i] = temp % q
            temp //= q
        
        ball = [idx]
        # Generate neighbors
        for i in range(n):
            old_val = w[i]
            for new_val in range(q):
                if new_val != old_val:
                    power = q ** (n - 1 - i)
                    new_idx = idx + (new_val - old_val) * power
                    ball.append(new_idx)
                    coverers[new_idx].append(idx)
        balls.append(ball)
        coverers[idx].append(idx) # Self coverage

    # Greedy Initialization
    uncovered = np.ones(total, dtype=bool)
    code = []
    cover_count = np.zeros(total, dtype=np.int32)
    
    # Fast greedy: pick codeword covering most uncovered
    # We can optimize this, but 729 is small.
    while np.any(uncovered):
        if time.perf_counter() > deadline:
            break
        best_idx = -1
        best_gain = 0
        # Iterate all possible codewords
        for i in range(total):
            # Count uncovered in ball[i]
            # Using numpy boolean indexing
            gain = np.count_nonzero(uncovered[balls[i]])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        code.append(best_idx)
        cover_count[balls[best_idx]] += 1
        uncovered[balls[best_idx]] = False

    # WalkSAT-style Local Search
    # Objective: Minimize |code|
    # State: boolean array 'in_code' of size total
    # We start with the greedy solution.
    in_code = np.zeros(total, dtype=bool)
    for c in code:
        in_code[c] = True
        
    current_size = len(code)
    
    # Precompute for faster updates?
    # Let's use a randomized local search that tries to remove or swap codewords.
    
    best_size = current_size
    best_in_code = in_code.copy()
    
    while time.perf_counter() < deadline:
        # Try random removals
        # Pick a random codeword in code, remove it, check if valid.
        # If invalid, try to fix by adding a codeword that covers the most newly uncovered words.
        
        if not np.any(in_code):
            break
            
        # Randomly select a codeword to potentially remove
        current_codewords = np.where(in_code)[0]
        if len(current_codewords) == 0:
            break
            
        c_to_remove = current_codewords[np.random.randint(len(current_codewords))]
        
        # Remove it
        in_code[c_to_remove] = False
        # Update cover_count
        cover_count[balls[c_to_remove]] -= 1
        
        # Check if any word is uncovered (cover_count == 0)
        uncovered_words = np.where(cover_count == 0)[0]
        
        if len(uncovered_words) == 0:
            # Success! We reduced size.
            current_size -= 1
            if current_size < best_size:
                best_size = current_size
                best_in_code = in_code.copy()
        else:
            # Failure. We need to add a codeword to cover uncovered_words.
            # Find the codeword w that covers the most uncovered_words.
            best_add = -1
            max_cover = 0
            for w in range(total):
                if in_code[w]:
                    continue
                # Count how many uncovered_words are in balls[w]
                # uncovered_words is an array of indices
                # We can check intersection
                # Since uncovered_words is small usually, and balls[w] is size 11
                # It's faster to iterate uncovered_words and check if they are in balls[w]?
                # Or iterate balls[w] and check if in uncovered_words?
                # balls[w] has 11 elements. uncovered_words might be small.
                # Let's use a set of uncovered_words for O(1) lookup?
                # Creating a set for every w is expensive.
                # Alternative: precompute a boolean matrix? No.
                # Let's just iterate balls[w] and check against a boolean mask of uncovered.
                
                # Actually, we can just count how many of the *specific* uncovered_words are in balls[w].
                # If len(uncovered_words) is small, this is fast.
                
                # Optimization: If len(uncovered_words) > 11, we can't cover them all with one w.
                # But we want to maximize coverage, not necessarily cover all.
                # However, to fix validity, we MUST cover all.
                # So if len(uncovered_words) > 11, we cannot fix it with a single addition.
                # We should revert the removal immediately.
                
                if len(uncovered_words) > 11:
                    break # Cannot fix with 1 add
                    
                count = 0
                for u in uncovered_words:
                    if u in balls[w]: # O(11) check
                        count += 1
                if count == len(uncovered_words):
                    best_add = w
                    max_cover = count
                    break # Found a perfect fix, take it
            
            if best_add != -1:
                # Add best_add
                in_code[best_add] = True
                cover_count[balls[best_add]] += 1
                # Size remains same (removed 1, added 1)
                # This is a "move" that maintains size but changes configuration.
                # It might allow further removals later.
            else:
                # Cannot fix. Revert removal.
                in_code[c_to_remove] = True
                cover_count[balls[c_to_remove]] += 1
                
    # Reconstruct code from best_in_code
    final_code = np.where(best_in_code)[0].tolist()
    
    # Convert indices to words
    result = []
    for idx in final_code:
        w = [0] * n
        temp = idx
        for i in range(n-1, -1, -1):
            w[i] = temp % q
            temp //= q
        result.append(w)
        
    return result

def _fallback(q, n, radius):
    total = q ** n
    if total <= 10000:
        res = []
        for i in range(total):
            w = [0]*n
            t = i
            for k in range(n-1, -1, -1):
                w[k] = t % q
                t //= q
            res.append(w)
        return res
    return []

def construct(q, n, radius):
    # Only optimized for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1()
    # Fallback
    return _fallback(q, n, radius)
