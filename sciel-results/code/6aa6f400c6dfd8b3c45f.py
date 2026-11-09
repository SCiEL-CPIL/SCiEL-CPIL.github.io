import time
import random
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 85.0  # Safety margin before 90s kill
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Precompute popcount for all 18-bit values
    popcount = np.zeros(1 << N, dtype=np.uint8)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def time_left():
        return deadline - time.time()

    def vectorized_greedy(seed=42):
        """Vectorized greedy construction with random ordering."""
        rng = random.Random(seed)
        
        # Candidates: all words with weight >= 3 (compatible with 0)
        # Actually, weight must be >= MIN_DIST to be compatible with 0
        # But we can include weight >= 3 and let the check filter
        candidates = [c for c in range(1, 1 << N) if popcount[c] >= 3]
        rng.shuffle(candidates)
        
        words = [0]
        words_set = {0}
        candidates_remaining = list(candidates)
        
        while time_left() > 3.0:
            if not candidates_remaining:
                break
            
            # Check candidates in chunks against all current words
            chunk_size = 5000
            added = False
            
            for i in range(0, len(candidates_remaining), chunk_size):
                if time_left() < 3.0:
                    break
                
                chunk = candidates_remaining[i:i+chunk_size]
                if not chunk:
                    continue
                
                c_arr = np.array(chunk, dtype=np.uint32)
                
                # Check each candidate against all words
                valid_mask = np.ones(len(chunk), dtype=bool)
                
                for w in words:
                    xors = c_arr ^ w
                    dists = popcount[xors]
                    valid_mask &= (dists >= MIN_DIST)
                    if not np.any(valid_mask):
                        break
                
                if np.any(valid_mask):
                    # Pick a random valid candidate
                    valid_indices = np.where(valid_mask)[0]
                    idx = rng.choice(valid_indices.tolist())
                    new_word = chunk[idx]
                    
                    words.append(new_word)
                    words_set.add(new_word)
                    candidates_remaining.remove(new_word)
                    added = True
                    break  # Restart chunk loop with updated words
            
            if not added:
                # No candidate could be added; try a different approach
                # Remove some candidates that are most constrained
                break
        
        return words

    def try_add_words(words, seed=0):
        """Try to add more words to the code using vectorized checks."""
        rng = random.Random(seed)
        words = list(words)
        words_set = set(words)
        
        # Generate random candidates to try adding
        added_any = True
        iterations = 0
        while added_any and time_left() > 2.0:
            added_any = False
            iterations += 1
            if iterations > 200:
                break
            
            # Try random candidates
            for _ in range(200):
                if time_left() < 2.0:
                    break
                
                c = rng.randint(1, (1 << N) - 1)
                if c in words_set:
                    continue
                if popcount[c] < 3:
                    continue
                
                # Check compatibility with all words
                c_arr = np.array([c], dtype=np.uint32)
                w_arr = np.array(words, dtype=np.uint32)
                xors = c_arr[0] ^ w_arr
                dists = popcount[xors]
                
                if np.all(dists >= MIN_DIST):
                    words.append(c)
                    words_set.add(c)
                    added_any = True
        
        return words

    def local_search_with_swaps(words, seed=0):
        """Local search: try to add words, and if stuck, try swaps."""
        rng = random.Random(seed)
        words = list(words)
        words_set = set(words)
        best_words = list(words)
        best_size = len(words)
        
        max_rounds = 50
        for round_num in range(max_rounds):
            if time_left() < 2.0:
                break
            
            # Try to add words
            words = try_add_words(words, seed=seed + round_num)
            
            if len(words) > best_size:
                best_size = len(words)
                best_words = list(words)
            
            # If no improvement, try a swap
            if len(words) == best_size and len(words) > 1:
                # Pick a random word to remove
                idx = rng.randint(0, len(words) - 1)
                removed = words[idx]
                words.pop(idx)
                words_set.discard(removed)
                
                # Try to add a new word
                improved = False
                for _ in range(300):
                    if time_left() < 2.0:
                        break
                    
                    c = rng.randint(1, (1 << N) - 1)
                    if c in words_set or c == removed:
                        continue
                    if popcount[c] < 3:
                        continue
                    
                    c_arr = np.array([c], dtype=np.uint32)
                    w_arr = np.array(words, dtype=np.uint32)
                    xors = c_arr[0] ^ w_arr
                    dists = popcount[xors]
                    
                    if np.all(dists >= MIN_DIST):
                        words.append(c)
                        words_set.add(c)
                        improved = True
                        if len(words) > best_size:
                            best_size = len(words)
                            best_words = list(words)
                        break
                
                if not improved:
                    # Put back the removed word
                    words.insert(idx, removed)
                    words_set.add(removed)
        
        return best_words

    # Main construction
    # Start with a good greedy construction
    best_words = vectorized_greedy(seed=42)
    best_len = len(best_words)
    
    # Try multiple greedy constructions with different seeds
    for seed in [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200]:
        if time_left() < 5.0:
            break
        
        cand = vectorized_greedy(seed=seed)
        if len(cand) > best_len:
            best_words = cand
            best_len = len(cand)
    
    # Local search on the best
    if time_left() > 5.0:
        improved = local_search_with_swaps(best_words, seed=123)
        if len(improved) > best_len:
            best_words = improved
            best_len = len(best_words)
    
    # Final systematic pass: try to add any remaining compatible words
    if time_left() > 2.0:
        current_set = set(best_words)
        # Try all words in order
        for c in range(1, 1 << N):
            if time_left() < 1.0:
                break
            if c in current_set:
                continue
            if popcount[c] < 3:
                continue
            
            c_arr = np.array([c], dtype=np.uint32)
            w_arr = np.array(best_words, dtype=np.uint32)
            xors = c_arr[0] ^ w_arr
            dists = popcount[xors]
            
            if np.all(dists >= MIN_DIST):
                best_words.append(c)
                current_set.add(c)
    
    # Ensure non-empty
    if not best_words:
        return [0]
    
    return best_words
