"""Greedy covering-code search with parameterized heuristic and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin


def _idx_to_word(idx, n, q):
    """Convert a linear index to a word over {0,1,...} with n digits (base q)."""
    digits = [0] * n
    for i in range(n):
        digits[i] = idx % q
        idx //= q
    return digits


def _word_to_idx(word, q, n):
    """Convert a word to a linear index."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _direct_sum_tetracode(q, n, radius):
    """Direct sum of ternary tetracode with all tails."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return [list(h) for h in tetracode]
    if n < 4:
        # Fallback for small n
        return [[0] * n]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _compute_covers(code, q, n, radius):
    """
    Returns a list of sets (or lists) of indices covered by each codeword.
    code: list of words (lists of ints)
    """
    total = q ** n
    covers = []
    for c in code:
        covered_indices = []
        for idx in range(total):
            w = _idx_to_word(idx, n, q)
            if _hamming(c, w) <= radius:
                covered_indices.append(idx)
        covers.append(covered_indices)
    return covers


def _greedy_delete(code, q, n, radius, time_limit=5.0):
    """
    Remove redundant codewords from code.
    Returns (new_code, coverage_count, covers)
    """
    total = q ** n
    # Initial coverage count
    coverage_count = np.zeros(total, dtype=np.int32)
    covers = []
    
    # Compute covers and initial counts
    for c in code:
        covered_indices = []
        for idx in range(total):
            w = _idx_to_word(idx, n, q)
            if _hamming(c, w) <= radius:
                covered_indices.append(idx)
                coverage_count[idx] += 1
        covers.append(covered_indices)
    
    start_time = time.time()
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > time_limit:
            break
        
        # Try to remove each codeword
        # We iterate backwards to allow deletion without index shifting issues
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            cov_i = covers[i]
            
            # Check if any word is covered ONLY by this codeword
            # i.e., coverage_count[w] == 1 for any w in cov_i
            # If so, we cannot remove c.
            # If coverage_count[w] > 1 for all w in cov_i, we can remove c.
            
            can_remove = True
            for w in cov_i:
                if coverage_count[w] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove codeword i
                # Update coverage counts
                for w in cov_i:
                    coverage_count[w] -= 1
                
                # Remove from lists
                code.pop(i)
                covers.pop(i)
                changed = True
                # Restart loop as indices changed
                break
        
    return code, coverage_count, covers


def _verify_code(code, q, n, radius):
    """Verify that every word is within radius of some codeword."""
    total = q ** n
    if total > 50000:
        # Sampling for very large spaces (not expected here for q=3, n=6)
        rng = np.random.default_rng(42)
        for _ in range(10000):
            w = list(rng.integers(0, q, size=n))
            found = False
            for cw in code:
                if _hamming(w, cw) <= radius:
                    found = True
                    break
            if not found:
                return False
        return True
    else:
        # Exact check
        words_list = [_idx_to_word(i, n, q) for i in range(total)]
        words_arr = np.array(words_list, dtype=np.int8)
        code_arr = np.array(code, dtype=np.int8)
        
        # Check if every row in words_arr is covered by some row in code_arr
        # dists[i, j] <= radius for some j
        diff = words_arr[:, np.newaxis, :] != code_arr[np.newaxis, :, :]
        dists = diff.sum(axis=2)
        covered = (dists <= radius).any(axis=1)
        return covered.all()


def _simulated_annealing(code, q, n, radius, time_limit=100.0):
    """
    Simulated Annealing to minimize code size.
    State: (code, coverage_count, covers)
    Move: Swap a codeword with a random word from the space.
    If valid, apply Greedy Deletion.
    Accept if size decreases or with probability based on temperature.
    """
    total = q ** n
    # Initialize state
    current_code, current_count, current_covers = _greedy_delete(code, q, n, radius, time_limit=5.0)
    current_size = len(current_code)
    
    # Precompute all words for random selection
    all_words = [_idx_to_word(i, n, q) for i in range(total)]
    
    # We need a fast way to check validity.
    # We maintain coverage_count.
    # Valid if min(coverage_count) >= 1.
    
    start_time = time.time()
    temperature = 1.0
    cooling = 0.999
    
    best_code = list(current_code)
    best_size = current_size
    
    # To speed up random word selection, we can just pick random indices.
    # To avoid picking a word already in code, we can pick random index and check.
    # Or maintain a set of indices in code.
    code_indices = set(_word_to_idx(c, q, n) for c in current_code)
    
    while time.time() - start_time < time_limit:
        # Pick a random codeword to swap out
        i = random.randint(0, len(current_code) - 1)
        c_out = current_code[i]
        
        # Pick a random word to swap in
        # We want a word NOT in current_code
        c_in_idx = random.randint(0, total - 1)
        if c_in_idx in code_indices:
            continue
        c_in = all_words[c_in_idx]
        
        # Calculate covers for c_in
        # We need to know which words c_in covers.
        # We can compute this on the fly or precompute a global "covers for all words" matrix?
        # 729x729 is small. Let's precompute a matrix M where M[a,b]=1 if a covers b.
        # Actually, we already have `current_covers` for current code.
        # We don't have covers for c_in.
        # Computing covers for c_in takes O(total * n) = 729 * 6 = 4374 ops.
        # This is fast.
        
        cov_in = []
        for idx in range(total):
            if _hamming(c_in, all_words[idx]) <= radius:
                cov_in.append(idx)
        
        # Check if swap is valid
        # Temporarily remove c_out and add c_in
        # c_out corresponds to current_covers[i]
        cov_out = current_covers[i]
        
        # Check min coverage after swap
        # For w in cov_out: count decreases by 1
        # For w in cov_in: count increases by 1
        
        # We need to check if any w ends up with count 0.
        # Only words in cov_out are at risk of becoming 0.
        # Words in cov_in \ cov_out are safe (count increases).
        # Words in cov_out \ cov_in are at risk.
        # Words in cov_out \ cap cov_in are at risk.
        
        valid = True
        for w in cov_out:
            # Count change: -1 from c_out, +1 if w in cov_in
            delta = -1
            if w in cov_in: # This check is O(len(cov_in)) which is bad.
                # Better: create a set or boolean array for cov_in
                pass
        
        # Optimization: Use numpy arrays for counts?
        # Let's use a temporary array or just check the risky words.
        # Risky words are those in cov_out.
        # For each w in cov_out:
        #   new_count = current_count[w] - 1 + (1 if w in cov_in else 0)
        #   if new_count == 0: invalid
        
        # To make "w in cov_in" fast, convert cov_in to a set or boolean array.
        cov_in_set = set(cov_in)
        
        for w in cov_out:
            new_count = current_count[w] - 1
            if w in cov_in_set:
                new_count += 1
            if new_count == 0:
                valid = False
                break
        
        if not valid:
            continue
            
        # Swap is valid. Apply it.
        # Update counts
        for w in cov_out:
            current_count[w] -= 1
        for w in cov_in:
            current_count[w] += 1
            
        # Update code and covers
        current_code[i] = c_in
        current_covers[i] = cov_in
        
        # Update code_indices
        code_indices.remove(_word_to_idx(c_out, q, n))
        code_indices.add(c_in_idx)
        
        # Apply Greedy Deletion
        # We need to re-run greedy deletion logic.
        # We can write a helper that modifies the state in place.
        
        changed = True
        while changed:
            changed = False
            for k in range(len(current_code) - 1, -1, -1):
                cov_k = current_covers[k]
                can_remove = True
                for w in cov_k:
                    if current_count[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for w in cov_k:
                        current_count[w] -= 1
                    current_code.pop(k)
                    current_covers.pop(k)
                    code_indices.remove(_word_to_idx(current_code[k] if k < len(current_code) else current_code[-1], q, n)) 
                    # Wait, index management after pop is tricky.
                    # Let's just rebuild code_indices after deletion? Or update carefully.
                    # Actually, we removed index k.
                    # The word removed was current_code[k] before pop.
                    # Let's store it.
                    # Re-doing this block carefully.
                    
                    # Better approach for deletion:
                    # Identify the word to remove.
                    # We are in a loop.
                    # Let's just restart the deletion loop from scratch if we remove one.
                    # It's simpler and less error prone.
                    
                    # For now, let's just accept the swap and run a full _greedy_delete again?
                    # _greedy_delete recomputes covers. That's expensive.
                    # Let's stick to the in-place update but be careful.
                    
                    # Let's abort the in-place deletion and just use the state.
                    # The swap itself might not reduce size, but it moves us to a new local minimum.
                    # We should run a full greedy deletion pass.
                    # To avoid recomputing covers from scratch, we can pass the current state.
                    # But _greedy_delete as written recomputes.
                    # Let's modify _greedy_delete to accept initial state?
                    # Or just accept the cost of recomputation if it's fast enough?
                    # 729 words * 6 dims * 729 checks = 3M ops. ~0.01s.
                    # If we do this 1000 times, 10s. Acceptable.
                    
                    break
            
            # If we broke out of the inner loop due to a deletion, we restart.
            # But we messed up the indices.
            # Let's just call a helper that does greedy deletion given (code, count, covers).
            
    return best_code


def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    
    # 1. Tetracode
    code = _direct_sum_tetracode(q, n, radius)
    
    # 2. Greedy Deletion
    code, count, covers = _greedy_delete(code, q, n, radius, time_limit=5.0)
    
    # 3. Simulated Annealing
    # We need a robust SA implementation.
    # The previous one was flawed. Let's write a cleaner one.
    
    total = q ** n
    all_words = [_idx_to_word(i, n, q) for i in range(total)]
    
    # Precompute covers for ALL words?
    # 729 words. Covers for each word.
    # covers_all[i] = list of indices covered by all_words[i]
    covers_all = []
    for i in range(total):
        c = all_words[i]
        cov = []
        for j in range(total):
            if _hamming(c, all_words[j]) <= radius:
                cov.append(j)
        covers_all.append(cov)
        
    # State: code (list of indices), count (array)
    # We work with indices for speed.
    current_code_indices = [_word_to_idx(c, q, n) for c in code]
    current_count = count
    current_covers = [covers_all[i] for i in current_code_indices]
    
    best_code_indices = list(current_code_indices)
    best_size = len(best_code_indices)
    
    start_sa_time = time.time()
    time_limit_sa = _DEADLINE - (start_sa_time - start_time) - 10
    
    temperature = 1.0
    cooling = 0.9995
    
    while time.time() - start_sa_time < time_limit_sa:
        # Pick random index in current code
        i = random.randint(0, len(current_code_indices) - 1)
        idx_out = current_code_indices[i]
        
        # Pick random index in space
        idx_in = random.randint(0, total - 1)
        if idx_in == idx_out:
            continue
            
        # Check if idx_in is already in code
        # For small code, linear search is fine.
        if idx_in in current_code_indices:
            continue
            
        cov_out = covers_all[idx_out]
        cov_in = covers_all[idx_in]
        
        # Check validity of swap
        valid = True
        cov_in_set = set(cov_in)
        for w in cov_out:
            new_count = current_count[w] - 1
            if w in cov_in_set:
                new_count += 1
            if new_count == 0:
                valid = False
                break
                
        if not valid:
            continue
            
        # Apply swap
        for w in cov_out:
            current_count[w] -= 1
        for w in cov_in:
            current_count[w] += 1
            
        current_code_indices[i] = idx_in
        current_covers[i] = cov_in
        
        # Greedy Deletion
        changed = True
        while changed:
            changed = False
            for k in range(len(current_code_indices) - 1, -1, -1):
                cov_k = current_covers[k]
                can_remove = True
                for w in cov_k:
                    if current_count[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove k
                    for w in cov_k:
                        current_count[w] -= 1
                    removed_idx = current_code_indices[k]
                    current_code_indices.pop(k)
                    current_covers.pop(k)
                    changed = True
                    break
                    
        new_size = len(current_code_indices)
        
        if new_size < best_size:
            best_size = new_size
            best_code_indices = list(current_code_indices)
            
        # Cooling
        temperature *= cooling
        
    # Convert best indices back to words
    best_code = [all_words[i] for i in best_code_indices]
    
    # Verify
    if _verify_code(best_code, q, n, radius):
        return best_code
    else:
        # Fallback
        return _direct_sum_tetracode(q, n, radius)

# EVOLVE-BLOCK-END
