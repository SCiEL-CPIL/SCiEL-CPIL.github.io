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


def _greedy_construct(q, n, radius, seed=0):
    """
    Greedy set cover construction.
    Starts with empty set, adds word that covers most uncovered points.
    """
    total = q ** n
    all_words = [_idx_to_word(i, n, q) for i in range(total)]
    
    # Precompute covers for all words
    covers_all = []
    for i in range(total):
        c = all_words[i]
        cov = set()
        for j in range(total):
            if _hamming(c, all_words[j]) <= radius:
                cov.add(j)
        covers_all.append(cov)
        
    uncovered = set(range(total))
    code_indices = []
    
    while uncovered:
        best_idx = -1
        max_covered = -1
        # To add randomness and allow restarts to explore different paths
        # We can shuffle the order or add small random noise to coverage count
        rng = random.Random(seed + len(code_indices))
        
        for i in range(total):
            if i in code_indices:
                continue
            covered_count = len(covers_all[i] & uncovered)
            # Add small random tie-breaker to escape deterministic local minima
            noise = rng.random() * 0.1
            if covered_count + noise > max_covered:
                max_covered = covered_count + noise
                best_idx = i
                
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        uncovered -= covers_all[best_idx]
        
    return code_indices


def _local_search(code_indices, q, n, radius, covers_all, time_limit=5.0, seed=0):
    """
    Local search with 1-swap and 2-swap moves.
    """
    total = q ** n
    all_words = [_idx_to_word(i, n, q) for i in range(total)]
    
    # Initialize state
    current_code = list(code_indices)
    current_count = np.zeros(total, dtype=np.int32)
    
    for idx in current_code:
        for w in covers_all[idx]:
            current_count[w] += 1
            
    best_code = list(current_code)
    best_size = len(best_code)
    
    start_time = time.time()
    rng = random.Random(seed)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to remove redundant words
        for i in range(len(current_code) - 1, -1, -1):
            idx = current_code[i]
            can_remove = True
            for w in covers_all[idx]:
                if current_count[w] == 1:
                    can_remove = False
                    break
            if can_remove:
                for w in covers_all[idx]:
                    current_count[w] -= 1
                current_code.pop(i)
                improved = True
                break # Restart removal loop
        
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            continue # Loop back to try more removals
        
        # 2. Try 1-swap moves (remove one, add one)
        # Only useful if it enables a subsequent removal? 
        # Or if we just want to move to a better basin.
        # For now, let's focus on 2-swaps which are more powerful for escaping minima.
        
        # 3. Try 2-swap moves (remove two, add two)
        # Pick two random words to remove
        if len(current_code) >= 2:
            i, j = rng.sample(range(len(current_code)), 2)
            idx_out1 = current_code[i]
            idx_out2 = current_code[j]
            
            # Pick two random words to add (not in current code)
            code_set = set(current_code)
            idx_in1 = rng.randint(0, total - 1)
            while idx_in1 in code_set:
                idx_in1 = rng.randint(0, total - 1)
            
            idx_in2 = rng.randint(0, total - 1)
            while idx_in2 in code_set or idx_in2 == idx_in1:
                idx_in2 = rng.randint(0, total - 1)
                
            cov_out1 = covers_all[idx_out1]
            cov_out2 = covers_all[idx_out2]
            cov_in1 = covers_all[idx_in1]
            cov_in2 = covers_all[idx_in2]
            
            # Check validity
            # We need to ensure no word drops to 0 coverage.
            # Words at risk: those in cov_out1 or cov_out2.
            # For each w in (cov_out1 U cov_out2):
            #   new_count = current_count[w] - (1 if w in cov_out1 else 0) - (1 if w in cov_out2 else 0) + (1 if w in cov_in1 else 0) + (1 if w in cov_in2 else 0)
            
            valid = True
            # Optimization: only check words in cov_out1 or cov_out2
            risky_words = set(cov_out1).union(set(cov_out2))
            
            # Precompute sets for fast lookup
            in1_set = set(cov_in1)
            in2_set = set(cov_in2)
            
            for w in risky_words:
                new_count = current_count[w]
                if w in cov_out1: new_count -= 1
                if w in cov_out2: new_count -= 1
                if w in in1_set: new_count += 1
                if w in in2_set: new_count += 1
                if new_count == 0:
                    valid = False
                    break
                    
            if valid:
                # Apply swap
                for w in cov_out1: current_count[w] -= 1
                for w in cov_out2: current_count[w] -= 1
                for w in cov_in1: current_count[w] += 1
                for w in cov_in2: current_count[w] += 1
                
                current_code[i] = idx_in1
                current_code[j] = idx_in2
                
                # Try to remove redundant words
                changed = True
                while changed:
                    changed = False
                    for k in range(len(current_code) - 1, -1, -1):
                        idx_k = current_code[k]
                        can_remove = True
                        for w in covers_all[idx_k]:
                            if current_count[w] == 1:
                                can_remove = False
                                break
                        if can_remove:
                            for w in covers_all[idx_k]:
                                current_count[w] -= 1
                            current_code.pop(k)
                            changed = True
                            break
                            
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    
                # If we didn't reduce size, we might have just moved to a new local min.
                # We accept the move anyway to explore.
                # But to avoid getting stuck in cycles, we might want to revert if no improvement after some steps?
                # For now, we just accept and continue.
                
        # If no moves were made, we are stuck. Break or randomize.
        # Since we always try a 2-swap if size >= 2, we will keep moving.
        
    return best_code


def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    
    total = q ** n
    all_words = [_idx_to_word(i, n, q) for i in range(total)]
    
    # Precompute covers for ALL words
    covers_all = []
    for i in range(total):
        c = all_words[i]
        cov = []
        for j in range(total):
            if _hamming(c, all_words[j]) <= radius:
                cov.append(j)
        covers_all.append(cov)
        
    best_code_indices = None
    best_size = float('inf')
    
    # Multiple restarts with different seeds
    # Time budget per restart
    remaining_time = _DEADLINE - (time.time() - start_time) - 5.0
    num_restarts = max(1, int(remaining_time / 10.0)) # Aim for ~10s per restart
    
    for seed in range(num_restarts):
        if time.time() - start_time > _DEADLINE - 5.0:
            break
            
        # 1. Greedy Construction
        code_indices = _greedy_construct(q, n, radius, seed=seed)
        
        # 2. Local Search
        time_left_for_ls = _DEADLINE - (time.time() - start_time) - 5.0
        if time_left_for_ls < 1.0:
            break
            
        polished_code = _local_search(code_indices, q, n, radius, covers_all, time_limit=time_left_for_ls, seed=seed)
        
        if len(polished_code) < best_size:
            best_size = len(polished_code)
            best_code_indices = polished_code
            
    # Convert best indices back to words
    if best_code_indices is None:
        # Fallback to tetracode if something went wrong
        best_code = _direct_sum_tetracode(q, n, radius)
    else:
        best_code = [all_words[i] for i in best_code_indices]
        
    # Verify
    if _verify_code(best_code, q, n, radius):
        return best_code
    else:
        # Fallback
        return _direct_sum_tetracode(q, n, radius)

# EVOLVE-BLOCK-END
