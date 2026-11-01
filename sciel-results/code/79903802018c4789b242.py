import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin before 120s limit
    
    # The task is specific to q=3, n=8, radius=1.
    # If parameters differ, fall back to a simple heuristic or seed logic.
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed-like logic for other parameters
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    total_words = q ** n
    
    # Precompute neighbors for all words using vectorized approach
    # For each word i, its neighbors are itself plus all words at Hamming distance 1
    neighbors = np.zeros((total_words, 17), dtype=np.int32)
    
    for i in range(total_words):
        # Convert index to base-q digits
        w = np.zeros(n, dtype=np.int32)
        temp = i
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
        
        neighbors[i, 0] = i
        
        for pos in range(n):
            for val in range(1, q):  # val = 1, 2 (skip 0 which is original)
                # Adjust: we need all different values, so for each pos, try q-1 other values
                new_val = (w[pos] + val) % q
                # Compute new index efficiently
                # The digit at position pos changes from w[pos] to new_val
                # Difference in index: (new_val - w[pos]) * q^(n-1-pos)
                power = q ** (n - 1 - pos)
                neighbors[i, 1 + pos * (q-1) + (val - 1)] = i + (new_val - w[pos]) * power
    
    # Greedy Set Cover
    covered_count = np.zeros(total_words, dtype=np.int32)
    code = []
    
    while True:
        if time.time() > deadline:
            break
        
        # Find the word that covers the most uncovered words
        best_word = -1
        max_uncovered = -1
        
        for i in range(total_words):
            # Check if this word covers any uncovered words
            nbs = neighbors[i]
            uncovered = np.sum(covered_count[nbs] == 0)
            
            if uncovered > max_uncovered:
                max_uncovered = uncovered
                best_word = i
        
        if best_word == -1 or max_uncovered == 0:
            break
        
        # Add best_word to code
        code.append(best_word)
        
        # Update covered counts
        covered_count[neighbors[best_word]] += 1
        
        # Early termination if all covered
        if np.all(covered_count > 0):
            break

    # Local Search Phase 1: Redundancy Removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            # Check if removing c leaves any word uncovered
            if np.any(covered_count[neighbors[c]] == 1):
                continue
            # Remove c
            code.pop(i)
            covered_count[neighbors[c]] -= 1
            changed = True
            break

    # Local Search Phase 2: Swap-based optimization (WalkSAT-style)
    # Try to replace a codeword with another that covers more uncovered words
    # or try to remove a codeword if it becomes redundant after some swaps
    
    best_code = list(code)
    best_size = len(code)
    
    # Multi-restart local search with random perturbations
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Start from current best
        code = list(best_code)
        covered_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            covered_count[neighbors[c]] += 1
        
        # Random perturbation: remove a random codeword and repair
        if len(code) > 1:
            remove_idx = random.randint(0, len(code) - 1)
            removed = code.pop(remove_idx)
            covered_count[neighbors[removed]] -= 1
            
            # Repair: add back words that are now uncovered
            uncovered_words = np.where(covered_count == 0)[0]
            for uw in uncovered_words:
                # Find a word that covers uw and add it
                # Try all words that are neighbors of uw
                candidates = neighbors[uw]
                # Pick the one that covers the most uncovered words
                best_cand = -1
                best_count = -1
                for cand in candidates:
                    cnt = np.sum(covered_count[neighbors[cand]] == 0)
                    if cnt > best_count:
                        best_count = cnt
                        best_cand = cand
                if best_cand >= 0:
                    code.append(best_cand)
                    covered_count[neighbors[best_cand]] += 1
        
        # Local search: try to remove redundant words
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if np.any(covered_count[neighbors[c]] == 1):
                    continue
                code.pop(i)
                covered_count[neighbors[c]] -= 1
                changed = True
                break
        
        # Local search: try swaps
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Try to swap a random codeword with a better one
            if len(code) == 0:
                break
                
            swap_idx = random.randint(0, len(code) - 1)
            old_c = code[swap_idx]
            
            # Find a word that would improve coverage if we swap
            # Try random candidates
            improved = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                new_c = random.randint(0, total_words - 1)
                if new_c == old_c:
                    continue
                
                # Check if new_c is already in code
                if new_c in code:
                    continue
                
                # Simulate swap: remove old_c, add new_c
                # Check if all words are still covered
                temp_covered = covered_count.copy()
                temp_covered[neighbors[old_c]] -= 1
                temp_covered[neighbors[new_c]] += 1
                
                if np.all(temp_covered > 0):
                    # Valid swap, but does it help?
                    # For now, accept if it doesn't increase size (same size)
                    # Actually, we want to keep same size but maybe enable future removals
                    code[swap_idx] = new_c
                    covered_count = temp_covered
                    improved = True
                    break
            
            if not improved:
                # Try to just remove a word if possible
                for i in range(len(code) - 1, -1, -1):
                    c = code[i]
                    if np.any(covered_count[neighbors[c]] == 1):
                        continue
                    code.pop(i)
                    covered_count[neighbors[c]] -= 1
                    break
        
        # Update best if improved
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
    
    code = best_code
    
    # Final redundancy removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            if np.any(covered_count[neighbors[c]] == 1):
                continue
            code.pop(i)
            covered_count[neighbors[c]] -= 1
            changed = True
            break

    # Convert indices back to lists of integers
    result = []
    for idx in code:
        w = [0] * n
        temp = idx
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
        result.append(w)
        
    return result
