import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Handle the specific case K_3(8,1)
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Generic fallback for other cases
    return _generic_greedy(q, n, radius, deadline)

def _solve_ternary_8_1(deadline):
    q = 3
    n = 8
    total = q ** n  # 6561
    
    # Precompute all words
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute balls (indices of words within distance 1)
    # word_to_idx mapping
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    balls = []
    for w in all_words:
        neighbors = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for i in range(n):
            original = w_list[i]
            for d in range(q):
                if d != original:
                    w_list[i] = d
                    neighbors.append(word_to_idx[tuple(w_list)])
            w_list[i] = original
        balls.append(np.array(neighbors, dtype=np.int32))
        
    # 1. Greedy Set Cover
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    remaining = total
    
    while remaining > 0 and time.time() < deadline:
        # Calculate gain for each candidate
        gains = np.array([uncovered[b].sum() for b in balls], dtype=np.int32)
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        chosen.append(best_idx)
        uncovered[balls[best_idx]] = False
        remaining -= best_gain
        
    # 2. Prune
    chosen = _prune(chosen, balls, total)
    
    # 3. Local Search with Randomized Restarts
    best_chosen = chosen
    best_size = len(chosen)
    
    # Run local search multiple times
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Perturb: Randomly swap some codewords to escape local minima
        current_chosen = list(best_chosen)
        
        # Perform a few random valid swaps or removals to perturb
        for _ in range(10):
            if not current_chosen:
                break
            # Try to remove a random codeword
            idx_to_remove = random.randint(0, len(current_chosen) - 1)
            c_to_remove = current_chosen[idx_to_remove]
            
            # Check if removal is valid
            coverage_count = np.zeros(total, dtype=np.int32)
            for c in current_chosen:
                coverage_count[balls[c]] += 1
            
            coverage_count[balls[c_to_remove]] -= 1
            if np.any(coverage_count == 0):
                # Invalid removal, try a swap
                # Find a replacement
                critical = [j for j in balls[c_to_remove] if coverage_count[j] == 0]
                if critical:
                    # Find a codeword that covers the critical words
                    candidates = balls[critical[0]]
                    swapped = False
                    for c_prime in candidates:
                        if c_prime in current_chosen:
                            continue
                        # Check if c_prime covers all critical words
                        valid = True
                        for j in critical:
                            if c_prime not in balls[j]:
                                valid = False
                                break
                        if valid:
                            current_chosen[idx_to_remove] = c_prime
                            swapped = True
                            break
                    if not swapped:
                        # If no swap found, just keep it
                        pass
            else:
                # Valid removal, keep it
                current_chosen.pop(idx_to_remove)
                
        # Now run the main local search on this perturbed set
        refined_chosen = _local_search(current_chosen, balls, total, deadline, time_per_restart)
        
        if len(refined_chosen) < best_size:
            best_chosen = refined_chosen
            best_size = len(refined_chosen)
            
    return [all_words[i] for i in best_chosen]

def _prune(chosen, balls, total):
    if not chosen:
        return []
    
    # Count coverage
    coverage_count = np.zeros(total, dtype=np.int32)
    for c in chosen:
        coverage_count[balls[c]] += 1
        
    keep = [True] * len(chosen)
    any_removed = True
    while any_removed:
        any_removed = False
        for i, c in enumerate(chosen):
            if not keep[i]:
                continue
            # Check if any word in ball[c] is covered only by c
            redundant = True
            for j in balls[c]:
                if coverage_count[j] <= 1:
                    redundant = False
                    break
            if redundant:
                keep[i] = False
                any_removed = True
                coverage_count[balls[c]] -= 1
                
    return [chosen[i] for i in range(len(chosen)) if keep[i]]

def _local_search(chosen, balls, total, deadline, time_limit):
    if not chosen:
        return []
        
    chosen = list(chosen)
    
    # Initial coverage count
    coverage_count = np.zeros(total, dtype=np.int32)
    for c in chosen:
        coverage_count[balls[c]] += 1
        
    chosen_set = set(chosen)
    
    improved = True
    while improved and time.time() < deadline and time.time() < time_limit:
        improved = False
        
        for i, c in enumerate(list(chosen)):
            if time.time() > deadline or time.time() > time_limit:
                break
                
            # Find critical words: covered only by c
            critical = [j for j in balls[c] if coverage_count[j] == 1]
            
            if not critical:
                # c is redundant, remove it
                chosen.pop(i)
                chosen_set.remove(c)
                coverage_count[balls[c]] -= 1
                improved = True
                break
            else:
                # Try to find a replacement c' that covers all critical words
                # c' must be in balls[j] for all j in critical
                candidates = balls[critical[0]]
                found_swap = False
                for c_prime in candidates:
                    if c_prime in chosen_set:
                        continue
                    # Check if c_prime covers all critical words
                    valid = True
                    for j in critical[1:]:
                        if c_prime not in balls[j]:
                            valid = False
                            break
                    if valid:
                        # Perform swap
                        chosen[i] = c_prime
                        chosen_set.remove(c)
                        chosen_set.add(c_prime)
                        
                        # Update coverage counts
                        coverage_count[balls[c]] -= 1
                        coverage_count[balls[c_prime]] += 1
                        
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
    return chosen

def _generic_greedy(q, n, radius, deadline):
    total = q ** n
    if total > 100000:
        # Fallback
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    balls = []
    for w in all_words:
        neighbors = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for i in range(n):
            original = w_list[i]
            for d in range(q):
                if d != original:
                    w_list[i] = d
                    neighbors.append(word_to_idx[tuple(w_list)])
            w_list[i] = original
        balls.append(np.array(neighbors, dtype=np.int32))
        
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    remaining = total
    
    while remaining > 0 and time.time() < deadline:
        gains = np.array([uncovered[b].sum() for b in balls], dtype=np.int32)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        chosen.append(best_idx)
        uncovered[balls[best_idx]] = False
        remaining -= gains[best_idx]
        
    return [all_words[i] for i in chosen]
