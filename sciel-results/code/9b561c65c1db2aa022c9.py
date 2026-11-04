import time
import itertools
import numpy as np

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
        
    # Greedy Set Cover
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    remaining = total
    
    while remaining > 0 and time.time() < deadline:
        # Calculate gain for each candidate
        # gains[i] = number of uncovered words in ball i
        gains = np.array([uncovered[b].sum() for b in balls], dtype=np.int32)
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        chosen.append(best_idx)
        uncovered[balls[best_idx]] = False
        remaining -= best_gain
        
    # Prune
    chosen = _prune(chosen, balls, total)
    
    # Local Search
    chosen = _local_search(chosen, balls, total, deadline)
    
    # Convert indices back to words
    return [all_words[i] for i in chosen]

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

def _local_search(chosen, balls, total, deadline):
    if not chosen:
        return []
        
    chosen = list(chosen)
    
    # Initial coverage count
    coverage_count = np.zeros(total, dtype=np.int32)
    for c in chosen:
        coverage_count[balls[c]] += 1
        
    chosen_set = set(chosen)
    
    # Precompute ball intersections for critical words? 
    # Or just check on the fly.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Recalculate coverage if structure changed? 
        # We maintain coverage_count incrementally.
        
        for i, c in enumerate(list(chosen)):
            if time.time() > deadline:
                break
                
            # Find critical words: covered only by c
            critical = [j for j in balls[c] if coverage_count[j] == 1]
            
            if not critical:
                # c is redundant, remove it
                chosen.pop(i)
                chosen_set.remove(c)
                coverage_count[balls[c]] -= 1
                improved = True
                # Restart loop as indices changed
                break
            else:
                # Find a replacement c' that covers all critical words
                # c' must be in intersection of balls of critical words
                # Intersection of balls of critical words
                # Since critical words are at distance 1 from c, their balls intersect at c.
                # But we need c' != c.
                
                # Candidate c' must be in balls[j] for all j in critical.
                # Let's pick the first critical word j0.
                # Candidates are in balls[j0].
                # Check if they cover all other critical words.
                
                candidates = balls[critical[0]]
                found_swap = False
                for c_prime in candidates:
                    if c_prime in chosen_set:
                        continue
                    # Check if c_prime covers all critical words
                    # c_prime must be in balls[j] for all j in critical
                    # Since critical words are distance 1 from c, and c_prime is distance 1 from c (if in balls[c]),
                    # the distance between c_prime and any critical word j is at most 2.
                    # But we need distance <= 1.
                    # Actually, if c_prime is in balls[j0], it covers j0.
                    # Does it cover other critical words?
                    # Critical words are distinct positions flipped from c.
                    # If c_prime flips position k, and critical word j flips position m.
                    # If k != m, distance(c_prime, j) = 2. Not covered.
                    # So c_prime must flip the SAME position as all critical words?
                    # No, critical words are different words.
                    # Wait.
                    # If c covers j1 and j2.
                    # j1 = c with pos 0 changed.
                    # j2 = c with pos 1 changed.
                    # A candidate c' must cover j1 and j2.
                    # d(c', j1) <= 1 and d(c', j2) <= 1.
                    # This implies c' is close to both.
                    # In ternary space, this is restrictive.
                    
                    # Efficient check:
                    # c' must be in balls[j] for all j in critical.
                    # We can just check:
                    valid = True
                    for j in critical[1:]:
                        if c_prime not in balls[j]:
                             # This check is O(1) if we had a set, but balls[j] is array.
                             # np.isin(c_prime, balls[j])
                             if not np.isin(c_prime, balls[j]):
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
                    # Restart loop to re-evaluate
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
