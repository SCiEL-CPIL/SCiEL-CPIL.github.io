import time
import numpy as np
import random
import itertools
import os

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters (not expected in this task, but good practice)
    if q == 3 and n >= 4:
        # Simple tetracode extension, not optimal but valid
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls (neighborhoods of radius 1)
    # balls[w] is a list of 17 indices: w itself and its 16 neighbors
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    # Precompute ball_sets for O(1) membership checks and fast intersection
    # Using frozensets for intersection
    ball_sets = [frozenset(row) for row in balls]
    
    all_indices = np.arange(total, dtype=np.int32)
    
    best_size = total
    best_set = None
    
    # We will run multiple iterations of Greedy + Local Search
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction
        # To get diverse starts, we can randomize the tie-breaking in greedy
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        # Greedy Loop
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break # Time out for this iteration
                
            # Count coverage for each candidate among uncovered words
            # counts[c] = number of uncovered words in balls[c]
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break # Should not happen if uncovered_mask has True
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best to encourage diversity
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts
            cover_count[balls[best_idx]] += 1
            
            # Update uncovered mask
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue # Invalid solution due to timeout in greedy
            
        # 2. Local Search
        # Phase 1: Remove redundant codewords
        # We iterate until no more can be removed
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is removable: all words in balls[c] are covered by >= 2 codewords
                # i.e., min(cover_count[balls[c]]) > 1
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Phase 2: Deterministic 1-Swap
        # Try to replace a codeword c with c' if it leads to a better state (size reduction or same size but better structure?)
        # Actually, 1-swap doesn't reduce size. It maintains size.
        # The goal of 1-swap is to create "removable" codewords for Phase 1.
        # Or to escape local minima.
        # We perform 1-swaps that might enable removals.
        # Strategy: For each c, try to find c' such that the new configuration has more "slack" or allows removal.
        # Simpler: Just try to swap c with c' if c' covers all critical words of c.
        # This doesn't change size, but might change the distribution.
        # Then run Phase 1 again.
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                # First, try to remove
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                # Try to swap c with some c'
                swapped = False
                # Critical set: words covered ONLY by c
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    # Should have been removed
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Find c' in intersection of balls[x] for x in critical_words
                # Start with ball of first critical word
                cand_set = ball_sets[critical_words[0]]
                for cw in critical_words[1:]:
                    cand_set = cand_set.intersection(ball_sets[cw])
                    if not cand_set:
                        break
                
                if cand_set:
                    # Remove c from candidates (we don't want to swap with itself)
                    if c in cand_set:
                        cand_set = cand_set - {c}
                    
                    if cand_set:
                        # Pick a candidate. 
                        # Heuristic: Pick the one that covers the most "low cover count" words?
                        # For now, pick the first one (arbitrary).
                        # To add diversity, we can randomize among candidates.
                        c_prime = next(iter(cand_set))
                        
                        # Perform swap
                        cover_count[balls[c]] -= 1
                        cover_count[balls[c_prime]] += 1
                        new_selected.append(c_prime)
                        swapped = True
                        improved = True
                        break
                        # Note: We break out of the swap logic for this c.
                        # We appended c_prime. We skipped c.
                        # The cover_count update is correct.
                    
                if not swapped:
                    new_selected.append(c)
            
            selected = new_selected
            
        # Phase 3: 2-Exchange (Remove 2, Add 1)
        # This is the only way to reduce size further if 1-swap/removal stalls.
        if len(selected) > 486 and time.time() < deadline - 5.0:
            n_selected = len(selected)
            if n_selected >= 2:
                # Use a set for fast membership check
                selected_set = set(selected)
                for _ in range(100): # Limit trials
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Pick two random distinct indices from selected
                    i1 = np.random.randint(n_selected)
                    i2 = np.random.randint(n_selected)
                    if i1 == i2:
                        continue
                    c1 = selected[i1]
                    c2 = selected[i2]
                    
                    # Calculate U1 and U2 (critical words)
                    U1 = balls[c1][(cover_count[balls[c1]] == 1)]
                    U2 = balls[c2][(cover_count[balls[c2]] == 1)]
                    
                    # Combine
                    U_combined = np.concatenate([U1, U2])
                    
                    if len(U_combined) == 0:
                        # Both removable?
                        cover_count[balls[c1]] -= 1
                        cover_count[balls[c2]] -= 1
                        selected = [c for c in selected if c != c1 and c != c2]
                        improved = True
                        break
                    
                    # Pick c3 randomly
                    c3 = all_indices[np.random.randint(total)]
                    if c3 in selected_set:
                        continue
                        
                    # Check if c3 covers all U_combined
                    valid = True
                    for u in U_combined:
                        if u not in ball_sets[c3]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        cover_count[balls[c1]] -= 1
                        cover_count[balls[c2]] -= 1
                        cover_count[balls[c3]] += 1
                        
                        # Update selected list
                        new_selected = []
                        for c in selected:
                            if c != c1 and c != c2:
                                new_selected.append(c)
                        new_selected.append(c3)
                        selected = new_selected
                        improved = True
                        break

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break # Hit lower bound (unlikely)
            if best_size <= 486:
                # We hit the known best. 
                # Try to improve further?
                # If we find 486, we are at SOTA.
                # Let's break to save time.
                break

    # Convert best_set to output format
    if best_set is None:
        # Fallback: Return a valid (though large) code
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
