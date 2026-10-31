import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        # Compute sumset
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        # Compute difference set
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    best_score = 0
    best_set = [0, 1, 3]
    
    # Update best if seed is better
    seed_score = compute_score(best_set)
    if seed_score > best_score:
        best_score = seed_score

    # Stage 1: Analytical candidates based on geometric progression with base B
    # U = {0, 1, B, B^2, ..., B^(k-1)}
    # We scan B and k to find good starting points
    
    candidates = []
    
    # Try various bases B
    for B in range(2, 100):
        # Try various sizes k
        for k in range(2, 12):
            if time.time() > deadline - 10:
                break
            U = [0]
            current_val = 1
            for i in range(1, k):
                U.append(current_val)
                current_val *= B
            # Ensure 0 is in U (it is) and U is sorted
            U = sorted(U)
            score = compute_score(U)
            if score > best_score:
                best_score = score
                best_set = list(U)
            candidates.append(list(U))
            
            # Also try variants where the first step is not 1 but small
            # e.g., {0, 1, 1+B, 1+B+(B^2-1), ...} is just shifting, not helpful for max U.
            # Better: {0, 1, B, B^2, ...} is the standard.
            # What if we drop 1? {0, B, B^2, ...}
            U2 = [0]
            current_val = B
            for i in range(1, k):
                U2.append(current_val)
                current_val *= B
            U2 = sorted(U2)
            score2 = compute_score(U2)
            if score2 > best_score:
                best_score = score2
                best_set = list(U2)
            candidates.append(list(U2))

    if time.time() > deadline - 5:
        return best_set

    # Stage 2: Local Search around the best candidates
    # We will refine the best_set and a few other top candidates
    
    # Keep a list of top candidates to refine
    top_candidates = []
    for c in candidates:
        top_candidates.append((compute_score(c), c))
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Refine the top 5 candidates
    for score, cand in top_candidates[:5]:
        if time.time() > deadline - 2:
            break
            
        current_set = list(cand)
        current_score = score
        improved = True
        max_iterations = 200
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline - 2:
            improved = False
            iterations += 1
            n = len(current_set)
            max_val = max(current_set)
            
            # 1. Try adding a new element
            if n < 10:
                # Try values near existing elements or in gaps
                # Sample some random positions between 1 and max_val*2
                for _ in range(50):
                    if time.time() > deadline - 2:
                        break
                    new_el = random.randint(1, max_val * 2)
                    if new_el not in current_set:
                        new_set = sorted(current_set + [new_el])
                        new_score = compute_score(new_set)
                        if new_score > current_score:
                            current_set = new_set
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current_set)
                            improved = True
                            break # Restart loop with new set
            
            # 2. Try replacing an element
            if n >= 3:
                for i in range(n):
                    if current_set[i] == 0:
                        continue
                    # Try small perturbations
                    for delta in [-2, -1, 1, 2, 5, 10, -10, -5]:
                        new_val = current_set[i] + delta
                        if new_val < 0 or new_val in current_set:
                            continue
                        # Check if new_val is 0 (not allowed to replace non-0 with 0 if 0 is elsewhere? 
                        # Actually 0 must be in set. If we replace non-0 with 0, we might have duplicate 0.
                        # So new_val must not be 0 unless current_set[i] is the only non-zero? No, 0 is always there.
                        if new_val == 0:
                            continue
                            
                        new_set = list(current_set)
                        new_set[i] = new_val
                        new_set = sorted(set(new_set))
                        if 0 not in new_set:
                            continue
                        new_score = compute_score(new_set)
                        if new_score > current_score:
                            current_set = new_set
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current_set)
                            improved = True
                            break
                    if improved:
                        break
            
            # 3. Try removing an element (if size > 2)
            if n > 2 and random.random() < 0.1:
                for i in range(n):
                    if current_set[i] == 0:
                        continue
                    new_set = [x for j, x in enumerate(current_set) if j != i]
                    new_score = compute_score(new_set)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set)
                        improved = True
                        break

    return list(best_set)
