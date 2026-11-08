import time
from math import log
import random
import numpy as np
from itertools import combinations

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    # Use sets for sums and diffs
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def exhaustive_search_small(n, max_val, deadline):
    """Find best set of size n (including 0) with elements <= max_val."""
    best_s = -1
    best_c = None
    range_vals = range(1, max_val + 1)
    count = 0
    for comb in combinations(range_vals, n - 1):
        if time.time() > deadline - 30:
            break
        U = [0] + list(comb)
        s = score_set(U)
        if s > best_s:
            best_s = s
            best_c = U
        count += 1
        if count % 10000 == 0 and time.time() > deadline - 30:
            break
    return best_s, best_c

def local_search(U, deadline, max_iter=None):
    """Perform local search with simulated annealing."""
    current_U = sorted(set(U))
    current_score = score_set(current_U)
    best_score = current_score
    best_set = current_U[:]
    
    T = 1.0
    cooling = 0.9995
    iter_count = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 10:
        if T < 0.0005:
            break
        if max_iter and iter_count >= max_iter:
            break
        
        iter_count += 1
        move_type = random.random()
        new_U = current_U[:]
        valid_move = False
        
        if move_type < 0.5:
            # Replace an element with nearby value
            if len(new_U) > 1:
                idx = random.randint(1, len(new_U) - 1)
                old_val = new_U[idx]
                # Try multiple nearby values
                for _ in range(5):
                    delta = random.randint(-20, 20)
                    new_val = old_val + delta
                    if new_val > 0 and new_val not in new_U:
                        new_U[idx] = new_val
                        new_U = sorted(set(new_U))
                        valid_move = True
                        break
        
        elif move_type < 0.75:
            # Swap two elements
            if len(new_U) > 2:
                idx1 = random.randint(1, len(new_U) - 1)
                idx2 = random.randint(1, len(new_U) - 1)
                if idx1 != idx2:
                    old1, old2 = new_U[idx1], new_U[idx2]
                    delta1 = random.randint(-15, 15)
                    delta2 = random.randint(-15, 15)
                    new1, new2 = old1 + delta1, old2 + delta2
                    if new1 > 0 and new2 > 0 and new1 != new2:
                        new_U[idx1] = new1
                        new_U[idx2] = new2
                        new_U = sorted(set(new_U))
                        if len(new_U) == len(current_U):
                            valid_move = True
        
        elif move_type < 0.9:
            # Add an element
            if len(new_U) < 60:
                max_existing = max(new_U)
                new_val = random.randint(1, max_existing + 30)
                if new_val not in new_U:
                    new_U.append(new_val)
                    new_U = sorted(set(new_U))
                    valid_move = True
        
        else:
            # Remove an element
            if len(new_U) > 2:
                idx = random.randint(1, len(new_U) - 1)
                new_U.pop(idx)
                new_U = sorted(set(new_U))
                valid_move = True
        
        if not valid_move or len(new_U) < 2 or 0 not in new_U:
            continue
            
        new_score = score_set(new_U)
        
        # Acceptance
        if new_score >= current_score:
            current_U = new_U
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_set = current_U[:]
        else:
            delta = new_score - current_score
            if delta / T > -50:  # Avoid overflow
                if random.random() < pow(2.718281828, delta / T):
                    current_U = new_U
                    current_score = new_score
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        T *= cooling
        
        if no_improve_count > max_no_improve:
            # Reset temperature occasionally
            T = max(T, 0.1)
            no_improve_count = 0
    
    return best_score, best_set

def final_polish(U, deadline):
    """High-precision polish pass with tighter tolerance."""
    current_U = sorted(set(U))
    current_score = score_set(current_U)
    best_score = current_score
    best_set = current_U[:]
    
    # Try small perturbations around each element
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(1, len(current_U)):
            if time.time() > deadline - 5:
                break
            old_val = current_U[i]
            for delta in [-2, -1, 1, 2, -3, 3]:
                new_val = old_val + delta
                if new_val <= 0 or new_val in current_U:
                    continue
                new_U = current_U[:]
                new_U[i] = new_val
                new_U = sorted(set(new_U))
                if len(new_U) != len(current_U):
                    continue
                s = score_set(new_U)
                if s > current_score + 1e-10:
                    current_U = new_U
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    if current_score > best_score:
        best_score = current_score
        best_set = current_U[:]
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_set = [0, 1]
    best_score = score_set(best_set)
    
    # Phase 1: Exhaustive search for small sets to find good seeds
    for n in [3, 4, 5]:
        if time.time() > deadline - 60:
            break
        max_val = 30 if n <= 4 else 20
        s, c = exhaustive_search_small(n, max_val, deadline)
        if c and s > best_score:
            best_score = s
            best_set = c[:]
    
    # Phase 2: Greedy construction to build a larger set
    U = best_set[:]
    current_score = best_score
    target_size = min(40, max(len(U) + 10, 25))
    
    while len(U) < target_size and time.time() < deadline - 40:
        best_candidate = None
        best_candidate_score = current_score
        
        max_existing = max(U) if U else 0
        # Try candidates in a range around existing values
        candidates = []
        for val in range(1, max_existing + 40):
            if val not in U:
                candidates.append(val)
        random.shuffle(candidates)
        
        sample_size = min(30, len(candidates))
        if candidates:
            sampled = random.sample(candidates, sample_size)
            for c in sampled:
                new_U = U + [c]
                new_U = sorted(set(new_U))
                s = score_set(new_U)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = c
        
        if best_candidate is not None:
            U.append(best_candidate)
            U.sort()
            current_score = best_candidate_score
            if current_score > best_score:
                best_score = current_score
                best_set = U[:]
        else:
            break
    
    # Phase 3: Local search refinement
    if time.time() < deadline - 20:
        s, c = local_search(best_set, deadline)
        if c and s > best_score:
            best_score = s
            best_set = c[:]
    
    # Phase 4: Final polish
    if time.time() < deadline - 5:
        s, c = final_polish(best_set, deadline)
        if c and s > best_score:
            best_score = s
            best_set = c[:]
    
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
