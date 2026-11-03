import time
import random
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
        
    sums = set()
    diffs = set()
    
    # Efficient computation of sum and difference sets
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
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    best_score = -1.0
    best_set = [0, 1, 4, 9]
    
    # Initial best from manual analysis
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)
        
    # 1. Grid search for small sets (n=3 to n=6)
    # For n=3: {0, 1, B}
    for B in range(2, 100):
        U = [0, 1, B]
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
            
    # For n=4: {0, 1, B, C}
    # B is likely around 3-10, C around 5-50
    for B in range(2, 15):
        for C in range(B + 1, 60):
            if time.time() > deadline - 20:
                break
            U = [0, 1, B, C]
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    # For n=5: {0, 1, B, C, D}
    # This is more expensive. Use random sampling around known good structures.
    # Known good: {0, 1, 4, 9, 16} was bad.
    # Try {0, 1, B, C, D} with B, C, D from a geometric-ish progression.
    random.seed(42)
    for _ in range(2000):
        if time.time() > deadline - 20:
            break
        B = random.randint(2, 10)
        C = random.randint(B + 1, 30)
        D = random.randint(C + 1, 100)
        U = [0, 1, B, C, D]
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
            
    # 2. Local search refinement on the best set found
    U = list(best_set)
    current_score = best_score
    
    # Define a neighborhood move: add, remove, or replace an element
    # Since n is small, we can be aggressive.
    max_val_limit = 200
    
    for _ in range(5000):
        if time.time() > deadline - 5:
            break
            
        move_type = random.random()
        new_U = U[:]
        improved = False
        
        if move_type < 0.4:
            # Add an element
            if len(U) < 8:
                # Pick a random candidate from a wide range
                cand = random.randint(1, max_val_limit)
                if cand not in U:
                    new_U.append(cand)
                    new_U.sort()
                    s = score_set(new_U)
                    if s > current_score:
                        U = new_U
                        current_score = s
                        improved = True
        elif move_type < 0.7:
            # Remove an element (keep 0)
            if len(U) > 3:
                idx = random.randint(1, len(U) - 1)
                new_U.pop(idx)
                s = score_set(new_U)
                if s > current_score:
                    U = new_U
                    current_score = s
                    improved = True
        else:
            # Replace an element
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Try a few random replacements
                for _ in range(5):
                    cand = random.randint(1, max_val_limit)
                    if cand == old_val or cand in U:
                        continue
                    new_U[idx] = cand
                    new_U.sort()
                    s = score_set(new_U)
                    if s > current_score:
                        U = new_U
                        current_score = s
                        improved = True
                        break
                        
        if not improved:
            # If no improvement, maybe try a random restart if stuck?
            # Or just continue.
            pass

    return U

# EVOLVE-BLOCK-END
