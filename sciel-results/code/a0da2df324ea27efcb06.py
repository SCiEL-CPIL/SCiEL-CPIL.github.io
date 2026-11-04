"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import math

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    maxU = max(U)
    if maxU == 0:
        return 1.0
    # Compute U+U
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    n_sum = len(sumset)
    n_diff = len(diffset)
    if n_sum == 0:
        return 0.0
    ratio = n_diff / n_sum
    if ratio <= 1:
        return 1.0
    log_num = math.log(ratio)
    log_den = math.log(2 * maxU + 1)
    if log_den <= 0:
        return 1.0
    return 1.0 + log_num / log_den

def construct():
    deadline = time.time() + 120.0  # Safety margin before 150s limit
    best_set = [0, 1, 2, 4]
    best_score = _score_set(best_set)
    
    # Try small sets by brute force search
    # Start with known good candidates
    candidates = [
        [0, 1, 2, 4],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 9],
        [0, 1, 3, 9, 27],
        [0, 1, 2, 5],
        [0, 1, 2, 5, 10],
        [0, 1, 2, 5, 10, 20],
        [0, 1, 2, 4, 7],
        [0, 1, 2, 4, 8, 15],
        [0, 1, 3, 4, 8],
        [0, 1, 3, 4, 8, 16],
    ]
    
    for cand in candidates:
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Try to improve by adding elements
    # Greedy search: try adding elements to the best set
    max_val = 100
    for _ in range(50):  # Limit iterations
        if time.time() > deadline - 30:
            break
        improved = False
        current = list(best_set)
        current_max = max(current)
        for new_elem in range(current_max + 1, max_val + 1):
            if new_elem in current:
                continue
            trial = current + [new_elem]
            s = _score_set(trial)
            if s > best_score + 1e-9:
                best_score = s
                best_set = trial
                current_max = new_elem
                improved = True
                break
        if not improved:
            # Try removing elements
            for i in range(len(best_set)):
                if best_set[i] == 0:
                    continue
                trial = best_set[:i] + best_set[i+1:]
                if len(trial) < 2:
                    continue
                s = _score_set(trial)
                if s > best_score + 1e-9:
                    best_score = s
                    best_set = trial
                    improved = True
                    break
        if not improved:
            break
    
    # Try a few more structured candidates
    structured = [
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 2, 4, 8, 16, 32, 64],
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
        [0, 1, 3, 9, 27, 81],
        [0, 1, 3, 9, 27, 81, 243],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 2, 4, 7, 14, 28],
        [0, 1, 2, 4, 7, 14, 28, 56],
    ]
    for cand in structured:
        if time.time() > deadline - 30:
            break
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    return best_set
# EVOLVE-BLOCK-END
