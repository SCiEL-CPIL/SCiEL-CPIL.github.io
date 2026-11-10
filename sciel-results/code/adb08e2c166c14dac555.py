"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def _max_sidon_size(s):
    """Compute the largest subset of s where all unordered pair sums (incl. diagonals) are unique."""
    n = len(s)
    # Try all subsets from largest to smallest
    for size in range(n, 1, -1):
        for subset in itertools.combinations(s, size):
            sums = set()
            valid = True
            for i in range(size):
                for j in range(i, size):
                    sm = subset[i] + subset[j]
                    if sm in sums:
                        valid = False
                        break
                    sums.add(sm)
                if not valid:
                    break
            if valid:
                return size
    return 2  # any 2 distinct elements form a Sidon set

def _evaluate(s):
    return _max_sidon_size(s)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    # Initial candidate: try a few structured sets
    candidates = []
    
    # Dense set
    candidates.append(list(range(12)))
    
    # Set with many small values
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Set with arithmetic progression
    candidates.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    
    # Mixed set
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])
    
    # Evaluate initial candidates
    best = candidates[0]
    best_score = _evaluate(best)
    for c in candidates[1:]:
        sc = _evaluate(c)
        if sc < best_score:
            best_score = sc
            best = c
    
    # Local search: try swapping elements
    random.seed(42)
    current = list(best)
    current_score = best_score
    
    while time.time() < deadline:
        improved = False
        # Try single-element replacements
        for i in range(12):
            old_val = current[i]
            for new_val in range(129):
                if new_val == old_val:
                    continue
                if new_val in current:
                    continue
                trial = current.copy()
                trial[i] = new_val
                trial_score = _evaluate(trial)
                if trial_score < current_score:
                    current = trial
                    current_score = trial_score
                    improved = True
                    break
            if time.time() > deadline:
                break
        if not improved:
            # Try random perturbation
            for _ in range(100):
                if time.time() > deadline:
                    break
                i = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val in current:
                    continue
                trial = current.copy()
                trial[i] = new_val
                trial_score = _evaluate(trial)
                if trial_score < current_score:
                    current = trial
                    current_score = trial_score
                    improved = True
                    break
        if not improved:
            break
    
    return sorted(current)
# EVOLVE-BLOCK-END
