import time
import itertools
import random

def _largest_sidon_subset_size(s):
    """Find the largest subset of s with all unordered pair sums unique (including diagonals)."""
    n = len(s)
    # Try subsets from largest to smallest
    for k in range(n, 0, -1):
        for combo in itertools.combinations(range(n), k):
            subset = [s[i] for i in combo]
            sums = set()
            valid = True
            for i in range(k):
                for j in range(i, k):
                    sm = subset[i] + subset[j]
                    if sm in sums:
                        valid = False
                        break
                    sums.add(sm)
                if not valid:
                    break
            if valid:
                return k
    return 0

def _evaluate_set(candidate):
    """Return the largest Sidon subset size for the given candidate set."""
    return _largest_sidon_subset_size(candidate)

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Start with a tight cluster: 12 consecutive integers
    best = list(range(12))  # [0, 1, ..., 11]
    best_score = _evaluate_set(best)
    
    # Try different starting clusters
    for start in range(0, 118):  # start such that start+11 <= 128
        candidate = list(range(start, start + 12))
        score = _evaluate_set(candidate)
        if score < best_score:
            best = candidate
            best_score = score
    
    # Local search: try swapping elements
    random.seed(42)
    current = best[:]
    current_score = best_score
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        # Try all single swaps
        for i in range(12):
            for j in range(i + 1, 12):
                for new_val in range(0, 129):
                    if new_val in current:
                        continue
                    candidate = current[:]
                    candidate[i] = new_val
                    candidate.sort()
                    score = _evaluate_set(candidate)
                    if score < current_score:
                        current = candidate
                        current_score = score
                        best = current[:]
                        best_score = current_score
                        improved = True
                        no_improve_count = 0
                        if time.time() > deadline:
                            return best
                    elif score == current_score:
                        # Accept with some probability for exploration
                        if random.random() < 0.1:
                            current = candidate
                            no_improve_count += 1
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                break
    
    return best
