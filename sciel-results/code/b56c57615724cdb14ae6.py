import math
import time

def _compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    
    # Compute |U+U|
    sum_set = set()
    for i in range(n):
        for j in range(i, n):
            sum_set.add(U[i] + U[j])
    sum_size = len(sum_set)
    
    # Compute |U-U|
    diff_set = set()
    for i in range(n):
        for j in range(n):
            diff_set.add(U[i] - U[j])
    diff_size = len(diff_set)
    
    if sum_size == 0 or (2 * max_u + 1) <= 1:
        return 0.0
    
    ratio = diff_size / sum_size
    if ratio <= 0:
        return 0.0
    
    score = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return score


def _greedy_sidon_set(max_size, deadline):
    """Construct a Sidon set greedily, trying to keep max element small."""
    U = [0]
    used_sums = set()
    used_diffs = set()
    
    candidate = 1
    while len(U) < max_size:
        if time.time() > deadline:
            break
        
        # Check if adding candidate maintains Sidon property
        is_sidon = True
        for u in U:
            s = u + candidate
            d1 = candidate - u
            d2 = u - candidate
            if s in used_sums or d1 in used_diffs or d2 in used_diffs:
                is_sidon = False
                break
        
        if is_sidon:
            # Add candidate
            for u in U:
                used_sums.add(u + candidate)
                used_diffs.add(candidate - u)
                used_diffs.add(u - candidate)
            U.append(candidate)
        
        candidate += 1
    
    return U


def _powers_of_two_set(n):
    """Construct set of powers of 2 including 0."""
    return [0] + [2**i for i in range(1, n)]


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Try powers of 2 for various sizes
    for n in range(3, 20):
        if time.time() > deadline:
            break
        U = _powers_of_two_set(n)
        s = _compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Try greedy Sidon set for various sizes
    for target_size in [5, 10, 15, 20, 30, 40, 50]:
        if time.time() > deadline:
            break
        U = _greedy_sidon_set(target_size, deadline)
        if len(U) >= 2:
            s = _compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U
    
    return best_U
