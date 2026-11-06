import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if subset has all unordered pair sums unique (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the largest Sidon subset size of S using backtracking with pruning."""
    S = sorted(S)
    n = len(S)
    
    # Try k from n down to 1
    for k in range(n, 0, -1):
        # Use DFS with pruning to find any Sidon subset of size k
        found = _find_sidon_subset(S, k, 0, [], set())
        if found:
            return k
    return 0

def _find_sidon_subset(S, k, start, current, used_sums):
    """DFS to find a Sidon subset of size k starting from index 'start'."""
    if len(current) == k:
        return True
    
    # Pruning: not enough elements left
    remaining = len(S) - start
    need = k - len(current)
    if remaining < need:
        return False
    
    for i in range(start, len(S)):
        x = S[i]
        # Check all pair sums involving x with existing elements
        valid = True
        new_sums = []
        for j in range(len(current)):
            s = x + current[j]
            if s in used_sums:
                valid = False
                break
            new_sums.append(s)
        if not valid:
            continue
        s = x + x
        if s in used_sums:
            continue
        new_sums.append(s)
        
        # Add x and its new sums
        for s in new_sums:
            used_sums.add(s)
        current.append(x)
        
        if _find_sidon_subset(S, k, i + 1, current, used_sums):
            return True
        
        # Backtrack
        current.pop()
        for s in new_sums:
            used_sums.discard(s)
    
    return False

def has_sidon_subset_size_k(S, k):
    """Check if S has any Sidon subset of size k."""
    S = sorted(S)
    return _find_sidon_subset(S, k, 0, [], set())

def no_sidon_size_gt_3(S):
    """Check if S has no Sidon subset of size 4 or larger."""
    return not has_sidon_subset_size_k(S, 4)

def local_search_for_max_sidon_3(deadline):
    """Local search to find a set of 12 with max Sidon size exactly 3 (no 4-Sidon)."""
    best = list(range(12))
    
    # Verify the baseline
    if not no_sidon_size_gt_3(best):
        return best  # Shouldn't happen, but safety
    
    # Try to find sets with max Sidon size <= 3 via local search
    initial_sets = [
        list(range(12)),
        list(range(0, 24, 2)),
        list(range(1, 13)),
        list(range(0, 26, 2)),
    ]
    
    for init in initial_sets:
        if time.time() > deadline:
            break
        
        current = sorted(init)
        if len(set(current)) != 12 or any(x < 0 or x > 128 for x in current):
            continue
        
        # Local search: perturb one element at a time
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                orig = current[i]
                for delta in [-3, -2, -1, 1, 2, 3]:
                    new_val = orig + delta
                    if new_val < 0 or new_val > 128:
                        continue
                    if new_val in current and new_val != orig:
                        continue
                    test = current[:]
                    test[i] = new_val
                    test = sorted(test)
                    if len(set(test)) != 12:
                        continue
                    if no_sidon_size_gt_3(test):
                        current = test
                        improved = True
                        break
                if improved:
                    break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    best_set = list(range(12))
    
    # Quick verification: check that no 4-element subset is Sidon
    has_4_sidon = False
    for combo in combinations(best_set, 4):
        if is_sidon(list(combo)):
            has_4_sidon = True
            break
    
    if not has_4_sidon:
        return best_set
    
    # Fallback: local search
    result = local_search_for_max_sidon_3(deadline)
    if no_sidon_size_gt_3(result):
        return result
    
    return best_set
