"""Ambient integer sets with small largest Sidon subset."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if a subset has all unordered pair sums unique including diagonals."""
    subset = sorted(subset)
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient):
    """Find the largest Sidon subset size of the ambient set."""
    ambient = sorted(ambient)
    n = len(ambient)
    # Try sizes from large to small
    for k in range(n, 0, -1):
        for subset in combinations(ambient, k):
            if _is_sidon(subset):
                return k
    return 0

def construct():
    deadline = time.time() + 55  # safety margin
    
    # Start with a promising initial set
    # Sets with many additive relations tend to have smaller Sidon subsets
    # Try arithmetic progressions and structured sets first
    
    best_set = list(range(12))
    best_score = _max_sidon_size(best_set)
    
    # Try some structured sets
    candidates = [
        list(range(0, 12, 1)),           # 0..11
        list(range(0, 24, 2)),           # even 0..22
        list(range(0, 26, 2)),           # even 0..24
        list(range(0, 28, 2)),           # even 0..26
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
    ]
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        score = _max_sidon_size(cand)
        if score < best_score:
            best_score = score
            best_set = list(cand)
            if best_score <= 4:
                break
    
    # Random local search
    if best_score > 4:
        current = list(best_set)
        current_score = best_score
        
        for iteration in range(100000):
            if time.time() > deadline:
                break
            
            # Random mutation: change one element
            idx = random.randint(0, 11)
            old_val = current[idx]
            new_val = random.randint(0, 128)
            
            # Ensure distinctness
            if new_val in [x for i, x in enumerate(current) if i != idx]:
                continue
            
            current[idx] = new_val
            score = _max_sidon_size(current)
            
            if score < current_score:
                current_score = score
                best_set = list(current)
                best_score = score
                if best_score <= 4:
                    break
            else:
                # Revert if not better
                current[idx] = old_val
    
    return best_set
# EVOLVE-BLOCK-END
