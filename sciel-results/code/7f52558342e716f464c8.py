"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_sidon_fast(subset):
    """Check if subset is Sidon: all unordered pair sums (i<=j) are unique."""
    n = len(subset)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size_fast(S):
    """Find the maximum size of a Sidon subset of S via branch-and-bound."""
    n = len(S)
    if n == 0:
        return 0
    
    # Sort S for better pruning
    S_sorted = sorted(S)
    
    best = 0
    
    # Branch and bound: try to build Sidon subset
    # Use the fact that if we can't extend a partial solution to size > best, prune
    def _branch(current, start_idx, current_sums):
        nonlocal best
        cur_size = len(current)
        remaining = n - start_idx
        
        # Prune: even if we add all remaining, can't beat best
        if cur_size + remaining <= best:
            return
        
        # Update best
        if cur_size > best:
            best = cur_size
        
        # Try adding each element from start_idx onward
        for i in range(start_idx, n):
            val = S_sorted[i]
            # Check if adding val keeps Sidon property
            valid = True
            for existing in current:
                s = val + existing
                if s in current_sums:
                    valid = False
                    break
            if valid:
                # Also check val+val
                if val + val in current_sums:
                    valid = False
            if valid:
                current.append(val)
                current_sums.add(val + val)
                for existing in current[:-1]:
                    current_sums.add(val + existing)
                
                _branch(current, i + 1, current_sums)
                
                # Backtrack
                current.pop()
                current_sums.discard(val + val)
                for existing in current:
                    current_sums.discard(val + existing)
    
    _branch([], 0, set())
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit
    
    # Start with arithmetic progression - known to have small Sidon subsets
    best_set = list(range(12))
    best_score = _max_sidon_size_fast(best_set)
    
    # Try various structured sets that encourage sum collisions
    candidates = []
    
    # Arithmetic progressions with different steps
    for step in range(1, 11):
        s = list(range(0, 12 * step, step))
        if len(s) == 12 and s[-1] <= 128:
            candidates.append(s)
    
    # Dense clusters: groups of close numbers
    # Two clusters of 6
    for gap in range(1, 20):
        for base in range(0, 128 - 6*gap):
            s = list(range(base, base + 6)) + list(range(base + gap, base + gap + 6))
            if len(set(s)) == 12 and max(s) <= 128:
                candidates.append(s)
                if len(candidates) > 5000:
                    break
        if len(candidates) > 5000:
            break
    
    # Three clusters of 4
    for g1 in range(1, 15):
        for g2 in range(g1 + 1, 30):
            for base in range(0, 128 - g2 - 4):
                s = (list(range(base, base + 4)) + 
                     list(range(base + g1, base + g1 + 4)) + 
                     list(range(base + g2, base + g2 + 4)))
                if len(set(s)) == 12 and max(s) <= 128:
                    candidates.append(s)
                    if len(candidates) > 8000:
                        break
            if len(candidates) > 8000:
                break
        if len(candidates) > 8000:
            break
    
    # Random sets
    random.seed(42)
    for _ in range(3000):
        s = random.sample(range(129), 12)
        candidates.append(s)
    
    # Evaluate candidates
    for s in candidates:
        if time.time() > deadline:
            break
        score = _max_sidon_size_fast(s)
        if score < best_score:
            best_score = score
            best_set = s
    
    # Local search: try swapping elements
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            best_set_copy = list(best_set)
            for i in range(12):
                if time.time() > deadline - 3:
                    break
                # Try a subset of new values for speed
                for new_val in range(0, 129, 2):  # even values first
                    if new_val in best_set_copy and best_set_copy[i] != new_val:
                        continue
                    old_val = best_set_copy[i]
                    best_set_copy[i] = new_val
                    if len(set(best_set_copy)) != 12:
                        best_set_copy[i] = old_val
                        continue
                    score = _max_sidon_size_fast(best_set_copy)
                    if score < best_score:
                        best_score = score
                        best_set = list(best_set_copy)
                        improved = True
                    best_set_copy[i] = old_val
    
    return best_set
# EVOLVE-BLOCK-END
