"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _count_sidon_violations(s):
    """Count the number of pairs (i,j) with i<=j such that s[i]+s[j] collides with another pair.
    More collisions means harder to find large Sidon subsets."""
    n = len(s)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in sums:
                # collision
                pass
            else:
                sums[sm] = 1
    # Count total pairs minus unique sums
    total_pairs = n * (n + 1) // 2
    return total_pairs - len(sums)

def _max_sidon_with_diagonals(s):
    """Find the largest subset of s where all unordered pair sums (incl. diagonals) are unique."""
    n = len(s)
    if n <= 1:
        return n
    # Try all subsets from largest to smallest
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            elems = [s[i] for i in subset]
            sums = set()
            ok = True
            for i in range(size):
                for j in range(i, size):
                    sm = elems[i] + elems[j]
                    if sm in sums:
                        ok = False
                        break
                    sums.add(sm)
                if not ok:
                    break
            if ok:
                return size
    return 1

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill
    
    best_set = None
    best_score = 12  # worst case
    
    # Strategy: Use small range [0,25] to force sum collisions
    # Then use stochastic local search to maximize collision count
    
    # Initial candidates: dense clusters in small range
    candidates = []
    
    # Dense arithmetic progressions in [0, 25]
    for start_val in range(0, 15):
        for step in [1, 2]:
            cand = [start_val + i * step for i in range(12)]
            if max(cand) <= 25 and len(set(cand)) == 12:
                candidates.append(cand)
    
    # Random sets in small range [0, 25]
    random.seed(42)
    for _ in range(200):
        cand = sorted(random.sample(range(26), 12))
        candidates.append(cand)
    
    # Also try some in [0, 35]
    for _ in range(100):
        cand = sorted(random.sample(range(36), 12))
        candidates.append(cand)
    
    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        score = _max_sidon_with_diagonals(cand)
        if score < best_score:
            best_score = score
            best_set = list(cand)
            if best_score <= 3:
                break
    
    # Stochastic local search: swap elements to maximize collision count
    if best_score > 3 and time.time() < deadline - 15:
        current = list(best_set)
        current_score = best_score
        current_violations = _count_sidon_violations(current)
        
        for iteration in range(50000):
            if time.time() > deadline - 5:
                break
            
            # Pick a random index to replace
            idx = random.randint(0, 11)
            old_val = current[idx]
            
            # Try replacing with a random value in [0, 30]
            new_val = random.randint(0, 30)
            if new_val == old_val:
                continue
            
            # Check if new_val already in set
            if new_val in current:
                continue
            
            # Apply swap
            current[idx] = new_val
            new_violations = _count_sidon_violations(current)
            
            if new_violations >= current_violations:
                # Accept (equal or better)
                current_violations = new_violations
                # Check if this improves the actual Sidon score
                if iteration % 100 == 0 or new_violations > current_violations + 2:
                    new_score = _max_sidon_with_diagonals(current)
                    if new_score < current_score:
                        current_score = new_score
                        best_score = new_score
                        best_set = list(current)
                        if best_score <= 3:
                            break
            else:
                # Reject, revert
                current[idx] = old_val
        
        # Final check
        final_score = _max_sidon_with_diagonals(best_set)
        if final_score < best_score:
            best_score = final_score
    
    if best_set is None:
        return list(range(12))
    return best_set
# EVOLVE-BLOCK-END
