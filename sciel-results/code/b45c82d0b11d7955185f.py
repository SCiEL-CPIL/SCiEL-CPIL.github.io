"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
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
    deadline = start + 80  # safety margin before 90s kill
    
    best_set = None
    best_score = 12  # worst case
    
    # Try structured candidates first
    # Strategy: cluster numbers to create many sum collisions
    candidates = []
    
    # Arithmetic progressions with small step
    for start_val in range(0, 30):
        for step in [1, 2, 3]:
            cand = [start_val + i * step for i in range(12)]
            if max(cand) <= 128 and len(set(cand)) == 12:
                candidates.append(cand)
    
    # Two-cluster approach: two groups far apart
    for gap in range(20, 60):
        for spread in range(1, 8):
            left = [i * spread for i in range(6)]
            right = [gap + i * spread for i in range(6)]
            cand = left + right
            if max(cand) <= 128 and len(set(cand)) == 12:
                candidates.append(cand)
    
    # Random sets
    random.seed(42)
    for _ in range(2000):
        cand = sorted(random.sample(range(129), 12))
        candidates.append(cand)
    
    # More targeted: try to create many 4-term arithmetic progressions
    # a, a+d, a+2d, a+3d gives a+(a+3d) = (a+d)+(a+2d)
    for base in range(0, 40):
        for d in [1, 2, 3, 4, 5]:
            cand = []
            # Add several 4-APs
            for offset in range(0, 8, 2):
                for k in range(4):
                    val = base + offset + k * d
                    if val <= 128 and val not in cand:
                        cand.append(val)
            if len(cand) >= 12:
                cand = cand[:12]
                if len(set(cand)) == 12:
                    candidates.append(cand)
    
    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        score = _max_sidon_with_diagonals(cand)
        if score < best_score:
            best_score = score
            best_set = cand
            if best_score <= 3:
                break
    
    # If we have time, do more random search
    if best_score > 3 and time.time() < deadline - 5:
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            cand = sorted(random.sample(range(129), 12))
            score = _max_sidon_with_diagonals(cand)
            if score < best_score:
                best_score = score
                best_set = cand
                if best_score <= 3:
                    break
    
    if best_set is None:
        return list(range(12))
    return best_set
# EVOLVE-BLOCK-END
