"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def _score(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        U = sorted(set(U) | {0})
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute U+U and U-U
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    best_set = [0, 1, 4, 9]
    best_score = _score(best_set)
    
    # Try various structured families
    candidates = []
    
    # Squares
    for k in range(2, 30):
        s = [i*i for i in range(k)]
        if 0 in s:
            candidates.append(s)
    
    # Cubes
    for k in range(2, 15):
        s = [i**3 for i in range(k)]
        candidates.append(s)
    
    # Powers of 2
    for k in range(2, 20):
        s = [2**i for i in range(k)]
        candidates.append(s)
    
    # Mixed: squares up to some point
    for k in range(3, 20):
        s = [i*i for i in range(k)]
        candidates.append(s)
    
    # Evaluate all candidates
    for s in candidates:
        if time.time() > deadline:
            break
        sc = _score(s)
        if sc > best_score:
            best_score = sc
            best_set = s[:]
    
    # Local search: try adding/removing elements
    if time.time() < deadline - 5:
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_set = set(current)
            max_elem = max(current) if current else 0
            
            # Try adding elements
            for add_val in range(0, max_elem + 50):
                if add_val in current_set:
                    continue
                new_set = sorted(current_set | {add_val})
                if time.time() > deadline:
                    break
                sc = _score(new_set)
                if sc > best_score:
                    best_score = sc
                    best_set = new_set
                    current = new_set
                    improved = True
            
            # Try removing elements (except 0)
            if len(current) > 2:
                for rem_val in list(current):
                    if rem_val == 0:
                        continue
                    new_set = sorted(current_set - {rem_val})
                    if time.time() > deadline:
                        break
                    sc = _score(new_set)
                    if sc > best_score:
                        best_score = sc
                        best_set = new_set
                        current = new_set
                        improved = True
    
    return best_set
# EVOLVE-BLOCK-END
