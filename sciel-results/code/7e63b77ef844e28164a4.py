import time
import math
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 140.0

def _elapsed():
    return time.time() - _START

def _score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
        U = sorted(U)
    n = len(U)
    if n <= 1:
        return 0.0
    M = U[-1]
    if M < 1:
        return 0.0
    
    # Compute sumset and diffset directly for small sets
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    sumset_size = len(sumset)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    
    return 1.0 + math.log(ratio) / math.log(base)

def _golomb_ruler(n):
    """Generate a Golomb ruler of length n (n marks)."""
    # Known optimal Golomb rulers for small n
    known = {
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 29],
        8: [0, 1, 4, 10, 18, 23, 29, 36],
        9: [0, 1, 4, 10, 18, 23, 29, 36, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 36, 44, 52],
    }
    if n in known:
        return known[n]
    # Fallback: greedy construction
    ruler = [0]
    for i in range(1, n):
        best_pos = -1
        best_diff = -1
        for pos in range(ruler[-1] + 1, ruler[-1] + i * i + 10):
            # Check if adding pos creates duplicate difference
            valid = True
            diffs = set()
            for j in range(len(ruler) + 1):
                if j < len(ruler):
                    for k in range(j + 1, len(ruler) + 1):
                        if k < len(ruler):
                            d = ruler[k] - ruler[j]
                        else:
                            d = pos - ruler[j]
                        if d in diffs:
                            valid = False
                            break
                        diffs.add(d)
                else:
                    for k in range(j + 1):
                        d = pos - ruler[k]
                        if d in diffs:
                            valid = False
                            break
                        diffs.add(d)
                if not valid:
                    break
            if valid:
                best_pos = pos
                break
        if best_pos == -1:
            best_pos = ruler[-1] + i
        ruler.append(best_pos)
    return ruler

def construct():
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try known Golomb rulers
    for n in range(2, 12):
        if _elapsed() > 30:
            break
        U = _golomb_ruler(n)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Exhaustive search for small sets
    for n in range(3, 8):
        if _elapsed() > 60:
            break
        max_val = 20 if n <= 5 else 30
        count = 0
        for combo in combinations(range(1, max_val + 1), n - 1):
            if _elapsed() > 70:
                break
            U = sorted([0] + list(combo))
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U
            count += 1
    
    # Local search from best
    U = list(best_U)
    cur_score = best_score
    rng = np.random.RandomState(123)
    
    for iteration in range(200):
        if _elapsed() > 120:
            break
        improved = False
        n = len(U)
        max_val = max(U) if U else 1
        
        # Try adding an element
        for _ in range(30):
            if _elapsed() > 125:
                break
            c = rng.randint(1, max_val * 3 + 10)
            if c not in U:
                trial = sorted(U + [c])
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
                    break
        
        # Try removing an element
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > 125:
                    break
                trial = U[:idx] + U[idx + 1:]
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U
                    best_score = cur_score
                    improved = True
                    break
        
        # Try swapping
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > 125:
                    break
                for _ in range(15):
                    if _elapsed() > 125:
                        break
                    new_val = rng.randint(1, max_val * 3 + 10)
                    if new_val not in U:
                        trial = U.copy()
                        trial[idx] = new_val
                        trial = sorted(trial)
                        s = _score_set(trial)
                        if s > cur_score + 1e-12:
                            cur_score = s
                            U = trial
                            best_U = U
                            best_score = cur_score
                            improved = True
                            break
                        if _elapsed() > 125:
                            break
        
        if not improved:
            # Try a random restart with perturbation
            if _elapsed() > 130:
                break
            U = list(best_U)
            if len(U) > 2:
                idx = rng.randint(1, len(U))
                U[idx] = rng.randint(1, max_val * 2 + 5)
                U = sorted(set(U))
                if 0 not in U:
                    U = [0] + U
                    U = sorted(U)
    
    return best_U
