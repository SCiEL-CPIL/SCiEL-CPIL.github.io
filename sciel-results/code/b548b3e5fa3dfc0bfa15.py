import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 6.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _is_golomb(U):
    """Check if U is a Golomb ruler."""
    diffs = set()
    n = len(U)
    for i in range(n):
        for j in range(i+1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _score(U):
    """Compute the score."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # For Golomb rulers, we can use the formula directly if we know it's Golomb
    # But to be safe and general, we compute |U-U| and |U+U|
    # However, if we are searching Golomb rulers, we can just use the formula
    # |U-U| = n(n-1)+1
    # |U+U| = n(n+1)/2
    # But only if it IS a Golomb ruler.
    # If it's not, the score might be different.
    # Given the insight, we focus on Golomb rulers.
    
    n = len(U)
    if _is_golomb(U):
        diff_count = n * (n - 1) + 1
        sum_count = n * (n + 1) // 2
    else:
        # Fallback: exact computation
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        sum_count = len(sumset)
        diff_count = len(diffset)
        
    if sum_count == 0:
        return 0.0
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(ratio, 2) / denom

def _optimize_golomb(n, time_budget=5.0):
    """Find a short Golomb ruler with n marks."""
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    # Known optimal Golomb rulers for small n
    known_optimal = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 48, 55],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 48, 62, 72],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 48, 62, 79, 85],
        13: [0, 1, 4, 10, 18, 23, 29, 37, 48, 62, 79, 99, 106],
        14: [0, 1, 4, 10, 18, 23, 29, 37, 48, 62, 79, 99, 121, 129],
        15: [0, 1, 4, 10, 18, 23, 29, 37, 48, 62, 79, 99, 121, 148, 153],
    }
    
    if n in known_optimal:
        U = list(known_optimal[n])
    else:
        # Greedy construction
        U = [0]
        for i in range(1, n):
            found = False
            # Start from a reasonable lower bound
            lower = max(1, U[-1] + 1)
            upper = lower + i * i + 20
            for pos in range(lower, upper):
                cand = U + [pos]
                if _is_golomb(cand):
                    U.append(pos)
                    found = True
                    break
            if not found:
                U.append(U[-1] + i * i + 20)
    
    best_U = list(U)
    best_len = max(U)
    start = time.time()
    rng = random.Random(n * 1000 + 7)
    
    # Precompute diff set for faster checking
    def _diff_set(arr):
        s = set()
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                s.add(arr[j] - arr[i])
        return s
    
    U_set = set(U)
    diff_set = _diff_set(U)
    
    while time.time() - start < time_budget and _remaining() > 10:
        improved = False
        
        # Try moving individual marks
        for i in range(1, n):
            orig = U[i]
            # Try nearby positions
            for d in range(-30, 31):
                if d == 0:
                    continue
                nv = orig + d
                if nv < 0 or nv == 0:
                    continue
                if nv in U_set:
                    continue
                
                # Check if new position creates duplicate differences
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    diff = abs(nv - U[j])
                    # Remove old differences involving position i
                    old_diff = abs(orig - U[j])
                    # Check if diff already exists without the old_diff contribution
                    # Simplified: just rebuild check
                    pass
                
                # Simpler approach: build new array and check
                new_U = U[:i] + [nv] + U[i+1:]
                new_U.sort()
                if _is_golomb(new_U):
                    new_len = max(new_U)
                    if new_len < best_len:
                        best_len = new_len
                        U = new_U
                        U_set = set(U)
                        improved = True
                        break
            if improved:
                break
        
        # Try swapping two marks
        if not improved:
            for i in range(1, n):
                for j in range(i+1, n):
                    U[i], U[j] = U[j], U[i]
                    new_U = sorted(U)
                    if _is_golomb(new_U):
                        new_len = max(new_U)
                        if new_len < best_len:
                            best_len = new_len
                            U = new_U
                            U_set = set(U)
                            improved = True
                            break
                    else:
                        U[i], U[j] = U[j], U[i]
                if improved:
                    break
        
        if not improved:
            # Perturb: move a random mark by a larger amount
            if rng.random() < 0.3:
                i = rng.randint(1, n-1)
                orig = U[i]
                delta = rng.randint(-50, 50)
                nv = orig + delta
                if nv > 0 and nv not in U_set:
                    U[i] = nv
                    U.sort()
                    if _is_golomb(U):
                        new_len = max(U)
                        if new_len < best_len:
                            best_len = new_len
                            U_set = set(U)
                    else:
                        U[i] = orig
                        U.sort()
    
    return U

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)
    
    best_U = [0, 1, 3]
    best_s = _score(best_U)
    
    # Try various n, allocating more time to larger n where improvements are likely
    # Score roughly scales with n and log(1/M), so larger n can be better
    for n in range(3, 20):
        if _remaining() < 15:
            break
        # Allocate time: more for larger n
        if n <= 10:
            alloc = min(3.0, _remaining() - 10)
        elif n <= 15:
            alloc = min(5.0, _remaining() - 10)
        else:
            alloc = min(8.0, _remaining() - 10)
        if alloc < 1.0:
            break
        U = _optimize_golomb(n, time_budget=alloc)
        s = _score(U)
        if s > best_s:
            best_s = s
            best_U = U
            
    # Final polish: try to improve the best solution found
    if _remaining() > 20:
        n_best = len(best_U)
        # Try a few more iterations on the best n
        for trial in range(3):
            if _remaining() < 10:
                break
            U = _optimize_golomb(n_best, time_budget=min(5.0, _remaining() - 5))
            s = _score(U)
            if s > best_s:
                best_s = s
                best_U = U
    
    return best_U
