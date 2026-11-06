import time
import random
import numpy as np

_DEADLINE = 85.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _square_nonzero_count(coeffs):
    """Return (count, square_array) for exact square of coeffs (ascending degree)."""
    n = len(coeffs)
    if n == 0:
        return 0, [0]
    # Use numpy convolution for speed
    arr = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(arr, arr)
    # Trim trailing zeros
    while len(sq) > 1 and sq[-1] == 0:
        sq = sq[:-1]
    cnt = np.count_nonzero(sq)
    return int(cnt), sq.tolist()

def _validate(coeffs):
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    for v in coeffs:
        if not isinstance(v, int) or v < -64 or v > 64:
            return False
    nz = sum(1 for v in coeffs if v != 0)
    if nz != 12:
        return False
    return True

def _score(coeffs):
    if not _validate(coeffs):
        return None
    cnt, _ = _square_nonzero_count(coeffs)
    return cnt

def _build_coeffs(exps, coeffs):
    max_deg = max(exps)
    arr = [0] * (max_deg + 1)
    for e, c in zip(exps, coeffs):
        arr[e] = c
    return arr

def _sa_search(exps, coeffs, rng, max_iters=3000, t0=10.0, t1=0.1):
    """Simulated annealing on exponents and coefficients."""
    current_exps = list(exps)
    current_coeffs = list(coeffs)
    current_arr = _build_coeffs(current_exps, current_coeffs)
    current_score, _ = _square_nonzero_count(current_arr)
    
    best_exps = list(current_exps)
    best_coeffs = list(current_coeffs)
    best_score = current_score
    
    n = 12
    for it in range(max_iters):
        if _elapsed() > _DEADLINE:
            break
        temp = t0 * (t1 / t0) ** (it / max_iters)
        
        # Choose move type
        move_type = rng.random()
        old_exps = list(current_exps)
        old_coeffs = list(current_coeffs)
        
        if move_type < 0.4:
            # Shift one exponent
            pos = rng.randrange(n)
            delta = rng.choice([-3, -2, -1, 1, 2, 3])
            new_exp = current_exps[pos] + delta
            if 0 <= new_exp <= 40 and new_exp not in current_exps:
                current_exps[pos] = new_exp
        elif move_type < 0.7:
            # Change one coefficient
            pos = rng.randrange(n)
            old_val = current_coeffs[pos]
            new_val = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if new_val != old_val:
                current_coeffs[pos] = new_val
        elif move_type < 0.85:
            # Swap two exponents
            i, j = rng.sample(range(n), 2)
            current_exps[i], current_exps[j] = current_exps[j], current_exps[i]
        else:
            # Shift a cluster of 2-4 consecutive (in list order) exponents
            size = rng.choice([2, 3, 4])
            start = rng.randrange(n - size + 1)
            delta = rng.choice([-2, -1, 1, 2])
            valid = True
            new_exps = list(current_exps)
            for k in range(start, start + size):
                new_e = new_exps[k] + delta
                if new_e < 0 or new_e > 40:
                    valid = False
                    break
            if valid:
                # Check no collisions
                for k in range(start, start + size):
                    if new_exps[k] in [new_exps[m] for m in range(n) if m != k]:
                        valid = False
                        break
                if valid:
                    for k in range(start, start + size):
                        current_exps[k] += delta
        
        new_arr = _build_coeffs(current_exps, current_coeffs)
        new_score, _ = _square_nonzero_count(new_arr)
        
        delta_score = new_score - current_score
        if delta_score <= 0 or rng.random() < np.exp(-delta_score / temp):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_exps = list(current_exps)
                best_coeffs = list(current_coeffs)
        else:
            current_exps = old_exps
            current_coeffs = old_coeffs
    
    return _build_coeffs(best_exps, best_coeffs), best_score

def construct():
    rng = random.Random(42)
    best_score = 999
    best_coeffs = None
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Consecutive exponents
    candidates.append((list(range(12)), [1]*12))
    
    # 2. Two clusters
    candidates.append(([0,1,2,3,4,5, 20,21,22,23,24,25], [1]*6 + [1]*6))
    
    # 3. Alternating signs in two clusters
    candidates.append(([0,1,2,3,4,5, 20,21,22,23,24,25], 
                       [1,-1,1,-1,1,-1] + [1,-1,1,-1,1,-1]))
    
    # 4. Golomb-like (distinct pairwise sums)
    candidates.append(([0, 1, 2, 4, 8, 16, 20, 24, 30, 36, 38, 40], [1]*12))
    
    # 5. Symmetric around center
    candidates.append(([0, 2, 4, 6, 8, 10, 30, 32, 34, 36, 38, 40], [1]*12))
    
    # 6. Random starts
    for _ in range(15):
        exps = sorted(rng.sample(range(41), 12))
        coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
        candidates.append((exps, coeffs))
    
    # 7. Structured: try to make pairwise sums collide
    # Use exponents that are multiples of a small number
    candidates.append(([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], [1]*12))
    candidates.append(([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33], [1]*12))
    
    # Run SA on each candidate
    for exps, coeffs in candidates:
        if _elapsed() > _DEADLINE:
            break
        arr, sc = _sa_search(exps, coeffs, rng, max_iters=2000)
        if sc < best_score:
            best_score = sc
            best_coeffs = arr
    
    # Final validation
    if best_coeffs is None or not _validate(best_coeffs):
        best_coeffs = [1]*12
    
    return best_coeffs
