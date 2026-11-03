"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

def _score_set(U):
    """Compute the score for a finite set U containing 0 using NumPy for speed."""
    U_set = sorted(set(U))
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = U_set[-1]
    if max_u == 0:
        return 1.0
    
    U_arr = np.array(U_set, dtype=np.int64)
    
    # Compute sumset and difference set
    # For n up to ~30, n^2 is 900, which is very fast for NumPy
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n. Uses known optimal ones for small n, greedy for larger."""
    if n <= 1:
        return [0]
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    }
    if n in known:
        return known[n]
    
    # Greedy construction for larger n
    ruler = [0]
    for i in range(1, n):
        pos = ruler[-1] + 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def _simulated_annealing(U_init, deadline, max_iters=50000):
    """
    Improve set U using Simulated Annealing.
    Moves: Swap an element, Add an element, Remove an element.
    """
    U = sorted(set(U_init))
    current_score = _score_set(U)
    best_U = U[:]
    best_score = current_score
    
    T = 0.5  # Initial temperature
    cooling_rate = 0.9995
    
    max_val_limit = 200  # Upper bound for values in the set
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        U_new = U[:]
        move_type = random.choice(['swap', 'add', 'remove'])
        
        if move_type == 'swap':
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)  # Don't swap 0
                old_val = U[idx]
                # Try a random new value or a nearby one
                if random.random() < 0.7:
                    new_val = max(1, min(max_val_limit, old_val + random.randint(-10, 10)))
                else:
                    new_val = random.randint(1, max_val_limit)
                if new_val != old_val and new_val not in U:
                    U_new[idx] = new_val
                    U_new = sorted(set(U_new))
                    if len(U_new) != len(U):
                        continue
                else:
                    continue
        elif move_type == 'add':
            if len(U) < 15:  # Limit size to keep scoring fast
                new_val = random.randint(1, max_val_limit)
                if new_val not in U:
                    U_new.append(new_val)
                    U_new = sorted(U_new)
                else:
                    continue
        elif move_type == 'remove':
            if len(U) > 2:  # Keep at least 0 and one other
                idx = random.randint(1, len(U) - 1)
                U_new.pop(idx)
        
        if not U_new or 0 not in U_new:
            continue
            
        new_score = _score_set(U_new)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            U = U_new
            if current_score > best_score:
                best_score = current_score
                best_U = U[:]
        
        T *= cooling_rate
        if T < 1e-4:
            T = 1e-4
            
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # 1. Exhaustive search for small n
    # n=3: C(50, 2) = 1225
    # n=4: C(50, 3) = 19600
    # n=5: C(50, 4) = 230300 -> ~230k * 25 ops (numpy) is fast
    # n=6: C(50, 5) = 2M -> might be tight, try up to 40 for n=6
    try:
        for n in range(3, 7):
            max_val = 50 if n <= 5 else 40
            for combo in combinations(range(1, max_val + 1), n - 1):
                if time.time() > deadline - 20: # Leave time for SA
                    break
                U = [0] + list(combo)
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
            if time.time() > deadline - 20:
                break
    except Exception:
        pass

    # 2. Heuristic candidates
    candidates = []
    
    # Golomb Rulers
    for n in range(3, 15):
        if time.time() > deadline - 10:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
        
    # "Almost AP" structures: {0, 1, ..., k-2, M}
    for k in range(3, 10):
        for M in range(k, 100):
            if time.time() > deadline - 10:
                break
            U = list(range(k - 1)) + [M]
            if len(set(U)) == len(U):
                candidates.append(U)
                
    # Random sparse sets
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        n = random.randint(4, 12)
        max_val = random.randint(50, 200)
        # Ensure 0 is included, pick n-1 others
        U = [0] + sorted(random.sample(range(1, max_val + 1), n - 1))
        candidates.append(U)

    # Evaluate candidates
    for U in candidates:
        if time.time() > deadline - 5:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # 3. Simulated Annealing on the best candidate found so far
    if time.time() < deadline - 5:
        # Run SA for a while
        sa_deadline = min(deadline, time.time() + 60)
        best_U, best_score = _simulated_annealing(best_U, sa_deadline, max_iters=100000)
        
        # If time remains, try SA on other good candidates
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            # Pick a few other candidates that scored well
            other_candidates = [U for U in candidates if U != best_U]
            other_candidates.sort(key=lambda x: _score_set(x), reverse=True)
            for U in other_candidates[:3]:
                if time.time() > deadline - 5:
                    break
                sa_deadline = min(deadline, time.time() + (remaining_time / 3))
                candidate_U, candidate_score = _simulated_annealing(U, sa_deadline, max_iters=20000)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_U = candidate_U

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
