import time
from math import log, floor, ceil
import itertools
import numpy as np
import random

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    U_arr = np.array(U, dtype=np.int64)
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def score_set_fast(U):
    """Faster scoring using set operations instead of numpy for small sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    # Use Python sets for small n, numpy for larger
    if n <= 30:
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
    else:
        U_arr = np.array(U, dtype=np.int64)
        sums_arr = U_arr[:, None] + U_arr[None, :]
        sums = set(np.unique(sums_arr).tolist())
        diffs_arr = U_arr[:, None] - U_arr[None, :]
        diffs = set(np.unique(diffs_arr).tolist())
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

# Known optimal Golomb rulers (minimal diameter for n marks)
GOLOMB_RULERS = {
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 9, 11],
    6: [0, 1, 4, 10, 12, 17],
    7: [0, 1, 4, 10, 18, 23, 25],
    8: [0, 1, 4, 10, 18, 23, 29, 34],
    9: [0, 1, 4, 10, 18, 23, 29, 37, 41],
    10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
    11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58],
    12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67],
    13: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67, 73],
    14: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67, 73, 82],
    15: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67, 73, 82, 89],
    16: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67, 73, 82, 89, 99],
}

def construct():
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    s = score_set(best_set)
    if s > best_score:
        best_score = s
    
    # Phase 1: Use known optimal Golomb rulers as strong starting points
    for n in range(4, 17):
        if time.time() > deadline - 60:
            break
        if n in GOLOMB_RULERS:
            U = GOLOMB_RULERS[n]
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 2: Exhaustive search for small sets with moderate max_val
    max_val = 40
    
    for n in range(3, 6):
        if time.time() > deadline - 50:
            break
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline - 50:
                break
            U = (0,) + combo
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 3: Power-of-2 and structured constructions
    if time.time() < deadline - 30:
        # Power of 2
        for m in range(3, 12):
            U = [0] + [2**k for k in range(1, m + 1)]
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
        
        # Mixed: powers of 2 with offsets
        for m in range(3, 10):
            for offset in range(0, 4):
                U = [0] + [(2**k + offset) for k in range(1, m + 1)]
                U = sorted(set(U))
                if len(U) < 2:
                    continue
                s = score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
        
        # Fibonacci-like
        for m in range(3, 15):
            fib = [0, 1]
            for _ in range(m - 1):
                fib.append(fib[-1] + fib[-2])
            U = fib[:m + 1]
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
        
        # Power of prime constructions
        for p in [3, 5, 7]:
            for m in range(3, 12):
                U = [0] + [p**k for k in range(1, m + 1)]
                s = score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
    
    # Phase 4: Geometric progression search with ternary refinement
    if time.time() < deadline - 20:
        for m in range(2, 12):
            if time.time() > deadline - 20:
                break
            
            b_low = 1.01
            b_high = 20.0
            
            # Coarse grid search
            best_b = None
            best_s_coarse = -1
            for i in range(60):
                if time.time() > deadline - 20:
                    break
                b = b_low + (b_high - b_low) * i / 59.0
                U = [0] + [floor(b**k) for k in range(1, m + 1)]
                U = sorted(set(U))
                if len(U) < 2:
                    continue
                s = score_set_fast(U)
                if s > best_s_coarse:
                    best_s_coarse = s
                    best_b = b
            
            if best_b is not None and time.time() < deadline - 20:
                # Ternary search around best_b
                lo = max(b_low, best_b * 0.8)
                hi = min(b_high, best_b * 1.2)
                for _ in range(30):
                    if time.time() > deadline - 20:
                        break
                    m1 = lo + (hi - lo) / 3
                    m2 = hi - (hi - lo) / 3
                    
                    U1 = [0] + [floor(m1**k) for k in range(1, m + 1)]
                    U1 = sorted(set(U1))
                    s1 = score_set_fast(U1) if len(U1) >= 2 else -1
                    
                    U2 = [0] + [floor(m2**k) for k in range(1, m + 1)]
                    U2 = sorted(set(U2))
                    s2 = score_set_fast(U2) if len(U2) >= 2 else -1
                    
                    if s1 < s2:
                        lo = m1
                    else:
                        hi = m2
                
                b_opt = (lo + hi) / 2
                U = [0] + [floor(b_opt**k) for k in range(1, m + 1)]
                U = sorted(set(U))
                if len(U) >= 2:
                    s = score_set_fast(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)
    
    # Phase 5: Local perturbation search around best set - expanded
    if time.time() < deadline - 10:
        current_set = set(best_set)
        
        # Try adding elements (wider range)
        for new_elem in range(1, 300):
            if time.time() > deadline - 10:
                break
            if new_elem in current_set:
                continue
            new_set = current_set | {new_elem}
            s = score_set_fast(list(new_set))
            if s > best_score:
                best_score = s
                best_set = list(new_set)
                current_set = new_set
        
        # Try removing elements
        if len(best_set) > 2:
            for elem in list(best_set):
                if time.time() > deadline - 10:
                    break
                if elem == 0:
                    continue
                new_set = set(best_set) - {elem}
                if len(new_set) < 2:
                    continue
                s = score_set_fast(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = list(new_set)
        
        # Try shifting elements by larger amounts
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            improved = False
            for i, elem in enumerate(list(best_set)):
                if time.time() > deadline - 10:
                    break
                if elem == 0:
                    continue
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    new_elem = elem + delta
                    if new_elem < 0 or new_elem in best_set:
                        continue
                    new_set = list(best_set)
                    new_set[i] = new_elem
                    new_set = sorted(set(new_set))
                    if 0 not in new_set or len(new_set) < 2:
                        continue
                    s = score_set_fast(new_set)
                    if s > best_score:
                        best_score = s
                        best_set = new_set
                        improved = True
                        break
            if not improved:
                break
        
        # Scale the set (multiply by small integers)
        for scale in [2, 3, 5]:
            if time.time() > deadline - 10:
                break
            scaled = [0] + [x * scale for x in best_set if x > 0]
            s = score_set_fast(scaled)
            if s > best_score:
                best_score = s
                best_set = scaled
    
    # Phase 6: Simulated annealing for final polish
    if time.time() < deadline - 3:
        current = sorted(set(best_set))
        current_score = score_set_fast(current)
        if current_score < 0:
            current_score = best_score
            current = best_set[:]
        
        best_sa = current[:]
        best_sa_score = current_score
        
        T = 0.05  # Initial temperature
        cooling = 0.999
        start_time = time.time()
        
        max_iter = 5000
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            n = len(current)
            r = random.random()
            if r < 0.4:
                move_type = 'perturb'
            elif r < 0.6:
                move_type = 'add'
            elif r < 0.75:
                move_type = 'remove'
            elif r < 0.9:
                move_type = 'swap'
            else:
                move_type = 'scale'
            
            new_set = current[:]
            
            if move_type == 'perturb' and n > 1:
                i = random.randint(1, n-1)
                old_val = current[i]
                max_val = max(current)
                scale_choice = random.random()
                if scale_choice < 0.5:
                    scale = max(1, int(max_val * 0.05))
                elif scale_choice < 0.8:
                    scale = max(1, int(max_val * 0.15))
                else:
                    scale = max(1, int(max_val * 0.3))
                new_val = old_val + random.randint(-scale, scale)
                if new_val <= 0:
                    new_val = old_val + random.randint(1, scale)
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
            
            elif move_type == 'add' and n < 25:
                max_val = max(current)
                if random.random() < 0.4:
                    new_val = max_val + random.randint(1, max(2, int(max_val * 0.2)))
                elif random.random() < 0.7:
                    new_val = random.randint(1, max_val)
                else:
                    new_val = random.randint(max_val + 1, max_val * 3)
                new_set.append(new_val)
                new_set = sorted(set(new_set))
                if len(new_set) == n:
                    continue
            
            elif move_type == 'remove' and n > 4:
                i = random.randint(1, n-1)
                new_set.pop(i)
            
            elif move_type == 'swap' and n > 2:
                i, j = random.sample(range(1, n), 2)
                new_set[i], new_set[j] = new_set[j], new_set[i]
                new_set = sorted(new_set)
            
            elif move_type == 'scale' and n > 1:
                factor = random.choice([0.8, 0.9, 1.1, 1.2, 1.5])
                new_set = [0] + [max(1, int(v * factor)) for v in current[1:]]
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
            
            else:
                continue
                
            new_score = score_set_fast(new_set)
            
            if new_score < 0:
                if random.random() < T:
                    current = new_set
                    current_score = new_score
            else:
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    if current_score > best_sa_score:
                        best_sa_score = current_score
                        best_sa = current[:]
                else:
                    delta = current_score - new_score
                    if delta < 0:
                        delta = 0
                    if random.random() < np.exp(-delta / max(T, 1e-10)):
                        current = new_set
                        current_score = new_score
            
            T *= cooling
        
        if best_sa_score > best_score:
            best_score = best_sa_score
            best_set = best_sa
    
    return best_set
