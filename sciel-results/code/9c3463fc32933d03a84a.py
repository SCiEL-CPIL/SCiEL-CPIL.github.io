"""Optimized program: Hybrid exhaustive + greedy + SA with FFT-based score evaluation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import log, exp

def _compute_score_fft(U, fft_cache=None):
    """Compute the score for a given set U using FFT for speed."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0 or 0 not in U:
        return 0.0
    max_u = int(U[-1])
    if max_u == 0:
        return 1.0
    
    size = 2 * max_u + 1
    indicator = np.zeros(size, dtype=np.float64)
    indicator[U] = 1.0
    
    fft_size = 1
    while fft_size < 2 * size:
        fft_size *= 2
    
    f = np.fft.rfft(indicator, n=fft_size)
    conv = np.fft.irfft(f * f, n=fft_size)
    sumset_mask = conv[:2 * max_u + 1] > 0.5
    len_sum = int(np.sum(sumset_mask))
    
    g = np.fft.rfft(indicator, n=fft_size)
    autocorr = np.fft.irfft(g * np.conj(g), n=fft_size)
    diff_counts = np.zeros(2 * max_u + 1, dtype=np.float64)
    diff_counts[max_u] = autocorr[0]
    for d in range(1, max_u + 1):
        diff_counts[max_u + d] = autocorr[d]
        diff_counts[max_u - d] = autocorr[d]
    
    len_diff = int(np.sum(diff_counts > 0.5))
    
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
        
    return 1 + log(ratio) / denominator

def _compute_score_direct(U):
    """Direct computation for small sets (exact, no FFT overhead)."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0 or 0 not in U:
        return 0.0
    max_u = int(U[-1])
    if max_u == 0:
        return 1.0
    
    sumset = np.unique(U[:, None] + U[None, :])
    diffset = np.unique(U[:, None] - U[None, :])
    len_sum = len(sumset)
    len_diff = len(diffset)
    
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
        
    return 1 + log(ratio) / denominator

def _score(U):
    """Choose the right scoring method based on set size."""
    n = len(set(U))
    if n <= 20:
        return _compute_score_direct(U)
    else:
        return _compute_score_fft(U)

def _greedy_construct(max_val, max_size, timeout=5.0):
    """Greedy constructive algorithm: iteratively add element that maximizes score gain."""
    start_time = time.time()
    current = [0]
    current_score = _score(current)
    
    while time.time() - start_time < timeout and len(current) < max_size:
        best_elem = None
        best_score = current_score
        best_gain = -1e10
        
        # Candidate pool: structured + random
        candidates = set()
        max_u = max(current) if current else 0
        n = len(current)
        
        # Structured candidates based on current set
        for x in current:
            for off in [1, 2, 3, 5, 8, 13, 21]:
                c = x + off
                if 0 <= c <= max_val and c not in current:
                    candidates.add(c)
        # Sums and differences
        for a in current:
            for b in current:
                s = a + b
                if 0 <= s <= max_val and s not in current:
                    candidates.add(s)
                d = abs(a - b)
                if 0 < d <= max_val and d not in current:
                    candidates.add(d)
        # Geometric-like
        if max_u > 0:
            for mult in [2, 3]:
                c = max_u * mult
                if c <= max_val:
                    candidates.add(c)
        # Random candidates
        for _ in range(20):
            candidates.add(random.randint(1, max_val))
        
        # Evaluate each candidate
        for cand in candidates:
            if cand in current:
                continue
            new_set = sorted(current + [cand])
            new_score = _score(new_set)
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_score = new_score
                best_elem = cand
        
        if best_elem is not None and best_gain > -1e-10:
            current.append(best_elem)
            current.sort()
            current_score = best_score
        else:
            break
    
    return current, current_score

def _sa_search(initial_set, max_val, timeout=15.0):
    """Simulated Annealing to find a good set."""
    start_time = time.time()
    
    current = sorted(set(initial_set))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _score(current)
    
    best = list(current)
    best_score = current_score
    
    T = 0.05
    T_min = 1e-6
    cooling = 0.9997
    
    while time.time() - start_time < timeout:
        if T < T_min:
            break
        
        move_type = random.random()
        
        if move_type < 0.35:
            new_elem = random.randint(0, max_val)
            if new_elem not in current:
                new_set = current + [new_elem]
            else:
                continue
        elif move_type < 0.6:
            removable = [x for x in current if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            new_set = [x for x in current if x != rem]
        elif move_type < 0.8:
            removable = [x for x in current if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            add = random.randint(0, max_val)
            new_set = [x for x in current if x != rem]
            if add not in new_set:
                new_set.append(add)
        else:
            if len(current) <= 1:
                continue
            idx = random.randint(0, len(current) - 1)
            if idx == 0 and current[0] == 0 and len(current) > 1:
                idx = random.randint(1, len(current) - 1)
            old_val = current[idx]
            delta = random.choice([-3, -2, -1, 1, 2, 3, 5, -5, 8, -8])
            new_val = old_val + delta
            if new_val < 0 or new_val > max_val:
                new_val = random.randint(0, max_val)
            new_set = list(current)
            new_set[idx] = new_val
            if len(set(new_set)) != len(new_set):
                continue
        
        new_set = sorted(set(new_set))
        if 0 not in new_set or len(new_set) < 2:
            continue
        
        new_score = _score(new_set)
        
        if new_score >= current_score or random.random() < exp((new_score - current_score) / T):
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        
        T *= cooling
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_result = [0, 1, 2, 4]
    best_score = _score(best_result)
    
    # Phase 1: Exhaustive search for small sets (n=3 to 6) in bounded range
    # This finds the optimal small structures that give high ratio with small denominator
    if time.time() < deadline - 30:
        for max_val in [15, 25, 40]:
            if time.time() > deadline - 20:
                break
            for n in range(3, 7):
                if time.time() > deadline - 15:
                    break
                for combo in combinations(range(1, max_val + 1), n - 1):
                    U = [0] + list(combo)
                    score = _compute_score_direct(U)
                    if score > best_score:
                        best_score = score
                        best_result = U
    
    # Phase 2: Greedy constructive algorithm from scratch
    if time.time() < deadline - 20:
        for max_val in [30, 50, 80]:
            if time.time() > deadline - 15:
                break
            greedy_set, greedy_score = _greedy_construct(max_val, max_size=15, timeout=min(5.0, deadline - time.time() - 5))
            if greedy_score > best_score:
                best_score = greedy_score
                best_result = greedy_set
    
    # Phase 3: SA refinement from best result and from structured initializations
    initial_sets = [
        list(best_result),
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 5, 10, 20, 40],
        [0, 2, 5, 11, 23, 47],
    ]
    
    for init in initial_sets:
        if time.time() > deadline - 10:
            break
        max_u_init = max(init) if init else 0
        max_val = min(max_u_init * 3 + 20, 150)
        sa_timeout = min(10.0, max(2.0, deadline - time.time() - 5))
        result, score = _sa_search(init, max_val, timeout=sa_timeout)
        if score > best_score:
            best_score = score
            best_result = result
    
    # Phase 4: Final polish - tight local search on best result
    if time.time() < deadline - 5:
        max_u_best = max(best_result) if best_result else 0
        max_val_final = min(max_u_best + 30, 200)
        result, score = _sa_search(best_result, max_val_final, timeout=min(8.0, deadline - time.time() - 2))
        if score > best_score:
            best_score = score
            best_result = result
    
    # Final validation
    final_score = _score(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 2, 4]
    
    return best_result
# EVOLVE-BLOCK-END
