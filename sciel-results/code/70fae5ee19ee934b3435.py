import time
import math
import numpy as np

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def compute_score_fast(U):
    """Faster scoring using pure Python set operations for small sets."""
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    n_sums = len(sums)
    n_diffs = len(diffs)
    if n_sums == 0:
        return 0
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_U = [0, 1, 3, 9]
    best_score = compute_score_fast(best_U)
    
    def try_set(U):
        nonlocal best_U, best_score
        if time.time() > deadline:
            return False
        s = compute_score_fast(U)
        if s > best_score + 1e-12:
            best_score = s
            best_U = U[:]
        return time.time() <= deadline

    # Search for sets {0, a, b, c} with 0 < a < b < c <= 40
    for a in range(1, 20):
        if time.time() > deadline:
            break
        for b in range(a + 1, 30):
            if time.time() > deadline:
                break
            for c in range(b + 1, 41):
                if not try_set([0, a, b, c]):
                    break

    # Search for sets {0, a, b, c, d} with 0 < a < b < c < d <= 30
    for a in range(1, 12):
        if time.time() > deadline:
            break
        for b in range(a + 1, 18):
            if time.time() > deadline:
                break
            for c in range(b + 1, 25):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 31):
                    if not try_set([0, a, b, c, d]):
                        break

    # Search for sets {0, a, b, c, d, e} with 0 < a < b < c < d < e <= 22
    for a in range(1, 8):
        if time.time() > deadline:
            break
        for b in range(a + 1, 13):
            if time.time() > deadline:
                break
            for c in range(b + 1, 18):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 21):
                    if time.time() > deadline:
                        break
                    for e in range(d + 1, 23):
                        if not try_set([0, a, b, c, d, e]):
                            break

    # Phase 2: Explore geometric progression structures
    for r in range(2, 15):
        if time.time() > deadline:
            break
        geo = [0]
        val = 1
        for i in range(6):
            if val > 200:
                break
            geo.append(val)
            val *= r
        if len(geo) >= 2:
            try_set(geo)
        if len(geo) >= 3:
            for i in range(1, len(geo)):
                for delta in range(-3, 4):
                    if delta == 0:
                        continue
                    candidate = geo[:]
                    new_val = candidate[i] + delta
                    if new_val <= 0:
                        continue
                    candidate[i] = new_val
                    candidate = sorted(set(candidate))
                    if len(candidate) >= 2:
                        try_set(candidate)

    # Local search: try adding, removing, or replacing elements in best_U
    if time.time() < deadline - 15:
        current_U = best_U[:]
        improved = True
        iterations = 0
        while improved and time.time() < deadline - 10 and iterations < 50:
            improved = False
            iterations += 1
            U_set = set(current_U)
            
            # Try adding a new element (wider range)
            for new_elem in range(1, 80):
                if new_elem in U_set:
                    continue
                candidate = sorted(list(U_set) + [new_elem])
                s = compute_score_fast(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate
                    current_U = candidate
                    U_set = set(current_U)
                    improved = True
                    break
                if time.time() > deadline - 10:
                    break
            
            if time.time() > deadline - 10:
                break
            
            # Try removing an element (keep 0)
            for i in range(1, len(current_U)):
                candidate = current_U[:i] + current_U[i+1:]
                if len(candidate) < 2:
                    continue
                s = compute_score_fast(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate
                    current_U = candidate
                    U_set = set(current_U)
                    improved = True
                    break
                if time.time() > deadline - 10:
                    break
            
            if time.time() > deadline - 10:
                break
            
            # Try replacing an element (keep 0)
            for i in range(1, len(current_U)):
                old_elem = current_U[i]
                candidates_to_try = set()
                for delta in range(-5, 6):
                    v = old_elem + delta
                    if v > 0:
                        candidates_to_try.add(v)
                for v in range(1, 80):
                    if v not in U_set:
                        candidates_to_try.add(v)
                
                for new_elem in sorted(candidates_to_try):
                    if new_elem == old_elem or new_elem in U_set:
                        continue
                    candidate = current_U[:]
                    candidate[i] = new_elem
                    candidate = sorted(candidate)
                    if len(set(candidate)) < len(candidate):
                        continue
                    s = compute_score_fast(candidate)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = candidate
                        current_U = candidate
                        U_set = set(current_U)
                        improved = True
                        break
                    if time.time() > deadline - 10:
                        break
                if improved:
                    break
                if time.time() > deadline - 10:
                    break
            
            # Try adding two elements at once
            if time.time() < deadline - 10:
                found = False
                for new1 in range(1, 50):
                    if new1 in U_set or found:
                        continue
                    for new2 in range(new1 + 1, 70):
                        if new2 in U_set or found:
                            continue
                        candidate = sorted(list(U_set) + [new1, new2])
                        s = compute_score_fast(candidate)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_U = candidate
                            current_U = candidate
                            U_set = set(current_U)
                            improved = True
                            found = True
                            break
                        if time.time() > deadline - 10:
                            break
    
    # Final polish: try small perturbations around the best solution
    if time.time() < deadline - 3:
        final_U = best_U[:]
        final_set = set(final_U)
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            improved = False
            # Try replacing one element with nearby values
            for i in range(1, len(final_U)):
                old = final_U[i]
                for delta in range(-4, 5):
                    new_val = old + delta
                    if new_val <= 0 or new_val in final_set:
                        continue
                    candidate = final_U[:]
                    candidate[i] = new_val
                    candidate = sorted(candidate)
                    if len(set(candidate)) < len(candidate):
                        continue
                    s = compute_score_fast(candidate)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = candidate
                        final_U = candidate
                        final_set = set(final_U)
                        improved = True
                        break
                    if time.time() > deadline - 2:
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Verify with the original compute_score
    final_score = compute_score(best_U)
    return best_U
