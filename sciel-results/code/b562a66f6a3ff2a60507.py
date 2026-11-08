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
    """Faster scoring using vectorized numpy for medium/large sets, pure Python for small."""
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    n = len(U)
    if n <= 15:
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        n_sums = len(sums)
        n_diffs = len(diffs)
    else:
        U_arr = np.array(U, dtype=np.int64)
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        n_sums = len(np.unique(sums))
        n_diffs = len(np.unique(diffs))
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
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
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

    # Phase 1: Exhaustive search for small sets
    # {0, a, b, c} with wider range
    for a in range(1, 25):
        if time.time() > deadline:
            break
        for b in range(a + 1, 40):
            if time.time() > deadline:
                break
            for c in range(b + 1, 60):
                if not try_set([0, a, b, c]):
                    break

    # {0, a, b, c, d}
    for a in range(1, 15):
        if time.time() > deadline:
            break
        for b in range(a + 1, 22):
            if time.time() > deadline:
                break
            for c in range(b + 1, 30):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 40):
                    if not try_set([0, a, b, c, d]):
                        break

    # {0, a, b, c, d, e}
    for a in range(1, 10):
        if time.time() > deadline:
            break
        for b in range(a + 1, 16):
            if time.time() > deadline:
                break
            for c in range(b + 1, 22):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 28):
                    if time.time() > deadline:
                        break
                    for e in range(d + 1, 35):
                        if not try_set([0, a, b, c, d, e]):
                            break

    # Phase 2: Explore geometric progression structures
    for r in range(2, 20):
        if time.time() > deadline:
            break
        geo = [0]
        val = 1
        for i in range(7):
            if val > 300:
                break
            geo.append(val)
            val *= r
        if len(geo) >= 2:
            try_set(geo)
        if len(geo) >= 3:
            for i in range(1, len(geo)):
                for delta in range(-4, 5):
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
            # Try replacing one element with a non-geometric value
            for i in range(1, len(geo)):
                for new_val in range(1, 50):
                    if new_val in geo:
                        continue
                    candidate = geo[:]
                    candidate[i] = new_val
                    candidate = sorted(set(candidate))
                    if len(candidate) >= 2:
                        try_set(candidate)
                    if time.time() > deadline:
                        break

    # Phase 3: Local search with multiple strategies
    if time.time() < deadline - 15:
        current_U = best_U[:]
        improved = True
        iterations = 0
        while improved and time.time() < deadline - 10 and iterations < 120:
            improved = False
            iterations += 1
            U_set = set(current_U)
            
            # Strategy 1: Try adding a new element (wider range)
            for new_elem in range(1, 200):
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
            
            # Strategy 2: Try removing an element
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
            
            # Strategy 3: Try replacing an element with nearby values first, then wider
            for i in range(1, len(current_U)):
                old_elem = current_U[i]
                # First try nearby values (wider range)
                for delta in range(-15, 16):
                    new_elem = old_elem + delta
                    if new_elem <= 0 or new_elem in U_set or new_elem == old_elem:
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
                
                # Then try wider range
                for new_elem in range(1, 200):
                    if new_elem in U_set or new_elem == old_elem:
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
            
            # Strategy 4: Try adding two elements at once
            if time.time() < deadline - 10:
                found = False
                for new1 in range(1, 80):
                    if new1 in U_set or found:
                        continue
                    for new2 in range(new1 + 1, 150):
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
    
    # Phase 4: High-precision final polish
    if time.time() < deadline - 5:
        final_U = best_U[:]
        final_set = set(final_U)
        polish_iters = 0
        while polish_iters < 500 and time.time() < deadline - 2:
            polish_iters += 1
            improved = False
            
            # Try replacing one element with a range of nearby values
            for i in range(1, len(final_U)):
                old = final_U[i]
                # Wider perturbation range for final polish
                for delta in range(-20, 21):
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
            
            # Try swapping two elements
            if not improved and len(final_U) >= 3:
                for i in range(1, len(final_U)):
                    for j in range(i + 1, len(final_U)):
                        for delta_i in range(-8, 9):
                            new_i = final_U[i] + delta_i
                            if new_i <= 0 or new_i in final_set:
                                continue
                            for delta_j in range(-8, 9):
                                new_j = final_U[j] + delta_j
                                if new_j <= 0 or new_j in final_set or new_j == new_i:
                                    continue
                                candidate = final_U[:]
                                candidate[i] = new_i
                                candidate[j] = new_j
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
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
    
    # Verify with the original compute_score
    final_score = compute_score(best_U)
    return best_U
