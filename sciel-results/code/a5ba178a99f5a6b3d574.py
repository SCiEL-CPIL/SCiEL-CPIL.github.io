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

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    best_U = [0, 1, 5, 7]
    best_score = compute_score(best_U)
    
    def try_set(U):
        nonlocal best_U, best_score
        if time.time() > deadline:
            return False
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        return time.time() <= deadline

    # Search for sets {0, a, b, c} with 0 < a < b < c <= 35
    for a in range(1, 18):
        if time.time() > deadline:
            break
        for b in range(a + 1, 28):
            if time.time() > deadline:
                break
            for c in range(b + 1, 36):
                if not try_set([0, a, b, c]):
                    break

    # Search for sets {0, a, b, c, d} with 0 < a < b < c < d <= 25
    for a in range(1, 10):
        if time.time() > deadline:
            break
        for b in range(a + 1, 15):
            if time.time() > deadline:
                break
            for c in range(b + 1, 21):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 26):
                    if not try_set([0, a, b, c, d]):
                        break

    # Search for sets {0, a, b, c, d, e} with 0 < a < b < c < d < e <= 18
    for a in range(1, 7):
        if time.time() > deadline:
            break
        for b in range(a + 1, 11):
            if time.time() > deadline:
                break
            for c in range(b + 1, 15):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 17):
                    if time.time() > deadline:
                        break
                    for e in range(d + 1, 19):
                        if not try_set([0, a, b, c, d, e]):
                            break

    # Local search: try adding, removing, or replacing elements in best_U
    if time.time() < deadline - 10:
        current_U = best_U[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            U_set = set(current_U)
            
            # Try adding a new element (wider range)
            for new_elem in range(1, 60):
                if new_elem in U_set:
                    continue
                candidate = sorted(list(U_set) + [new_elem])
                s = compute_score(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate
                    current_U = candidate
                    U_set = set(current_U)
                    improved = True
                    break
                if time.time() > deadline - 5:
                    break
            
            if time.time() > deadline - 5:
                break
            
            # Try removing an element (keep 0)
            for i in range(1, len(current_U)):
                candidate = current_U[:i] + current_U[i+1:]
                if len(candidate) < 2:
                    continue
                s = compute_score(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate
                    current_U = candidate
                    U_set = set(current_U)
                    improved = True
                    break
                if time.time() > deadline - 5:
                    break
            
            if time.time() > deadline - 5:
                break
            
            # Try replacing an element (keep 0) - wider range
            for i in range(1, len(current_U)):
                old_elem = current_U[i]
                for new_elem in range(1, 60):
                    if new_elem == old_elem or new_elem in U_set:
                        continue
                    candidate = current_U[:]
                    candidate[i] = new_elem
                    candidate = sorted(candidate)
                    s = compute_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_U = candidate
                        current_U = candidate
                        U_set = set(current_U)
                        improved = True
                        break
                    if time.time() > deadline - 5:
                        break
                if improved:
                    break
                if time.time() > deadline - 5:
                    break
            
            # Try swapping two elements (keep 0)
            if len(current_U) >= 3:
                for i in range(1, len(current_U)):
                    for j in range(i + 1, len(current_U)):
                        for new_i in range(1, 60):
                            if new_i in U_set:
                                continue
                            candidate = current_U[:]
                            candidate[i] = new_i
                            candidate = sorted(candidate)
                            if len(set(candidate)) < len(candidate):
                                continue
                            s = compute_score(candidate)
                            if s > best_score:
                                best_score = s
                                best_U = candidate
                                current_U = candidate
                                U_set = set(current_U)
                                improved = True
                                break
                            if time.time() > deadline - 5:
                                break
                        if improved:
                            break
                    if improved or time.time() > deadline - 5:
                        break
            
            # Try adding two elements at once
            if time.time() < deadline - 5:
                for new1 in range(1, 40):
                    if new1 in U_set:
                        continue
                    for new2 in range(new1 + 1, 50):
                        if new2 in U_set:
                            continue
                        candidate = sorted(list(U_set) + [new1, new2])
                        s = compute_score(candidate)
                        if s > best_score:
                            best_score = s
                            best_U = candidate
                            current_U = candidate
                            U_set = set(current_U)
                            improved = True
                            break
                        if time.time() > deadline - 5:
                            break
                    if improved or time.time() > deadline - 5:
                        break
    
    # Final polish: try small perturbations around the best solution
    if time.time() < deadline - 2:
        final_U = best_U[:]
        final_set = set(final_U)
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            improved = False
            # Try replacing one element with nearby values
            for i in range(1, len(final_U)):
                old = final_U[i]
                for delta in range(-3, 4):
                    new_val = old + delta
                    if new_val <= 0 or new_val in final_set:
                        continue
                    candidate = final_U[:]
                    candidate[i] = new_val
                    candidate = sorted(candidate)
                    if len(set(candidate)) < len(candidate):
                        continue
                    s = compute_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_U = candidate
                        final_U = candidate
                        final_set = set(final_U)
                        improved = True
                        break
                    if time.time() > deadline - 1:
                        break
                if improved:
                    break
            if not improved:
                break
    
    return best_U
