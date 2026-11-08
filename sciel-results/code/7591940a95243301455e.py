import time
import math

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
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

    # Search for sets {0, a, b, c} with 0 < a < b < c <= 30
    for a in range(1, 15):
        if time.time() > deadline:
            break
        for b in range(a + 1, 25):
            if time.time() > deadline:
                break
            for c in range(b + 1, 31):
                if not try_set([0, a, b, c]):
                    break

    # Search for sets {0, a, b, c, d} with 0 < a < b < c < d <= 20
    for a in range(1, 8):
        if time.time() > deadline:
            break
        for b in range(a + 1, 12):
            if time.time() > deadline:
                break
            for c in range(b + 1, 18):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 21):
                    if not try_set([0, a, b, c, d]):
                        break

    # Search for sets {0, a, b, c, d, e} with 0 < a < b < c < d < e <= 15
    for a in range(1, 6):
        if time.time() > deadline:
            break
        for b in range(a + 1, 9):
            if time.time() > deadline:
                break
            for c in range(b + 1, 12):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 14):
                    if time.time() > deadline:
                        break
                    for e in range(d + 1, 16):
                        if not try_set([0, a, b, c, d, e]):
                            break

    # Local search: try adding, removing, or replacing elements in best_U
    if time.time() < deadline - 5:
        current_U = best_U[:]
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            U_set = set(current_U)
            
            # Try adding a new element
            for new_elem in range(1, 40):
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
                if time.time() > deadline - 2:
                    break
            
            if time.time() > deadline - 2:
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
                if time.time() > deadline - 2:
                    break
            
            if time.time() > deadline - 2:
                break
            
            # Try replacing an element (keep 0)
            for i in range(1, len(current_U)):
                old_elem = current_U[i]
                for new_elem in range(1, 40):
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
                    if time.time() > deadline - 2:
                        break
                if improved:
                    break
                if time.time() > deadline - 2:
                    break
    
    return best_U
