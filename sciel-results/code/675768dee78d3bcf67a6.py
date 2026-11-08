import time
import math
from itertools import combinations

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
    
    best_U = [0, 1, 3, 9]
    best_score = compute_score(best_U)
    
    def search_size(n, max_range):
        nonlocal best_U, best_score
        for combo in combinations(range(1, max_range + 1), n - 1):
            if time.time() > deadline:
                return
            U = [0] + list(combo)
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Size 4: C(30,3) = 4060
    search_size(4, 30)
    
    # Size 5: C(40,4) = 91390
    if time.time() < deadline - 25:
        search_size(5, 40)
    
    # Size 6: C(35,5) = 324632
    if time.time() < deadline - 15:
        search_size(6, 35)
    
    # Local search from best found
    if time.time() < deadline - 5:
        U = best_U[:]
        current_score = best_score
        for _ in range(300):
            if time.time() > deadline:
                break
            improved = False
            max_val = U[-1]
            # Try adding
            for new_val in range(1, min(max_val + 20, 100)):
                if new_val in U:
                    continue
                new_U = U + [new_val]
                s = compute_score(new_U)
                if s > current_score + 1e-12:
                    current_score = s
                    U = new_U
                    improved = True
                    break
            if improved:
                continue
            # Try removing (not 0)
            for i in range(1, len(U)):
                new_U = U[:i] + U[i+1:]
                if len(new_U) < 2:
                    continue
                s = compute_score(new_U)
                if s > current_score + 1e-12:
                    current_score = s
                    U = new_U
                    improved = True
                    break
            if improved:
                continue
            # Try modifying
            for i in range(1, len(U)):
                old = U[i]
                for new_val in range(1, max_val + 5):
                    if new_val == old or new_val in U:
                        continue
                    new_U = U[:]
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if len(new_U) != len(U):
                        continue
                    s = compute_score(new_U)
                    if s > current_score + 1e-12:
                        current_score = s
                        U = new_U
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        if current_score > best_score:
            best_score = current_score
            best_U = U[:]
    
    return best_U
