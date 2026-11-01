import math
import time
import numpy as np

def construct():
    START = time.time()
    DEADLINE = START + 110  # Safety margin for 120s limit

    def elapsed():
        return time.time() - START

    def time_ok(limit_frac=1.0):
        return elapsed() < DEADLINE - 5 * limit_frac

    def compute_score(u):
        """Compute score using numpy for speed."""
        if not u:
            return -1.0
        u_arr = np.array(u, dtype=np.int64)
        n = len(u_arr)
        if 0 not in u_arr:
            return -1.0
        max_u = int(u_arr.max())
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        
        # Sumset: all a+b for a,b in u
        sums = u_arr[:, None] + u_arr[None, :]
        sumset_size = len(np.unique(sums))
        
        # Diffset: all a-b for a,b in u
        diffs = u_arr[:, None] - u_arr[None, :]
        diffset_size = len(np.unique(diffs))
        
        if sumset_size == 0:
            return -1.0
        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    # Known optimal Golomb rulers
    # Source: OEIS A003022 or similar
    golomb_rulers = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 44],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 82],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 82, 94],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 82, 94, 112],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 82, 94, 112, 121],
    ]
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    # Evaluate all known Golomb rulers
    for u in golomb_rulers:
        if not time_ok():
            break
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)
            
    # Local search to improve
    def local_search(start_u, time_budget=30):
        start_t = time.time()
        current_u = sorted(set(start_u))
        current_score = compute_score(current_u)
        best_u = list(current_u)
        best_s = current_score
        
        while time.time() - start_t < time_budget and time_ok():
            improved = False
            max_u = max(current_u)
            
            # Try adding elements
            # Search range: up to max_u + len(current_u)^2
            search_hi = max_u + len(current_u) * len(current_u) + 10
            for new_val in range(max_u + 1, search_hi + 1):
                if not time_ok():
                    break
                candidate = sorted(current_u + [new_val])
                s = compute_score(candidate)
                if s > current_score + 1e-12:
                    current_u = candidate
                    current_score = s
                    if s > best_s:
                        best_s = s
                        best_u = list(candidate)
                    improved = True
                    break
            
            # Try removing elements (if len > 2)
            if len(current_u) > 2:
                for i in range(len(current_u)):
                    if current_u[i] == 0:
                        continue
                    candidate = [x for j, x in enumerate(current_u) if j != i]
                    if len(candidate) >= 2:
                        s = compute_score(candidate)
                        if s > current_score + 1e-12:
                            current_u = candidate
                            current_score = s
                            if s > best_s:
                                best_s = s
                                best_u = list(candidate)
                            improved = True
                            break
                            
            # Try shifting elements
            if not improved:
                for i in range(len(current_u)):
                    if current_u[i] == 0:
                        continue
                    for delta in [-2, -1, 1, 2]:
                        new_val = current_u[i] + delta
                        if new_val < 1:
                            continue
                        candidate = list(current_u)
                        candidate[i] = new_val
                        candidate = sorted(set(candidate))
                        if 0 not in candidate or len(candidate) != len(current_u):
                            continue
                        s = compute_score(candidate)
                        if s > current_score + 1e-12:
                            current_u = candidate
                            current_score = s
                            if s > best_s:
                                best_s = s
                                best_u = list(candidate)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
                
        return best_u, best_s

    if time_ok(5.0):
        ls_u, ls_s = local_search(best_set, time_budget=60)
        if ls_s > best_score:
            best_score = ls_s
            best_set = ls_u

    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set = [0] + best_set
        
    return best_set
