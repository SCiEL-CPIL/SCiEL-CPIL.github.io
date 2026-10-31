import math
import time
import numpy as np

def construct():
    DEADLINE = 140  # seconds, leave 10s safety margin

    def compute_score(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        arr = np.array(u, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
        r = diffset_size / sumset_size
        return 1 + math.log(r) / math.log(m)
    
    def is_golomb(u):
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    # Known verified Golomb rulers (optimal or near-optimal)
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 22],
    ]
    
    # Local search: try adding, removing, or swapping elements
    def local_search(base_u, time_limit):
        start_time = time.time()
        best_u = list(base_u)
        best_score = compute_score(best_u)
        improved = True
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            best_candidate = None
            best_candidate_score = best_score
            
            # Try adding a new element
            current_max = max(best_u)
            n = len(best_u)
            search_hi = current_max + n * n
            for new_mark in range(current_max + 1, search_hi + 1):
                candidate = sorted(best_u + [new_mark])
                if not is_golomb(candidate):
                    continue
                s = compute_score(candidate)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = candidate
                if (time.time() - start_time) > time_limit:
                    break
            
            # Try removing an element (if size > 3)
            if best_u and len(best_u) > 3 and (time.time() - start_time) < time_limit:
                for i in range(1, len(best_u)):  # keep 0
                    candidate = [x for j, x in enumerate(best_u) if j != i]
                    if not is_golomb(candidate):
                        continue
                    s = compute_score(candidate)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate
                    if (time.time() - start_time) > time_limit:
                        break
            
            # Try swapping: replace one element with another
            if best_u and (time.time() - start_time) < time_limit:
                for i in range(1, len(best_u)):  # don't swap 0
                    old_val = best_u[i]
                    # Try replacing with nearby values
                    for delta in range(1, max(10, len(best_u) * 3)):
                        for new_val in [old_val - delta, old_val + delta]:
                            if new_val < 0:
                                continue
                            if new_val in best_u:
                                continue
                            candidate = sorted([x if x != old_val else new_val for x in best_u])
                            if not is_golomb(candidate):
                                continue
                            s = compute_score(candidate)
                            if s > best_candidate_score:
                                best_candidate_score = s
                                best_candidate = candidate
                        if (time.time() - start_time) > time_limit:
                            break
            
            if best_candidate is not None:
                best_u = best_candidate
                best_score = best_candidate_score
                improved = True
        
        return best_u, best_score
    
    # Run local search on multiple starting points
    start_points = [
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 9, 11],
    ]
    
    for sp in start_points:
        remaining = DEADLINE - (time.time() - (DEADLINE - 140))  # approximate
        if remaining < 5:
            break
        # Allocate time proportionally
        time_for_this = min(remaining * 0.3, 40)
        if time_for_this < 5:
            time_for_this = max(5, remaining * 0.5)
        ext_u, ext_score = local_search(sp, time_for_this)
        if ext_u and len(ext_u) >= 3:
            candidates.append(ext_u)
    
    # Greedy Golomb ruler construction
    def greedy_golomb(n, time_limit):
        start_time = time.time()
        u = [0]
        for i in range(1, n):
            if (time.time() - start_time) > time_limit:
                break
            pos = 0
            found = False
            while not found:
                pos += 1
                candidate = sorted(u + [pos])
                if is_golomb(candidate):
                    u = candidate
                    found = True
                if pos > n * n * 20:
                    break
        return u
    
    greedy_u = greedy_golomb(10, time_limit=10)
    if greedy_u and len(greedy_u) >= 6:
        candidates.append(greedy_u)
    
    # Find best
    best_set = [0, 1, 3]
    best_score = 0.0
    for u in candidates:
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = u
    
    return best_set
