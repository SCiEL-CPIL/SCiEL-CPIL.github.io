import numpy as np
import time

def _score_of(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if len(U) < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sum_set = np.unique(sums)
    diff_set = np.unique(diffs)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    log_ratio = np.log(ratio)
    log_denom = np.log(2 * max_u + 1)
    if log_denom <= 0:
        return 0.0
    return 1.0 + log_ratio / log_denom

def _score_of_fast(U_arr):
    """Fast score computation assuming U_arr is sorted unique numpy array."""
    if len(U_arr) < 2:
        return 0.0
    max_u = U_arr[-1]
    if max_u == 0:
        return 0.0
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sum_set = np.unique(sums)
    diff_set = np.unique(diffs)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    log_ratio = np.log(ratio)
    log_denom = np.log(2 * max_u + 1)
    if log_denom <= 0:
        return 0.0
    return 1.0 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_score = -np.inf
    best_set = [0, 1, 3]
    best_score = _score_of(best_set)
    
    np.random.seed(12345)
    
    # Phase 1: Explore small sets systematically
    # For sets of size 3-6, try various configurations
    
    # Size 3: {0, a, b}
    max_a = 30
    max_b = 60
    for a in range(1, max_a + 1):
        if time.time() > deadline:
            break
        for b in range(a + 1, max_b + 1):
            U = [0, a, b]
            s = _score_of(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Phase 2: Local search from best
    current = best_set[:]
    current_score = best_score
    
    max_val = 100
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        strategy = np.random.randint(4)
        
        if strategy == 0:  # Add element
            if len(current) < 10:
                new_elem = np.random.randint(1, max_val + 1)
                if new_elem not in current:
                    new_set = sorted(current + [new_elem])
                    s = _score_of(new_set)
                    if s > current_score:
                        current = new_set
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = current[:]
        
        elif strategy == 1:  # Remove element
            if len(current) > 2:
                idx = np.random.randint(1, len(current))  # Don't remove 0
                new_set = current[:idx] + current[idx+1:]
                s = _score_of(new_set)
                if s > current_score:
                    current = new_set
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = current[:]
        
        elif strategy == 2:  # Replace element
            if len(current) >= 2:
                idx = np.random.randint(1, len(current))
                new_elem = np.random.randint(1, max_val + 1)
                if new_elem not in current:
                    new_set = current[:idx] + [new_elem] + current[idx+1:]
                    new_set = sorted(set(new_set))
                    if 0 in new_set and len(new_set) >= 2:
                        s = _score_of(new_set)
                        if s > current_score:
                            current = new_set
                            current_score = s
                            if s > best_score:
                                best_score = s
                                best_set = current[:]
        
        elif strategy == 3:  # Perturb all elements slightly
            if len(current) >= 2:
                new_set = [0]
                for i in range(1, len(current)):
                    delta = np.random.randint(-3, 4)
                    new_val = current[i] + delta
                    if new_val > 0 and new_val not in new_set:
                        new_set.append(new_val)
                    else:
                        new_set.append(current[i])
                new_set = sorted(set(new_set))
                if 0 in new_set and len(new_set) >= 2:
                    s = _score_of(new_set)
                    if s > current_score:
                        current = new_set
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = current[:]
    
    # Phase 3: Final polish
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(1, len(best_set)):
                for delta in range(-3, 4):
                    if delta == 0:
                        continue
                    new_val = best_set[i] + delta
                    if new_val <= 0 or new_val in best_set:
                        continue
                    new_set = best_set[:]
                    new_set[i] = new_val
                    new_set = sorted(set(new_set))
                    s = _score_of(new_set)
                    if s > best_score:
                        best_score = s
                        best_set = new_set[:]
                        improved = True
                        break
    
    return best_set
