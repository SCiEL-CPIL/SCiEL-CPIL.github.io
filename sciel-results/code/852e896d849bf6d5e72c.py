import itertools
import time
from math import log
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def fast_score_set(U):
    """Faster scoring using numpy for larger sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pair sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    sum_count = len(np.unique(sums))
    
    # Compute all pair differences (i < j), including negatives and 0
    i_idx, j_idx = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx] - U_arr[i_idx]
    # Include 0 and negatives
    all_diffs = np.concatenate([diffs, -diffs, np.array([0])])
    diff_count = len(np.unique(all_diffs))
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def build_from_radices(radices, digit_counts):
    """Build set U from mixed-radix expansion: U = {sum d_i * r_i^i}
    where d_i ranges over [0, digit_counts[i]).
    This gives a structured set with controlled sumset/difference-set sizes.
    """
    U = [0]
    for i in range(len(radices)):
        r = radices[i]
        dc = digit_counts[i]
        # For each existing element, add r^i * d for d in 0..dc-1
        # But we need to be careful: the expansion is positional
        # Actually, let's think of it differently:
        # U = {sum_{i} d_i * base_i where base_0=1, base_i = prod_{j<i} r_j}
        # and 0 <= d_i < digit_counts[i]
        
        # Rebuild properly
        pass
    
    # Proper construction:
    U = set()
    base = 1
    # Start with just {0}
    current = [0]
    for i in range(len(radices)):
        r = radices[i]
        dc = digit_counts[i]
        new_current = []
        for u in current:
            for d in range(dc):
                new_current.append(u + d * base)
        current = new_current
        base *= r
    
    U = sorted(set(current))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    return U

def greedy_radix_search(deadline, best_score, best_set):
    """Greedy hill-climbing on radix/digit parameters."""
    current_best_score = best_score
    current_best_set = best_set[:]
    
    # Start with small structured sets and try to optimize radices
    # Try various initial configurations
    initial_configs = [
        # (radices, digit_counts)
        ([2, 2, 2], [2, 2, 2]),
        ([3, 3], [2, 2]),
        ([2, 3, 2], [2, 2, 2]),
        ([3, 2, 3], [2, 2, 2]),
        ([2, 2, 3], [2, 2, 2]),
        ([4, 2], [2, 2]),
        ([2, 4], [2, 2]),
        ([3, 3, 2], [2, 2, 2]),
        ([2, 2, 2, 2], [2, 2, 2, 2]),
        ([5, 2], [2, 2]),
        ([2, 5], [2, 2]),
        ([3, 4], [2, 2]),
        ([4, 3], [2, 2]),
        ([2, 3, 3], [2, 2, 2]),
        ([3, 2, 2], [2, 2, 2]),
    ]
    
    # First, find the best initial config
    for radices, digits in initial_configs:
        if time.time() > deadline:
            break
        U = build_from_radices(radices, digits)
        if len(U) < 2 or len(U) > 200:
            continue
        s = fast_score_set(U)
        if s > current_best_score:
            current_best_score = s
            current_best_set = U[:]
    
    # Now do local search: perturb radices and digits
    max_perturb_iters = 3000
    no_improve_count = 0
    max_no_improve = 50
    
    # We need to track the current radices/digits representation
    # For simplicity, work directly on the set but use structured perturbations
    current_set = current_best_set[:]
    current_score = current_best_score
    
    for iteration in range(max_perturb_iters):
        if time.time() > deadline:
            break
        if no_improve_count > max_no_improve:
            break
        
        n = len(current_set)
        if n < 2:
            break
        
        improved = False
        strategy = iteration % 4
        
        if strategy == 0:
            # Replace one element with a nearby value
            idx = 1 + (iteration % (n - 1))
            old_val = current_set[idx]
            for delta in range(-3, 4):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0 or new_val in current_set:
                    continue
                candidate = current_set[:]
                candidate[idx] = new_val
                candidate = sorted(set(candidate))
                if 0 not in candidate or len(candidate) != n:
                    continue
                s = fast_score_set(candidate)
                if s > current_score:
                    current_score = s
                    current_set = candidate
                    if s > current_best_score:
                        current_best_score = s
                        current_best_set = candidate[:]
                    improved = True
                    break
        
        elif strategy == 1:
            # Add a new element (if set isn't too large)
            if n < 30:
                # Try values that might create good structure
                max_u = max(current_set)
                for new_val in [max_u + 1, max_u + 2, max_u + 3, max_u * 2, max_u * 2 + 1]:
                    if new_val <= 0 or new_val in current_set:
                        continue
                    candidate = sorted(current_set + [new_val])
                    s = fast_score_set(candidate)
                    if s > current_score:
                        current_score = s
                        current_set = candidate
                        if s > current_best_score:
                            current_best_score = s
                            current_best_set = candidate[:]
                        improved = True
                        break
        
        elif strategy == 2:
            # Remove an element (if set is large enough)
            if n > 3:
                idx = 1 + (iteration % (n - 1))
                candidate = current_set[:idx] + current_set[idx+1:]
                candidate = sorted(set(candidate))
                if 0 in candidate and len(candidate) >= 2:
                    s = fast_score_set(candidate)
                    if s > current_score:
                        current_score = s
                        current_set = candidate
                        if s > current_best_score:
                            current_best_score = s
                            current_best_set = candidate[:]
                        improved = True
        
        else:
            # Swap two elements
            if n >= 3:
                i1 = 1 + (iteration % (n - 1))
                i2 = 1 + ((iteration + 1) % (n - 1))
                if i1 == i2:
                    i2 = (i2 + 1) % (n - 1) + 1
                candidate = current_set[:]
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                candidate = sorted(set(candidate))
                if 0 in candidate and len(candidate) == n:
                    s = fast_score_set(candidate)
                    if s > current_score:
                        current_score = s
                        current_set = candidate
                        if s > current_best_score:
                            current_best_score = s
                            current_best_set = candidate[:]
                        improved = True
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
    
    return current_best_score, current_best_set

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    best_score = 1.0
    best_set = [0, 1]
    
    # Phase 1: Quick exhaustive search for very small n
    for n in range(3, 6):
        if time.time() > deadline:
            break
        range_vals = range(1, 30 + 1)
        
        for comb in itertools.combinations(range_vals, n - 1):
            if time.time() > deadline:
                break
            U = [0] + list(comb)
            s = fast_score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Phase 2: Try structured geometric/mixed-radix sets
    structured_sets = [
        [0, 1, 3, 7],
        [0, 1, 3, 8],
        [0, 1, 3, 9],
        [0, 1, 4, 9],
        [0, 1, 4, 16],
        [0, 1, 2, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 8, 16],
        [0, 1, 4, 9, 16],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 4, 9],
        [0, 1, 3, 6, 10],
        [0, 1, 3, 9, 27],
        [0, 1, 4, 16, 64],
        [0, 1, 2, 6, 18],
        [0, 1, 3, 12, 48],
        [0, 1, 2, 5, 10, 20],
        [0, 1, 3, 6, 12, 24],
        [0, 1, 4, 8, 16, 32],
        [0, 1, 2, 6, 18, 54],
        [0, 1, 3, 9, 27, 81],
        [0, 2, 4, 8, 16],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 5, 10, 20, 40],
        [0, 1, 3, 7, 15, 31, 63],
        [0, 1, 4, 9, 16, 25, 36],
        [0, 1, 2, 4, 8, 16, 32],
    ]
    for s_set in structured_sets:
        if time.time() > deadline:
            break
        s = fast_score_set(s_set)
        if s > best_score:
            best_score = s
            best_set = s_set[:]
    
    # Phase 3: Mixed-radix greedy search
    if time.time() < deadline - 30:
        best_score, best_set = greedy_radix_search(deadline - 30, best_score, best_set)
    
    # Phase 4: Final polish - fine-tune with wider delta range
    if best_score > 1.0 and time.time() < deadline - 10:
        current_set = best_set[:]
        current_score = best_score
        fine_tune_iterations = 5000
        no_improve = 0
        max_no_improve = 40
        
        for iteration in range(fine_tune_iterations):
            if time.time() > deadline - 5:
                break
            if no_improve > max_no_improve:
                break
            
            n = len(current_set)
            if n < 2:
                break
            
            improved = False
            # Try different perturbation strategies
            if iteration % 2 == 0:
                # Replace element
                idx = 1 + (iteration % (n - 1))
                old_val = current_set[idx]
                # Wider delta range
                deltas = list(range(-5, 6))
                import random
                random.shuffle(deltas)
                for delta in deltas:
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0 or new_val in current_set:
                        continue
                    candidate = current_set[:]
                    candidate[idx] = new_val
                    candidate = sorted(set(candidate))
                    if 0 not in candidate or len(candidate) != n:
                        continue
                    s = fast_score_set(candidate)
                    if s > current_score:
                        current_score = s
                        current_set = candidate
                        if s > best_score:
                            best_score = s
                            best_set = candidate[:]
                        improved = True
                        break
            else:
                # Add or remove
                if iteration % 4 == 1 and n < 40:
                    # Add
                    max_u = max(current_set)
                    for new_val in [max_u + 1, max_u + 2, max_u + 3, max_u + 5, max_u + 10]:
                        if new_val in current_set:
                            continue
                        candidate = sorted(current_set + [new_val])
                        s = fast_score_set(candidate)
                        if s > current_score:
                            current_score = s
                            current_set = candidate
                            if s > best_score:
                                best_score = s
                                best_set = candidate[:]
                            improved = True
                            break
                elif iteration % 4 == 3 and n > 3:
                    # Remove
                    idx = 1 + (iteration % (n - 1))
                    candidate = current_set[:idx] + current_set[idx+1:]
                    candidate = sorted(set(candidate))
                    if 0 in candidate and len(candidate) >= 2:
                        s = fast_score_set(candidate)
                        if s > current_score:
                            current_score = s
                            current_set = candidate
                            if s > best_score:
                                best_score = s
                                best_set = candidate[:]
                            improved = True
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
    
    return best_set

if __name__ == "__main__":
    start = time.time()
    U = construct()
    elapsed = time.time() - start
    print("Result:", U)
    print("Score:", score_set(U))
    print("Time: %.2f s" % elapsed)
