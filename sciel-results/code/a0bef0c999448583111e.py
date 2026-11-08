import time
import random
import numpy as np
from math import log

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    # Sumset: all pairwise sums (with repetition)
    sums = U_arr[:, None] + U_arr[None, :]
    sums_unique = np.unique(sums)
    # Difference set: all pairwise differences (with sign)
    diffs = U_arr[:, None] - U_arr[None, :]
    diffs_unique = np.unique(diffs)
    
    sum_count = len(sums_unique)
    diff_count = len(diffs_unique)
    
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

def score_set_fast(U):
    """Vectorized scoring for speed."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    sums_unique = np.unique(sums)
    diffs = U_arr[:, None] - U_arr[None, :]
    diffs_unique = np.unique(diffs)
    sum_count = len(sums_unique)
    diff_count = len(diffs_unique)
    if sum_count == 0:
        return -1
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
    max_u = int(U_arr.max())
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def generate_structural_seeds():
    """Generate diverse structural seeds focused on geometric and mixed structures."""
    seeds = []
    
    # Pure Powers of 2 (Baseline)
    for k in range(3, 12):
        seeds.append([0] + [2**i for i in range(1, k)])
        
    # Geometric with non-integer base, rounded
    for base in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.5, 3.0]:
        U = [0]
        val = 1
        seen = {0}
        for _ in range(25):
            val = int(round(val * base))
            if val <= 0:
                val = 1
            if val not in seen:
                U.append(val)
                seen.add(val)
            else:
                # If collision, try to break it slightly
                val += 1
                if val not in seen:
                    U.append(val)
                    seen.add(val)
        if len(U) > 1:
            seeds.append(sorted(U))
            
    # "Block" structures: clusters of dense points separated by large gaps
    # e.g., {0, 1, 2, 10, 11, 12, 30, 31, 32}
    for block_size in [2, 3, 4]:
        for gap in [5, 10, 20, 50, 100]:
            U = [0]
            seen = {0}
            for i in range(1, 10):
                start = i * gap
                for j in range(block_size):
                    v = start + j
                    if v not in seen:
                        U.append(v)
                        seen.add(v)
            if len(U) > 1:
                seeds.append(sorted(U))
                
    # Random sparse with specific constraints
    for _ in range(20):
        n = random.randint(5, 15)
        # Generate sorted random points
        U = [0]
        last = 0
        for _ in range(n-1):
            # Exponential-ish spacing
            step = random.randint(1, max(1, int(last * 0.5) + 1))
            last += step
            U.append(last)
        seeds.append(sorted(U))
        
    return seeds

def local_search_structural(U, deadline, max_n=30):
    """
    Structural local search:
    1. Try moving each element to nearby values.
    2. Try adding/removing elements at the end.
    3. Try swapping elements.
    """
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + [x for x in U if x > 0]
    current_score = score_set_fast(U)
    best_score = current_score
    best_U = list(U)
    
    improved = True
    iteration = 0
    max_val_bound = max(U) * 3 + 100 if U else 100
    
    while improved and time.time() < deadline - 2 and iteration < 500:
        improved = False
        iteration += 1
        n = len(U)
        
        # 1. Move elements (local perturbation)
        # Focus on moving elements to values that might create sum collisions
        # or maintain difference distinctness.
        for i in range(1, n):
            if time.time() > deadline - 2:
                break
            old_val = U[i]
            # Try a range of new values
            # Range depends on the gap to neighbors
            prev_val = U[i-1]
            next_val = U[i+1] if i < n-1 else old_val + max(10, old_val//2)
            
            search_low = max(1, prev_val + 1)
            search_high = min(max_val_bound, next_val - 1)
            
            # Sample a few values in the gap
            if search_high > search_low:
                candidates = set()
                # Neighbors of old_val
                for delta in range(-5, 6):
                    v = old_val + delta
                    if search_low <= v <= search_high:
                        candidates.add(v)
                # Random in gap
                for _ in range(5):
                    candidates.add(random.randint(search_low, search_high))
                
                for new_val in candidates:
                    if new_val in U:
                        continue
                    new_U = [U[j] if j != i else new_val for j in range(n)]
                    new_U = sorted(new_U)
                    new_score = score_set_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        best_score = new_score
                        best_U = list(U)
                        improved = True
                        break
            if improved:
                break
                
        # 2. Add element
        if not improved and n < max_n and time.time() < deadline - 2:
            max_u = max(U)
            # Try adding near the end or in large gaps
            candidates = set()
            # Near max
            for v in range(max(1, max_u - 10), max_u + 11):
                if v not in U:
                    candidates.add(v)
            # In gaps
            for i in range(1, n):
                gap = U[i] - U[i-1]
                if gap > 10:
                    mid = (U[i] + U[i-1]) // 2
                    for v in [mid, mid-1, mid+1]:
                        if U[i-1] < v < U[i] and v not in U:
                            candidates.add(v)
                            
            for new_val in candidates:
                new_U = sorted(U + [new_val])
                new_score = score_set_fast(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    best_score = new_score
                    best_U = list(U)
                    improved = True
                    break
                    
        # 3. Remove element
        if not improved and n > 3 and time.time() < deadline - 2:
            for i in range(1, n):
                new_U = [U[j] for j in range(n) if j != i]
                new_score = score_set_fast(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    best_score = new_score
                    best_U = list(U)
                    improved = True
                    break
                    
    return best_score, best_U

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate diverse seeds
    seeds = generate_structural_seeds()
    
    # Score all seeds
    best_score = -1
    best_U = [0, 1]
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        s = score_set_fast(seed)
        if s > best_score:
            best_score = s
            best_U = list(seed)
    
    # Run local search from the best seed and a few others
    # Allocate time: ~80s for LS, ~30s for polishing (if needed)
    ls_deadline = start_time + 90
    
    # Run LS from the best seed and top 3 others
    ls_seeds = [best_U]
    scored_seeds = [(score_set_fast(s), s) for s in seeds]
    scored_seeds.sort(reverse=True)
    for s_score, s in scored_seeds:
        if s != best_U and len(ls_seeds) < 4:
            ls_seeds.append(list(s))
            
    for seed in ls_seeds:
        if time.time() > ls_deadline - 5:
            break
        ls_score, ls_U = local_search_structural(seed, ls_deadline, max_n=30)
        if ls_score > best_score:
            best_score = ls_score
            best_U = ls_U
    
    # Final polish: very small moves
    if time.time() < deadline - 2:
        # Run local_search_structural again with a tighter deadline?
        # Or just return best_U
        pass
    
    return best_U

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
