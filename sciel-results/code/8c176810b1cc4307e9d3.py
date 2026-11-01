"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    U_arr = np.array(sorted(U), dtype=np.int64)
    maxU = int(U_arr[-1])
    if maxU <= 0:
        return 0.0
    
    # Compute |U+U| and |U-U|
    # For efficiency, use set operations on numpy arrays
    sums = set()
    diffs = set()
    U_list = sorted(U)
    for i in range(n):
        for j in range(i, n):
            sums.add(U_list[i] + U_list[j])
        for j in range(n):
            diffs.add(U_list[i] - U_list[j])
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0 or diff_count == 0:
        return 0.0
    
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    
    denom = 2 * maxU + 1
    if denom <= 0:
        return 0.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(denom)
    
    if log_denom == 0:
        return 0.0
    
    score = 1.0 + log_ratio / log_denom
    return score

def _score_set_fast(U):
    """Faster score computation using numpy for larger sets."""
    if not U or 0 not in U:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    U_arr = np.array(sorted(U), dtype=np.int64)
    maxU = int(U_arr[-1])
    if maxU <= 0:
        return 0.0
    
    # Use numpy broadcasting for sums and diffs (works for n up to ~500)
    if n <= 500:
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        sum_count = len(np.unique(sums))
        diff_count = len(np.unique(diffs))
    else:
        # Fallback to set-based for large n
        U_list = sorted(U)
        sums = set()
        diffs = set()
        for i in range(n):
            for j in range(i, n):
                sums.add(U_list[i] + U_list[j])
            for j in range(n):
                diffs.add(U_list[i] - U_list[j])
        sum_count = len(sums)
        diff_count = len(diffs)
    
    if sum_count == 0 or diff_count == 0:
        return 0.0
    
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    
    denom = 2 * maxU + 1
    if denom <= 0:
        return 0.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(denom)
    
    if log_denom == 0:
        return 0.0
    
    score = 1.0 + log_ratio / log_denom
    return score

def _sidon_seed(n, max_val):
    """Generate a Sidon-like set of size n within [0, max_val]."""
    # Use greedy construction: pick elements to minimize collisions
    # A Sidon set has all pairwise sums distinct
    # Start with 0, then greedily add elements
    U = [0]
    used_sums = {0}
    
    # Use a quadratic sequence as starting point
    # For Sidon sets, squares work: {0, 1, 4, 9, 16, ...}
    # But we want them within [0, max_val]
    
    # Greedy approach: try candidates in order of some heuristic
    candidates = list(range(1, max_val + 1))
    random.shuffle(candidates)
    
    for c in candidates:
        if len(U) >= n:
            break
        # Check if adding c creates duplicate sums
        valid = True
        new_sums = set()
        for u in U:
            s = u + c
            if s in used_sums:
                valid = False
                break
            new_sums.add(s)
        if valid:
            U.append(c)
            used_sums.update(new_sums)
    
    U.sort()
    return U

def _perturbed_ap(n, max_val, noise_frac=0.1):
    """Generate a perturbed arithmetic progression."""
    if n <= 1:
        return [0]
    step = max_val // (n - 1) if n > 1 else 0
    if step <= 0:
        step = 1
    base = [i * step for i in range(n)]
    # Ensure 0 is first
    base[0] = 0
    # Add small perturbations
    noise = int(step * noise_frac)
    if noise > 0:
        for i in range(1, n):
            base[i] += random.randint(-noise, noise)
    # Ensure sorted and non-negative
    base = sorted(set(max(0, x) for x in base))
    # Pad if needed
    while len(base) < n:
        base.append(max_val)
    base = base[:n]
    base.sort()
    if base[0] != 0:
        base[0] = 0
    return base

def _local_search(U, max_val, max_iters=2000, timeout=None, start_time=None):
    """Local search with swap/add/remove moves."""
    if timeout is None:
        timeout = 10.0
    if start_time is None:
        start_time = time.time()
    
    U = sorted(U)
    current_score = _score_set_fast(U)
    best_score = current_score
    best_U = U[:]
    
    n = len(U)
    
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        move_type = random.choice(['swap', 'add', 'remove'])
        
        if move_type == 'swap' and n >= 2:
            # Swap one element with another value
            idx = random.randint(0, n - 1)
            old_val = U[idx]
            new_val = random.randint(0, max_val)
            if new_val == old_val:
                continue
            # Check uniqueness
            if new_val in U:
                continue
            new_U = U[:]
            new_U[idx] = new_val
            new_U.sort()
            new_score = _score_set_fast(new_U)
            if new_score > current_score:
                U = new_U
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_U = U[:]
        
        elif move_type == 'add' and n < 300:
            # Add a new element
            new_val = random.randint(0, max_val)
            if new_val in U:
                continue
            new_U = U + [new_val]
            new_U.sort()
            new_score = _score_set_fast(new_U)
            if new_score > current_score:
                U = new_U
                n = len(U)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_U = U[:]
        
        elif move_type == 'remove' and n > 2:
            # Remove an element (but keep 0)
            idx = random.randint(1, n - 1)  # Don't remove 0
            new_U = U[:idx] + U[idx+1:]
            new_score = _score_set_fast(new_U)
            if new_score > current_score:
                U = new_U
                n = len(U)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_U = U[:]
    
    return best_U, best_score

def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize score."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    random.seed(42)
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # Try different seed strategies
    strategies = []
    
    # Strategy 1: Sidon-like seeds of various sizes
    for n in [5, 10, 20, 50, 100, 200]:
        for max_val in [100, 500, 2000, 10000]:
            if n * n > max_val * 2:  # Rough feasibility check
                continue
            U = _sidon_seed(n, max_val)
            if len(U) >= 2:
                strategies.append(U)
    
    # Strategy 2: Perturbed AP seeds
    for n in [5, 10, 20, 50, 100]:
        for max_val in [100, 500, 2000, 10000]:
            U = _perturbed_ap(n, max_val)
            if len(U) >= 2:
                strategies.append(U)
    
    # Strategy 3: Random sets
    for n in [5, 10, 20, 50]:
        for max_val in [100, 500, 2000]:
            U = [0] + sorted(random.sample(range(1, max_val + 1), min(n - 1, max_val)))
            strategies.append(U)
    
    # Evaluate all seeds
    for U in strategies:
        if time.time() > deadline:
            break
        score = _score_set_fast(U)
        if score > best_score:
            best_score = score
            best_U = U[:]
    
    # Local search from best seed
    if time.time() < deadline - 5:
        # Determine a reasonable max_val for local search
        maxU = max(best_U) if best_U else 100
        max_val = max(maxU * 2, 100)
        
        remaining = deadline - time.time() - 2
        if remaining > 5:
            best_U, best_score = _local_search(
                best_U, max_val, 
                max_iters=5000,
                timeout=remaining,
                start_time=time.time()
            )
    
    # Final polish: try to improve with targeted moves
    if time.time() < deadline - 2:
        maxU = max(best_U) if best_U else 100
        max_val = max(maxU * 3, 500)
        remaining = deadline - time.time() - 1
        if remaining > 3:
            best_U, best_score = _local_search(
                best_U, max_val,
                max_iters=3000,
                timeout=remaining,
                start_time=time.time()
            )
    
    return best_U
# EVOLVE-BLOCK-END
