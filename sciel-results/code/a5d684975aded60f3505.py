"""Improved program for sumset/difference-set exponent task."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if len(U) < 1:
        return 0.0
    U_set = set(U)
    if 0 not in U_set:
        return 0.0
    
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute |U+U| and |U-U|
    # For efficiency, use set operations
    U_list = list(U_set)
    n = len(U_list)
    
    # Sumset: all a+b for a,b in U
    # Difference set: all a-b for a,b in U
    # Using numpy for speed if set is small enough
    if n <= 5000:
        arr = np.array(U_list, dtype=np.int64)
        # Sumset
        sums = np.add.outer(arr, arr)
        sumset = set(np.unique(sums).tolist())
        
        # Difference set
        diffs = np.subtract.outer(arr, arr)
        diffset = set(np.unique(diffs).tolist())
    else:
        # Fallback for larger sets - this is slow
        sumset = set()
        for a in U_list:
            for b in U_list:
                sumset.add(a + b)
        diffset = set()
        for a in U_list:
            for b in U_list:
                diffset.add(a - b)
    
    sumset_size = len(sumset)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
    
    import math
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    
    log_ratio = math.log(ratio)
    denom = math.log(2 * max_u + 1)
    
    if denom <= 0:
        return 0.0
    
    score = 1 + log_ratio / denom
    return score


def _greedy_construct(target_size, max_val, deadline):
    """
    Greedy construction: start with {0}, add elements one by one
    to maximize the score.
    """
    U = [0]
    best_score = _compute_score(U)
    
    # Try to build up to target_size elements
    # Candidate pool: integers from 1 to max_val
    # We'll use a heuristic: prefer elements that are "sparse" in some sense
    
    # Start with a base pattern, then refine
    # Common good patterns: powers of 2, or arithmetic progressions with large gaps
    
    # Strategy: try several initial patterns, then locally optimize
    patterns = []
    
    # Pattern 1: Powers of 2
    p = 1
    while p <= max_val:
        patterns.append(p)
        p *= 2
    patterns = [0] + patterns[:target_size - 1]
    
    # Pattern 2: Fibonacci-like
    fib = [0, 1]
    while len(fib) < target_size and fib[-1] + fib[-2] <= max_val:
        fib.append(fib[-1] + fib[-2])
    patterns.append(fib[:target_size])
    
    # Pattern 3: Exponential base 3
    p = 1
    exp3 = [0]
    while len(exp3) < target_size and p <= max_val:
        exp3.append(p)
        p *= 3
    patterns.append(exp3[:target_size])
    
    # Pattern 4: Mixed - powers of 2 with some perturbation
    base2 = [0]
    p = 1
    while len(base2) < target_size and p <= max_val:
        base2.append(p)
        p = p * 2 + 1  # 1, 3, 7, 15, ...
    patterns.append(base2[:target_size])
    
    # Pattern 5: Linear with large step
    step = max_val // max(1, target_size - 1)
    lin = [0]
    for i in range(1, target_size):
        val = i * step
        if val <= max_val:
            lin.append(val)
    if len(lin) == target_size:
        patterns.append(lin)
    
    # Evaluate each pattern
    best_U = None
    for pat in patterns:
        if time.time() > deadline:
            break
        if len(pat) < 1:
            continue
        # Ensure sorted and unique
        pat = sorted(set(pat))
        if 0 not in pat:
            pat = [0] + pat
        s = _compute_score(pat)
        if best_U is None or s > best_score:
            best_score = s
            best_U = pat[:]
    
    if best_U is None:
        best_U = [0]
    
    # Local optimization: try adding/removing/swapping elements
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        current_size = len(best_U)
        
        # Try adding a new element
        for trial in range(100):
            if time.time() > deadline:
                break
            # Random candidate from 1 to max_val
            import random
            cand = random.randint(1, max_val)
            if cand in best_U:
                continue
            new_U = sorted(best_U + [cand])
            s = _compute_score(new_U)
            if s > best_score + 1e-9:
                best_U = new_U
                best_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing an element
        if len(best_U) > 2:
            for idx in range(1, len(best_U)):
                if time.time() > deadline:
                    break
                new_U = best_U[:idx] + best_U[idx+1:]
                s = _compute_score(new_U)
                if s > best_score + 1e-9:
                    best_U = new_U
                    best_score = s
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swapping an element
        if len(best_U) > 2:
            import random
            for trial in range(50):
                if time.time() > deadline:
                    break
                idx = random.randint(1, len(best_U) - 1)
                cand = random.randint(1, max_val)
                if cand in best_U:
                    continue
                new_U = best_U[:idx] + [cand] + best_U[idx+1:]
                new_U = sorted(set(new_U))
                if len(new_U) != len(best_U):
                    continue
                s = _compute_score(new_U)
                if s > best_score + 1e-9:
                    best_U = new_U
                    best_score = s
                    improved = True
                    break
    
    return best_U


def construct():
    """Construct a finite set U to maximize the sumset/difference-set exponent."""
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin out of 150s
    
    # Start with a reasonable construction
    # The paper mentions sizes 2003 and 54265, but we need to balance
    # computation time.
    
    # For small sets, we can be more exhaustive
    # Let's try a few different target sizes and max values
    
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Try exponential constructions with different bases
    for base in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        
        U = [0]
        val = 1
        while val <= 1000000 and len(U) < 50:
            U.append(val)
            val *= base
        
        s = _compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Try the greedy construction with various parameters
    for target_size in [10, 20, 30, 50]:
        if time.time() > deadline:
            break
        
        for max_val in [1000, 10000, 100000]:
            if time.time() > deadline:
                break
            
            U = _greedy_construct(target_size, max_val, deadline)
            s = _compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Final check
    if time.time() > deadline - 5:
        return best_U
    
    # Try one more round of optimization on the best found
    U = _greedy_construct(len(best_U), max(best_U) + 1000, deadline)
    s = _compute_score(U)
    if s > best_score:
        best_U = U
    
    return best_U

# EVOLVE-BLOCK-END
