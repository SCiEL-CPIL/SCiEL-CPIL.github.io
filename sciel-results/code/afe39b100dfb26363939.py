"""Construct a finite set U of nonnegative integers containing 0 to maximize
the sumset/difference-set exponent lower bound.

Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
"""
import math
import random
import time
import sys
import numpy as np

def _compute_score_fast(U):
    """Compute score using numpy for speed."""
    if not U or 0 not in U:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    max_u = int(arr.max())
    if max_u == 0:
        return 1.0
    # Sumset: all pairwise sums
    sums = arr[:, None] + arr[None, :]
    sumset_size = len(np.unique(sums))
    # Diffset: all pairwise differences
    diffs = arr[:, None] - arr[None, :]
    diffset_size = len(np.unique(diffs))
    log_num = math.log(diffset_size) - math.log(sumset_size)
    log_den = math.log(2 * max_u + 1)
    if log_den == 0:
        return 1.0
    return 1.0 + log_num / log_den

def _compute_score(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    U_set = set(U)
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    log_num = math.log(len(diffset)) - math.log(len(sumset))
    log_den = math.log(2 * max_u + 1)
    if log_den == 0:
        return 1.0
    return 1.0 + log_num / log_den

def _geometric_set(base, n, offset=0):
    """Generate a geometric-like set: {offset + base^i for i in range(n)}."""
    if n <= 0:
        return [0]
    elements = [0]
    val = 1
    for i in range(1, n):
        elements.append(offset + val)
        val *= base
    return sorted(set(elements))

def _parametric_set(params):
    """Generate a set from parameters: mix of geometric and arithmetic components.
    params: list of (base, count, scale) tuples
    """
    elements = {0}
    for base, count, scale in params:
        val = 1
        for i in range(count):
            elements.add(int(scale * val))
            val *= base
    return sorted(elements)

def _simulated_annealing(params, deadline, seed=42):
    """Simulated annealing over parameter space to optimize set construction."""
    random.seed(seed)
    np.random.seed(seed)
    
    current_params = [p[:] for p in params]
    current_U = _parametric_set(current_params)
    if 0 not in current_U:
        current_U.insert(0, 0)
    current_score = _compute_score_fast(current_U)
    
    best_U = current_U[:]
    best_score = current_score
    best_params = [p[:] for p in current_params]
    
    T = 1.0  # Initial temperature
    T_min = 1e-6
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Propose a perturbation
        p_idx = random.randint(0, len(current_params) - 1)
        p = current_params[p_idx]
        
        # Perturb one parameter
        if random.random() < 0.3:
            # Change base
            new_base = max(2, int(p[0] * random.uniform(0.7, 1.3)))
            new_p = [new_base, p[1], p[2]]
        elif random.random() < 0.6:
            # Change count
            new_count = max(1, int(p[1] * random.uniform(0.5, 1.5)))
            new_p = [p[0], new_count, p[2]]
        else:
            # Change scale
            new_scale = max(1, int(p[2] * random.uniform(0.7, 1.3)))
            new_p = [p[0], p[1], new_scale]
        
        # Ensure base >= 2
        new_p[0] = max(2, new_p[0])
        new_p[1] = max(1, new_p[1])
        new_p[2] = max(1, new_p[2])
        
        new_params = [cp[:] for cp in current_params]
        new_params[p_idx] = new_p
        
        new_U = _parametric_set(new_params)
        if 0 not in new_U:
            new_U.insert(0, 0)
        
        # Limit size to keep computation fast
        if len(new_U) > 500:
            T *= cooling
            continue
        
        new_score = _compute_score_fast(new_U)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_params = new_params
            current_U = new_U
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = new_U[:]
                best_params = [p[:] for p in new_params]
        
        T *= cooling
    
    return best_U, best_score

def _local_search(U, deadline, seed=123):
    """Local search: try adding/removing/swapping elements."""
    random.seed(seed)
    U = sorted(U)
    U_set = set(U)
    best_score = _compute_score_fast(U)
    best_U = U[:]
    
    while time.time() < deadline:
        # Try adding a random element
        if random.random() < 0.6 and len(U) < 200:
            max_u = max(U) if U else 0
            # Sample from a range that might be useful
            if max_u > 0:
                cand = random.randint(1, max_u * 2)
            else:
                cand = random.randint(1, 100)
            
            if cand not in U_set:
                new_U = sorted(U + [cand])
                new_score = _compute_score_fast(new_U)
                if new_score > best_score + 1e-12:
                    U = new_U
                    U_set.add(cand)
                    best_score = new_score
                    best_U = U[:]
        else:
            # Try removing an element
            if len(U) > 2:
                idx = random.randint(0, len(U) - 1)
                if U[idx] != 0:
                    removed = U[idx]
                    new_U = U[:idx] + U[idx+1:]
                    new_score = _compute_score_fast(new_U)
                    if new_score > best_score + 1e-12:
                        U = new_U
                        U_set.discard(removed)
                        best_score = new_score
                        best_U = U[:]
    
    return best_U, best_score

def _construct_with_budget():
    """Main construction with time budget management."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety
    
    best_U = [0]
    best_score = 1.0
    
    # Phase 1: Try several parametric constructions
    param_sets = [
        [(2, 15, 1)],       # Powers of 2
        [(3, 12, 1)],       # Powers of 3
        [(2, 10, 1), (3, 8, 100)],  # Mixed
        [(2, 8, 1), (5, 6, 50)],
        [(3, 10, 1), (2, 8, 1000)],
        [(4, 10, 1)],       # Powers of 4
        [(2, 12, 1), (2, 6, 10000)],
    ]
    
    for params in param_sets:
        if time.time() > deadline - 30:
            break
        U = _parametric_set(params)
        if 0 not in U:
            U.insert(0, 0)
        if len(U) > 300:
            continue
        score = _compute_score_fast(U)
        if score > best_score:
            best_score = score
            best_U = U[:]
    
    # Phase 2: Simulated annealing from best parametric set
    if time.time() < deadline - 20:
        # Extract approximate params from best_U
        if len(best_U) > 1:
            sa_deadline = deadline - 15
            U_sa, score_sa = _simulated_annealing([(2, 12, 1)], sa_deadline, seed=42)
            if score_sa > best_score:
                best_score = score_sa
                best_U = U_sa[:]
    
    # Phase 3: Local search refinement
    if time.time() < deadline - 5:
        U_ls, score_ls = _local_search(best_U, deadline - 3, seed=123)
        if score_ls > best_score:
            best_score = score_ls
            best_U = U_ls[:]
    
    # Phase 4: Try more SA runs with different seeds if time permits
    if time.time() < deadline - 10:
        for seed in [7, 99, 12345]:
            if time.time() > deadline - 5:
                break
            U_sa2, score_sa2 = _simulated_annealing([(3, 10, 1)], deadline - 3, seed=seed)
            if score_sa2 > best_score:
                best_score = score_sa2
                best_U = U_sa2[:]
    
    return sorted(best_U)

def construct():
    """Construct a finite set U of nonnegative integers containing 0.
    
    Returns a list of nonnegative integers.
    """
    try:
        result = _construct_with_budget()
        # Ensure 0 is in the result and all elements are nonnegative integers
        result = [x for x in result if isinstance(x, int) and x >= 0]
        if 0 not in result:
            result.insert(0, 0)
        return sorted(result)
    except Exception:
        # Fallback: return a simple valid set
        return [0, 1, 3]

# EVOLVE-BLOCK-START
# (construct is defined above; this block marker is kept for compatibility)
# EVOLVE-BLOCK-END
