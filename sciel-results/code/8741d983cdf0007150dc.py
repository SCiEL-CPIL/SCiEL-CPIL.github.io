"""
Rewritten program: LP relaxation + local search for minimizing max |sum| over APs.
Uses scipy.optimize.linprog to find a balanced continuous solution, then rounds and refines.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import linprog

_N = 256

def _generate_aps(n, min_len=2):
    """
    Generate all arithmetic progressions in [0, n-1] with length >= min_len.
    Returns:
        ap_indices: List of lists, where ap_indices[j] is a list of indices for the j-th AP.
    """
    ap_indices = []
    for d in range(1, n):
        for r in range(d):
            seq = list(range(r, n, d))
            if len(seq) >= min_len:
                ap_indices.append(seq)
    return ap_indices

def _precompute_ap_containing(ap_indices, n):
    """
    For each index k, find which APs contain it.
    Returns:
        ap_containing: List of lists, where ap_containing[k] is a list of AP indices j 
                       such that k is in ap_indices[j].
    """
    ap_containing = [[] for _ in range(n)]
    for j, indices in enumerate(ap_indices):
        for k in indices:
            ap_containing[k].append(j)
    return ap_containing

def _solve_lp(n, ap_indices, deadline):
    """
    Solve LP relaxation: minimize D subject to -D <= sum_{i in AP} x_i <= D for all APs,
    and -1 <= x_i <= 1 for all i.
    Returns continuous solution x* or None if timeout.
    """
    num_aps = len(ap_indices)
    # Variables: x_0, ..., x_{n-1}, D (total n+1 variables)
    # Objective: minimize D => c = [0, ..., 0, 1]
    c = np.zeros(n + 1)
    c[-1] = 1.0
    
    # Constraints:
    # For each AP j: sum_{i in AP_j} x_i - D <= 0  =>  row with 1s for AP indices, -1 for D
    # For each AP j: -sum_{i in AP_j} x_i - D <= 0 =>  row with -1s for AP indices, -1 for D
    A_ub = np.zeros((2 * num_aps, n + 1))
    b_ub = np.zeros(2 * num_aps)
    
    for j, indices in enumerate(ap_indices):
        # sum x_i - D <= 0
        A_ub[2*j, indices] = 1.0
        A_ub[2*j, -1] = -1.0
        # -sum x_i - D <= 0
        A_ub[2*j+1, indices] = -1.0
        A_ub[2*j+1, -1] = -1.0
    
    # Bounds: -1 <= x_i <= 1, D >= 0
    bounds = [(-1, 1)] * n + [(0, None)]
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            return result.x[:n]
    except Exception:
        pass
    
    return None

def _round_continuous(x_continuous, n):
    """
    Round continuous solution to signs using randomized rounding with bias.
    """
    signs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if x_continuous[i] >= 0:
            signs[i] = 1
        else:
            signs[i] = -1
    return signs

def _local_search(signs, ap_indices, ap_containing, deadline):
    """
    Local search to refine the sign assignment.
    Returns improved signs and max value.
    """
    n = _N
    num_aps = len(ap_indices)
    
    # Compute initial sums
    current_sums = np.zeros(num_aps, dtype=np.int32)
    for j, indices in enumerate(ap_indices):
        current_sums[j] = np.sum(signs[indices])
    
    current_max = int(np.max(np.abs(current_sums)))
    best_signs = signs.copy()
    best_max = current_max
    
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 1:
        # Try to find a position that can reduce the max
        pos = rng.randint(0, n - 1)
        
        old_sign = signs[pos]
        new_sign = -old_sign
        
        affected_aps = ap_containing[pos]
        if len(affected_aps) == 0:
            continue
            
        # Save old sums for affected APs
        old_sums = current_sums[affected_aps].copy()
        
        # Apply flip: change is -2 * old_sign
        delta = -2 * old_sign
        current_sums[affected_aps] += delta
        signs[pos] = new_sign
        
        # Check new max efficiently
        new_affected_max = int(np.max(np.abs(current_sums[affected_aps])))
        
        # Determine new global max for this solution
        if new_affected_max > current_max:
            new_max = new_affected_max
        else:
            new_max = current_max
        
        # Acceptance criteria
        if new_max < current_max:
            current_max = new_max
            no_improve_count = 0
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        elif new_max == current_max:
            if rng.random() < 0.3:
                no_improve_count += 1
            else:
                # Revert
                current_sums[affected_aps] = old_sums
                signs[pos] = old_sign
                no_improve_count += 1
        else:
            # Revert
            current_sums[affected_aps] = old_sums
            signs[pos] = old_sign
            no_improve_count += 1
        
        if no_improve_count >= max_no_improve:
            break
    
    return best_signs, best_max

def _construct_signs_lp():
    """
    Construct signs using LP relaxation + local search.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = _N
    ap_indices = _generate_aps(n, min_len=2)
    num_aps = len(ap_indices)
    
    if num_aps == 0:
        return [1] * n

    ap_containing = _precompute_ap_containing(ap_indices, n)
    
    global_best_signs = None
    global_best_max = float('inf')
    
    # Try LP relaxation first
    lp_solution = _solve_lp(n, ap_indices, deadline)
    
    if lp_solution is not None:
        # Round the LP solution
        signs = _round_continuous(lp_solution, n)
        # Refine with local search
        refined_signs, refined_max = _local_search(signs, ap_indices, ap_containing, deadline)
        global_best_signs = refined_signs
        global_best_max = refined_max
    
    # If LP failed or we have time, try quadratic character init + local search
    if time.time() < deadline - 3:
        signs = np.zeros(n, dtype=np.int8)
        p = 257
        for i in range(n):
            a = (i + 1) % p
            if a == 0:
                signs[i] = 1
            else:
                if pow(a, (p - 1) // 2, p) == 1:
                    signs[i] = 1
                else:
                    signs[i] = -1
        
        refined_signs, refined_max = _local_search(signs, ap_indices, ap_containing, deadline)
        
        if refined_max < global_best_max:
            global_best_max = refined_max
            global_best_signs = refined_signs
    
    # If still have time, try random init + local search
    if time.time() < deadline - 3:
        rng = random.Random(123)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        refined_signs, refined_max = _local_search(signs, ap_indices, ap_containing, deadline)
        
        if refined_max < global_best_max:
            global_best_max = refined_max
            global_best_signs = refined_signs
    
    if global_best_signs is None:
        return [1] * n
        
    return [int(x) for x in global_best_signs]

def construct():
    """
    Construct 256 signs (-1 or +1) minimizing the maximum absolute signed sum
    over any arithmetic progression in [1, 256].
    """
    # Use the LP + local search method
    signs = _construct_signs_lp()
    return signs

# EVOLVE-BLOCK-END
