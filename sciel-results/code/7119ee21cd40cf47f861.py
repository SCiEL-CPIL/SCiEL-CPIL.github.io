import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0
    
    N = 256
    nums = list(range(2, 257))
    n = len(nums)
    
    # Build conflict bitmasks for fast checking
    # conflict_mask[i] is a bitmask where bit j is set if i conflicts with j
    conflict_mask = [0] * n
    conflicts = []
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                conflicts.append((i, j))
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Adjacency list for quick neighbor access
    adj = [set() for _ in range(n)]
    for i, j in conflicts:
        adj[i].add(j)
        adj[j].add(i)
    
    # Precompute weights
    weights = [1.0 / nums[i] for i in range(n)]
    
    # Precompute sorted candidate lists for local search
    # For each vertex i, precompute list of non-neighbors sorted by weight descending
    non_neighbors = [sorted([j for j in range(n) if j != i and (conflict_mask[i] >> j) & 1 == 0], 
                            key=lambda j: -weights[j]) for i in range(n)]
    
    # Check if a set (as bitmask) is valid
    def is_valid_bitmask(mask):
        # For each bit set in mask, check it doesn't conflict with any other bit
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            # Check if i conflicts with any other bit in mask (excluding i itself)
            if (conflict_mask[i] & (mask ^ lsb)) != 0:
                return False
            m ^= lsb
        return True
    
    def sum_weight_bitmask(mask):
        total = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            total += weights[i]
            m ^= lsb
        return total
    
    # Try MIP first
    best_mask = 0
    best_val = 0.0
    
    c = np.array([-weights[i] for i in range(n)])
    m_conf = len(conflicts)
    if m_conf > 0:
        A = np.zeros((m_conf, n))
        for k, (i, j) in enumerate(conflicts):
            A[k, i] = 1.0
            A[k, j] = 1.0
        lb = np.zeros(m_conf)
        ub = np.ones(m_conf)
        constraints = LinearConstraint(A, lb=lb, ub=ub)
    else:
        constraints = LinearConstraint(np.zeros((1, n)), lb=0, ub=1)
    
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    integrality = np.ones(n)
    
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if result is not None and result.success and result.x is not None:
            mask = 0
            for i in range(n):
                if result.x[i] > 0.5:
                    mask |= (1 << i)
            if is_valid_bitmask(mask):
                best_mask = mask
                best_val = sum_weight_bitmask(mask)
    except Exception:
        pass
    
    # If MIP didn't give a good result, try greedy
    if best_val < 0.1:
        # Greedy by weight descending
        order = sorted(range(n), key=lambda i: -weights[i])
        mask = 0
        for i in order:
            if (conflict_mask[i] & mask) == 0:
                mask |= (1 << i)
        if is_valid_bitmask(mask):
            val = sum_weight_bitmask(mask)
            if val > best_val:
                best_mask = mask
                best_val = val
    
    # Local search with bitmask operations - optimized
    def local_search(mask, max_time):
        nonlocal best_mask, best_val
        current_mask = mask
        current_val = sum_weight_bitmask(current_mask)
        improved = True
        iters = 0
        max_iters = 5000
        
        while improved and iters < max_iters and time.time() < max_time:
            improved = False
            iters += 1
            
            # Try adding each non-selected element (use precomputed non_neighbors)
            for i in range(n):
                if (current_mask >> i) & 1:
                    continue
                if (conflict_mask[i] & current_mask) == 0:
                    current_mask = current_mask | (1 << i)
                    current_val += weights[i]
                    improved = True
                    if current_val > best_val:
                        best_val = current_val
                        best_mask = current_mask
                    break
            
            # Try swapping: remove one, add another (use precomputed non_neighbors)
            if not improved:
                for i in range(n):
                    if not ((current_mask >> i) & 1):
                        continue
                    temp_mask = current_mask ^ (1 << i)
                    # Try candidates from non_neighbors[i] sorted by weight
                    for j in non_neighbors[i]:
                        if (current_mask >> j) & 1:
                            continue
                        if (conflict_mask[j] & temp_mask) == 0:
                            current_mask = temp_mask | (1 << j)
                            current_val = current_val - weights[i] + weights[j]
                            improved = True
                            if current_val > best_val:
                                best_val = current_val
                                best_mask = current_mask
                            break
                    if improved:
                        break
            
            # Try 2-swap: remove two, add two (more expensive but better exploration)
            if not improved and time.time() < max_time - 3:
                # Only try every 3 iterations
                if iters % 3 == 0:
                    found_improvement = False
                    # Get list of selected elements
                    selected = [i for i in range(n) if (current_mask >> i) & 1]
                    for idx_i, i in enumerate(selected):
                        if found_improvement:
                            break
                        for idx_j, j in enumerate(selected[idx_i+1:], idx_i+1):
                            if found_improvement:
                                break
                            temp_mask = current_mask ^ (1 << i) ^ (1 << j)
                            # Try to find two good additions from non-neighbors of temp_mask
                            candidates = []
                            for k in range(n):
                                if (temp_mask >> k) & 1:
                                    continue
                                if (conflict_mask[k] & temp_mask) == 0:
                                    candidates.append(k)
                                    if len(candidates) == 10:  # Limit candidates
                                        break
                            # Try pairs from candidates sorted by weight
                            candidates.sort(key=lambda x: -weights[x])
                            for a_idx, k1 in enumerate(candidates):
                                if found_improvement:
                                    break
                                for k2 in candidates[a_idx+1:]:
                                    if (conflict_mask[k1] & (1 << k2)) == 0:
                                        new_val = current_val - weights[i] - weights[j] + weights[k1] + weights[k2]
                                        if new_val > current_val + 1e-12:
                                            current_mask = temp_mask | (1 << k1) | (1 << k2)
                                            current_val = new_val
                                            improved = True
                                            found_improvement = True
                                            if current_val > best_val:
                                                best_val = current_val
                                                best_mask = current_mask
                                            break
        
        return current_mask, current_val
    
    # Run local search from MIP/greedy solution
    if time.time() < DEADLINE - 5:
        best_mask, best_val = local_search(best_mask, DEADLINE - 5)
    
    # Try multiple greedy starts and local search
    greedy_strategies = [
        lambda i: -weights[i],  # by weight
        lambda i: -weights[i] / (len(adj[i]) + 1),  # weight/degree
        lambda i: weights[i] * (len(adj[i]) + 1),  # weight*degree
        lambda i: -weights[i] * (len(adj[i]) + 1),  # prefer low-degree high-weight
    ]
    
    for strat_idx, strategy in enumerate(greedy_strategies):
        if time.time() >= DEADLINE - 15:
            break
        order = sorted(range(n), key=strategy)
        mask = 0
        for i in order:
            if (conflict_mask[i] & mask) == 0:
                mask |= (1 << i)
        if is_valid_bitmask(mask):
            val = sum_weight_bitmask(mask)
            if val > best_val:
                best_val = val
                best_mask = mask
            # Local search from this start
            if time.time() < DEADLINE - 8:
                new_mask, new_val = local_search(mask, DEADLINE - 8)
                if new_val > best_val:
                    best_val = new_val
                    best_mask = new_mask
    
    # Final verification and extract result
    result_list = []
    m = best_mask
    while m:
        lsb = m & (-m)
        i = lsb.bit_length() - 1
        result_list.append(nums[i])
        m ^= lsb
    
    # Verify validity
    valid = True
    for idx_a in range(len(result_list)):
        for idx_b in range(idx_a + 1, len(result_list)):
            a, b = result_list[idx_a], result_list[idx_b]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                valid = False
                break
        if not valid:
            break
    
    if not valid or not result_list:
        # Fall back to simple greedy
        order = sorted(range(n), key=lambda i: -weights[i])
        mask = 0
        for i in order:
            if (conflict_mask[i] & mask) == 0:
                mask |= (1 << i)
        result_list = []
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            result_list.append(nums[i])
            m ^= lsb
    
    if not result_list:
        result_list = [2]
    
    return result_list
