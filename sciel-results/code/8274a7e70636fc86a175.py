import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    # 90s limit. Use 75s for safety.
    deadline = start_time + 75.0
    
    N = 256
    candidates = list(range(2, N + 1))
    n_total = len(candidates)
    
    # Identify "free" primes: primes p > N/2
    # These do not conflict with any other number in [2, N] except themselves.
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(math.isqrt(x)) + 1):
            if x % i == 0:
                return False
        return True
        
    free_primes = []
    free_sum = 0.0
    rest_indices = []
    
    for i, val in enumerate(candidates):
        if is_prime(val) and val > N // 2:
            free_primes.append(val)
            free_sum += 1.0 / val
        else:
            rest_indices.append(i)
            
    # The problem decomposes into:
    # 1. Sum of weights of free_primes (always included)
    # 2. MWIS on the induced subgraph of rest_indices
    
    n_rest = len(rest_indices)
    rest_vals = [candidates[i] for i in rest_indices]
    weights = [1.0 / val for val in rest_vals]
    
    # Build conflict graph for rest_indices
    # conflict_mask[i] is bitmask of conflicts within rest_indices
    conflict_mask = [0] * n_rest
    conflicts = []
    
    for i in range(n_rest):
        a = rest_vals[i]
        for j in range(i + 1, n_rest):
            b = rest_vals[j]
            # lcm(a, b) <= N
            g = math.gcd(a, b)
            if a * b <= N * g:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                conflicts.append((i, j))
                
    # Objective: Maximize sum(weights[i] * x[i])
    # Minimize -sum(weights[i] * x[i])
    c = np.array([-weights[i] for i in range(n_rest)])
    
    # Constraints: x[i] + x[j] <= 1
    m_conf = len(conflicts)
    if m_conf > 0:
        A = np.zeros((m_conf, n_rest))
        for k, (i, j) in enumerate(conflicts):
            A[k, i] = 1.0
            A[k, j] = 1.0
        lb = np.zeros(m_conf)
        ub = np.ones(m_conf)
    else:
        # No conflicts, take all
        A = np.zeros((1, n_rest))
        lb = np.array([0.0])
        ub = np.array([1.0])
        
    bounds = Bounds(lb=np.zeros(n_rest), ub=np.ones(n_rest))
    integrality = np.ones(n_rest)
    
    best_val_rest = 0.0
    best_mask_rest = 0
    
    # Fallback: Greedy + Local Search
    def greedy_local_search():
        # Greedy: sort by weight desc, add if no conflict
        order = sorted(range(n_rest), key=lambda i: -weights[i])
        mask = 0
        for i in order:
            if (conflict_mask[i] & mask) == 0:
                mask |= (1 << i)
        
        val = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            val += weights[i]
            m ^= lsb
            
        # Local Search: Try to add/remove/swap
        improved = True
        iters = 0
        max_iters = 5000
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            
            # Try adding
            for i in range(n_rest):
                if (mask >> i) & 1:
                    continue
                if (conflict_mask[i] & mask) == 0:
                    new_mask = mask | (1 << i)
                    new_val = val + weights[i]
                    if new_val > val + 1e-12:
                        mask = new_mask
                        val = new_val
                        improved = True
                        
            # Try removing (rarely useful for max weight, but might allow adding many)
            # Skip removal for now to keep it fast, addition is the main gain.
            # Try swapping: remove i, add j
            # This is O(N^2) per iteration. Might be too slow.
            # Stick to addition only for speed, which is a valid 1-improvement heuristic.
            
        return val, mask

    # Run MILP
    milp_success = False
    try:
        # Set time limit for HiGHS
        # Note: scipy milp doesn't expose time limit directly in older versions,
        # but we can rely on the global deadline check in the wrapper if we were looping.
        # However, milp blocks. We must trust it to finish or use a smaller instance.
        # With N=232, it should be fast.
        
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb=lb, ub=ub),
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 30.0} # HiGHS option
        )
        
        if res.success and res.x is not None:
            mask = 0
            val = 0.0
            for i in range(n_rest):
                if res.x[i] > 0.5:
                    mask |= (1 << i)
                    val += weights[i]
            # Verify validity
            valid = True
            m = mask
            while m:
                lsb = m & (-m)
                i = lsb.bit_length() - 1
                if (conflict_mask[i] & (mask ^ lsb)) != 0:
                    valid = False
                    break
                m ^= lsb
            if valid:
                best_val_rest = val
                best_mask_rest = mask
                milp_success = True
    except Exception:
        pass
    
    # If MILP failed or didn't improve, use greedy/local search
    if not milp_success:
        val, mask = greedy_local_search()
        if val > best_val_rest:
            best_val_rest = val
            best_mask_rest = mask
    
    # Construct final result
    result = free_primes[:]
    
    m = best_mask_rest
    while m:
        lsb = m & (-m)
        i = lsb.bit_length() - 1
        result.append(rest_vals[i])
        m ^= lsb
        
    if not result:
        result = [2]
        
    return result
