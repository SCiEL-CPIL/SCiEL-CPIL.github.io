# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_M_vectorized(domain, values, max_x):
    """
    Compute M(x) = sum_k f(k) * floor(x/k) for x in [1, max_x]
    using a vectorized difference-array approach.
    """
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for idx in range(len(domain)):
        k = int(domain[idx])
        f_k = values[idx]
        if f_k == 0.0:
            continue
        
        max_j = max_x // k
        if max_j == 0:
            continue
        
        starts = np.arange(1, max_j + 1, dtype=np.int64) * k
        ends = np.arange(2, max_j + 2, dtype=np.int64) * k
        ends = np.minimum(ends, max_x + 1)
        vals = f_k * np.arange(1, max_j + 1, dtype=np.float64)
        
        np.add.at(diff, starts, vals)
        np.add.at(diff, ends, -vals)
    
    M = np.cumsum(diff)
    return M[:max_x + 1]


def _try_constant_f(K, c, time_limit):
    """
    Try f(k) = -c for all k in 1..K.
    Returns (domain, values, a, valid) if feasible.
    """
    elapsed = time.time() - time_limit[0]
    if elapsed > 540:
        return None
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = np.full(K, -c, dtype=np.float64)
    
    # Clip to [-10, 10]
    values_clipped = np.clip(values, -10, 10)
    
    # Compute S = sum_k f(k)/k on clipped values
    S = np.sum(values_clipped / domain.astype(np.float64))
    
    # Apply S correction: f(1) = f(1) - S
    values_corrected = values_clipped.copy()
    values_corrected[0] = values_clipped[0] - S
    
    # Compute M(x) for x in [1, 10K]
    max_x = 10 * K
    M = _compute_M_vectorized(domain, values_corrected, max_x)
    
    # Check constraint
    if np.max(M[1:max_x + 1]) > 1.0001:
        return None
    
    # Compute objective
    k_float = domain.astype(np.float64)
    a = -np.sum(values_corrected * np.log(k_float) / k_float)
    
    return (domain, values_corrected, a)


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = 600.0
    time_limit = [start_time]

    best_result = None
    best_a = -np.inf

    # Strategy: use f(k) = -c for all k, with c chosen to maximize objective
    # while satisfying M(x) <= 1.0001 for all x in [1, 10K]
    
    # The binding constraint is typically M(1) = f'(1) <= 1.0001
    # f'(1) = -c - S where S = sum(-c/k) = -c*H_K
    # So f'(1) = -c + c*H_K = c*(H_K - 1)
    # For f'(1) <= 1.0001: c <= 1.0001/(H_K - 1)
    
    for K in [200000, 100000, 50000, 20000, 10000, 5000]:
        if time.time() - start_time > deadline - 90:
            break
        
        # Compute H_K approximately
        H_K = np.sum(1.0 / np.arange(1, K + 1, dtype=np.float64))
        
        # Maximum c from M(1) constraint
        c_max = 1.0001 / (H_K - 1) if H_K > 1 else 0.001
        
        # Try c = c_max (should be tightest)
        result = _try_constant_f(K, c_max, time_limit)
        if result is not None:
            domain, values, a = result
            if a > best_a:
                best_a = a
                best_result = (domain, values)
        
        # Also try slightly smaller c in case M(x) for x>1 is binding
        for c_frac in [0.99, 0.95, 0.9, 0.8, 0.7, 0.5]:
            if time.time() - start_time > deadline - 60:
                break
            c_try = c_max * c_frac
            result = _try_constant_f(K, c_try, time_limit)
            if result is not None:
                domain, values, a = result
                if a > best_a:
                    best_a = a
                    best_result = (domain, values)
        
        if time.time() - start_time > deadline - 60:
            break

    # Also try a hybrid: small negative for small k, larger negative for large k
    if time.time() - start_time < deadline - 120:
        K = 200000
        # Try: f(k) = -c1 for k <= K1, -c2 for k > K1
        # where c2 > c1 to get more benefit from log(k)/k for large k
        for K1_frac in [0.01, 0.05, 0.1, 0.2]:
            if time.time() - start_time > deadline - 60:
                break
            K1 = int(K * K1_frac)
            if K1 < 1:
                K1 = 1
            
            # For the two-region approach, we need to solve for c1, c2
            # such that M(x) <= 1.0001 for all x
            # This is more complex, so let's just try a few values
            
            H_K1 = np.sum(1.0 / np.arange(1, K1 + 1, dtype=np.float64)) if K1 > 0 else 0
            H_K = np.sum(1.0 / np.arange(1, K + 1, dtype=np.float64))
            H_K_minus_K1 = H_K - H_K1
            
            # S = -c1*H_K1 - c2*(H_K - H_K1)
            # f'(1) = -c1 - S = -c1 + c1*H_K1 + c2*(H_K - H_K1) = c1*(H_K1 - 1) + c2*(H_K - H_K1)
            # For f'(1) <= 1.0001
            
            # Try c1 = c2 = c (reduces to constant case, already handled)
            # Try c2 = 2*c1
            for ratio in [1.5, 2, 3, 5]:
                if time.time() - start_time > deadline - 30:
                    break
                # c1*(H_K1 - 1) + ratio*c1*(H_K - H_K1) <= 1.0001
                denom = (H_K1 - 1) + ratio * (H_K - H_K1)
                if denom <= 0:
                    c1_max = 0.01
                else:
                    c1_max = 1.0001 / denom
                
                c1 = c1_max
                c2 = ratio * c1
                
                # Build arrays
                domain = np.arange(1, K + 1, dtype=np.int64)
                values = np.empty(K, dtype=np.float64)
                values[:K1] = -c1
                values[K1:] = -c2
                
                values_clipped = np.clip(values, -10, 10)
                S = np.sum(values_clipped / domain.astype(np.float64))
                values_corrected = values_clipped.copy()
                values_corrected[0] = values_clipped[0] - S
                
                max_x = 10 * K
                M = _compute_M_vectorized(domain, values_corrected, max_x)
                
                if np.max(M[1:max_x + 1]) <= 1.0001:
                    k_float = domain.astype(np.float64)
                    a = -np.sum(values_corrected * np.log(k_float) / k_float)
                    if a > best_a:
                        best_a = a
                        best_result = (domain, values_corrected)

    if best_result is None:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_result


# EVOLVE-BLOCK-END
