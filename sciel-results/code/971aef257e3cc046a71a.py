# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_M_vectorized(domain, values, max_x):
    """
    Compute M(x) = sum_k f(k) * floor(x/k) for x in [1, max_x].
    Uses direct range addition via difference array with slice ops.
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
        
        # floor(x/k) = j for x in [j*k, min((j+1)*k-1, max_x)]
        # We add f_k * j to diff[j*k] and subtract from diff[min((j+1)*k, max_x+1)]
        # But doing this per-j is expensive. Instead, use the fact that
        # sum over j of f_k*j * [x in [jk, (j+1)k-1]]
        # We can compute this as: for each block j, add f_k*j to range [jk, min((j+1)k, max_x+1))
        
        # For large k (few blocks), iterate over blocks
        # For small k (many blocks), the blocks are many but each is short
        if max_j <= 200:
            # Few blocks: iterate
            for j in range(1, max_j + 1):
                start = j * k
                end = min((j + 1) * k, max_x + 1)
                diff[start] += f_k * j
                diff[end] -= f_k * j
        else:
            # Many blocks: use vectorized approach
            starts = np.arange(1, max_j + 1, dtype=np.int64) * k
            ends = np.arange(2, max_j + 2, dtype=np.int64) * k
            ends = np.minimum(ends, max_x + 1)
            vals = f_k * np.arange(1, max_j + 1, dtype=np.float64)
            
            # Use np.add.at for starts and ends
            np.add.at(diff, starts, vals)
            np.add.at(diff, ends, -vals)
    
    M = np.cumsum(diff)
    return M[:max_x + 1]


def _try_profile(K, profile, time_limit):
    """
    Try f(k) = -profile(k) for k in 1..K.
    profile is a function or array of length K.
    Returns (domain, values_corrected, a) if feasible, else None.
    """
    elapsed = time.time() - time_limit[0]
    if elapsed > 540:
        return None
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    
    if callable(profile):
        values = np.array([-profile(k) for k in range(1, K + 1)], dtype=np.float64)
    else:
        values = -np.asarray(profile, dtype=np.float64)
    
    # Clip to [-10, 10]
    values_clipped = np.clip(values, -10, 10)
    
    # Compute S = sum_k f(k)/k on clipped values
    k_float = domain.astype(np.float64)
    S = np.sum(values_clipped / k_float)
    
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
    a = -np.sum(values_corrected * np.log(k_float) / k_float)
    
    return (domain, values_corrected, a)


def _try_constant_f(K, c, time_limit):
    """
    Try f(k) = -c for all k in 1..K.
    Returns (domain, values, a) if feasible.
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

    def update_best(result):
        nonlocal best_a, best_result
        if result is not None:
            domain, values, a = result
            if a > best_a:
                best_a = a
                best_result = (domain, values)

    # Strategy 1: constant f(k) = -c
    for K in [200000, 100000, 50000]:
        if time.time() - start_time > deadline - 120:
            break
        
        H_K = np.sum(1.0 / np.arange(1, K + 1, dtype=np.float64))
        c_max = 1.0001 / (H_K - 1) if H_K > 1 else 0.001
        
        result = _try_constant_f(K, c_max, time_limit)
        update_best(result)
        
        for c_frac in [0.95, 0.8, 0.6]:
            if time.time() - start_time > deadline - 90:
                break
            result = _try_constant_f(K, c_max * c_frac, time_limit)
            update_best(result)

    # Strategy 2: decaying profile f(k) = -c / k^alpha
    # The objective is a = sum c/k^alpha * log(k)/k = c * sum log(k)/k^(alpha+1)
    # The M(1) constraint: c*(H_K - 1) for alpha=0
    # For alpha > 0: S = -c * sum 1/k^(alpha+1), f(1) = -c + c*sum 1/k^(alpha+1)
    #   = c*(sum_{k=2}^K 1/k^(alpha+1))
    #   For large K and alpha>0, sum 1/k^(alpha+1) converges to zeta(alpha+1)-1
    #   So f(1) = c * (zeta(alpha+1) - 1) for large K
    #   Constraint: c * (zeta(alpha+1) - 1) <= 1.0001
    #   => c <= 1.0001 / (zeta(alpha+1) - 1)
    #   
    #   For alpha=0.5: zeta(1.5) ≈ 2.612, so c <= 1.0001/1.612 ≈ 0.62
    #   a = c * sum log(k)/k^1.5 ≈ c * integral_1^K log(x)/x^1.5 dx
    #   = c * [2*log(x)/x^0.5 + 4/(x^0.5) * (1 - log(x)*0) ] ... let me just compute numerically
    
    if time.time() - start_time < deadline - 120:
        K = 200000
        domain = np.arange(1, K + 1, dtype=np.int64)
        k_float = domain.astype(np.float64)
        
        for alpha in [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]:
            if time.time() - start_time > deadline - 60:
                break
            
            # Compute sum 1/k^(alpha+1) for the S correction
            if alpha == 0.0:
                sum_inv = np.sum(1.0 / k_float)
            else:
                sum_inv = np.sum(1.0 / k_float ** (alpha + 1))
            
            # f(1) after correction = -c + c * sum_inv = c * (sum_inv - 1)
            # Wait: f(1) = f_clipped(1) - S = -c/1^alpha - (-c * sum_inv) = -c + c*sum_inv = c*(sum_inv - 1)
            # Constraint: c*(sum_inv - 1) <= 1.0001
            c_max = 1.0001 / (sum_inv - 1) if sum_inv > 1 else 0.01
            
            # Also check M(x) for x > 1. For decaying profiles, M(x) should be fine.
            # Try c = c_max
            profile = c_max / k_float ** alpha
            result = _try_profile(K, profile, time_limit)
            update_best(result)
            
            # Try slightly smaller c
            for frac in [0.9, 0.7]:
                if time.time() - start_time > deadline - 30:
                    break
                profile = c_max * frac / k_float ** alpha
                result = _try_profile(K, profile, time_limit)
                update_best(result)

    # Strategy 3: two-region with different decay
    if time.time() - start_time < deadline - 60:
        K = 200000
        domain = np.arange(1, K + 1, dtype=np.int64)
        k_float = domain.astype(np.float64)
        
        for K1_frac in [0.05, 0.1, 0.2]:
            if time.time() - start_time > deadline - 30:
                break
            K1 = max(1, int(K * K1_frac))
            
            # Region 1: k <= K1, f(k) = -c1
            # Region 2: k > K1, f(k) = -c2/k
            # S = -c1*H_K1 - c2*(H_K - H_K1)
            # f(1) = -c1 + c1*H_K1 + c2*(H_K - H_K1) = c1*(H_K1-1) + c2*(H_K-H_K1)
            # Constraint: c1*(H_K1-1) + c2*(H_K-H_K1) <= 1.0001
            
            H_K1 = np.sum(1.0 / np.arange(1, K1 + 1, dtype=np.float64))
            H_K = np.sum(1.0 / k_float)
            H_rest = H_K - H_K1
            
            for ratio in [2, 3, 5, 10]:
                if time.time() - start_time > deadline - 15:
                    break
                # c1*(H_K1-1) + ratio*c1*H_rest <= 1.0001
                denom = (H_K1 - 1) + ratio * H_rest
                if denom <= 0:
                    c1_max = 0.01
                else:
                    c1_max = 1.0001 / denom
                
                c1 = c1_max
                c2 = ratio * c1
                
                profile = np.empty(K, dtype=np.float64)
                profile[:K1] = c1
                profile[K1:] = c2 / k_float[K1:]
                
                result = _try_profile(K, profile, time_limit)
                update_best(result)

    if best_result is None:
        # Fallback: small valid function
        K = 100
        domain = np.arange(1, K + 1, dtype=np.int64)
        c = 0.01
        values = np.full(K, -c, dtype=np.float64)
        k_float = domain.astype(np.float64)
        S = np.sum(values / k_float)
        values[0] -= S
        return domain, values

    return best_result


# EVOLVE-BLOCK-END
