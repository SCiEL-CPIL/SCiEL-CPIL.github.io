import numpy as np
import time
from scipy.optimize import linprog

def _compute_mobius(K):
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600.0 - 10.0
    
    # Use a larger set of k to solve LP
    K_max = 200
    ks = np.arange(2, K_max + 1)
    m = len(ks)
    
    # Objective: maximize -sum f_k log(k)/k
    # c = -log(ks)/ks
    c = -np.log(ks) / ks
    
    # Constraints: M(x) = sum_{k>=2} f(k) * (floor(x/k) - x/k) <= 1
    c_obj = -c
    
    # Build constraint matrix efficiently
    # For each x, row is [floor(x/k) - x/k for k in ks]
    # We can vectorize this
    max_x = 10 * K_max
    
    # Precompute the constraint matrix using vectorized operations
    # x ranges from 1 to max_x, k ranges from 2 to K_max
    x_vals = np.arange(1, max_x + 1)
    # For each x, compute floor(x/k) - x/k for all k
    # This is a (max_x, m) matrix
    # floor(x/k) can be computed as x // k for integer x, k
    x_col = x_vals.reshape(-1, 1)  # (max_x, 1)
    k_row = ks.reshape(1, -1)  # (1, m)
    floor_part = x_col // k_row  # integer division
    frac_part = x_col.astype(np.float64) / k_row
    A_ub = floor_part.astype(np.float64) - frac_part  # (max_x, m)
    b_ub = np.ones(max_x)
    
    # Bounds: -10 <= f_k <= 10
    bounds = [(-10, 10)] * m
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'time_limit': 300})
        if res.success:
            f_vals = res.x
            # Construct domain and values
            # f(1) = -sum_{k>=2} f(k)/k
            f1 = -np.sum(f_vals / ks)
            
            domain = np.concatenate(([1], ks))
            values = np.concatenate(([f1], f_vals))
            
            # Verify: after clipping and f(1) replacement, M(x) <= 1
            # Clip values
            clipped = np.clip(values, -10, 10)
            S = np.sum(clipped / domain)
            # Replace f(1)
            idx1 = np.where(domain == 1)[0]
            if len(idx1) > 0:
                clipped[idx1[0]] = clipped[idx1[0]] - S
            else:
                # f(1) = 0 before replacement, so new f(1) = -S
                clipped = np.insert(clipped, 0, -S)
                domain = np.insert(domain, 0, 1)
            
            # Check M(x) using vectorized computation
            feasible = True
            K_check = int(domain.max())
            x_check = np.arange(1, 10 * K_check + 1)
            # M(x) = sum_k clipped[k] * floor(x/k)
            # Compute all M(x) at once
            x_col_check = x_check.reshape(-1, 1)
            d_row = domain.reshape(1, -1)
            floor_matrix = x_col_check // d_row
            M_vals = np.sum(clipped.reshape(1, -1) * floor_matrix, axis=1)
            if np.any(M_vals > 1.0001):
                feasible = False
            
            if feasible:
                return domain, values
        else:
            pass
    except Exception:
        pass
    
    # Fallback: try smaller K_max
    for K_try in [100, 50]:
        if time.time() > deadline:
            break
        ks_try = np.arange(2, K_try + 1)
        m_try = len(ks_try)
        c_try = -np.log(ks_try) / ks_try
        c_obj_try = -c_try
        
        max_x_try = 10 * K_try
        x_vals_try = np.arange(1, max_x_try + 1)
        x_col_try = x_vals_try.reshape(-1, 1)
        k_row_try = ks_try.reshape(1, -1)
        floor_part_try = x_col_try // k_row_try
        frac_part_try = x_col_try.astype(np.float64) / k_row_try
        A_ub_try = floor_part_try.astype(np.float64) - frac_part_try
        b_ub_try = np.ones(max_x_try)
        
        bounds_try = [(-10, 10)] * m_try
        
        try:
            res_try = linprog(c_obj_try, A_ub=A_ub_try, b_ub=b_ub_try, bounds=bounds_try,
                            method='highs', options={'presolve': True, 'time_limit': 100})
            if res_try.success:
                f_vals_try = res_try.x
                f1_try = -np.sum(f_vals_try / ks_try)
                
                domain_try = np.concatenate(([1], ks_try))
                values_try = np.concatenate(([f1_try], f_vals_try))
                
                clipped_try = np.clip(values_try, -10, 10)
                S_try = np.sum(clipped_try / domain_try)
                idx1_try = np.where(domain_try == 1)[0]
                if len(idx1_try) > 0:
                    clipped_try[idx1_try[0]] = clipped_try[idx1_try[0]] - S_try
                else:
                    clipped_try = np.insert(clipped_try, 0, -S_try)
                    domain_try = np.insert(domain_try, 0, 1)
                
                K_check_try = int(domain_try.max())
                x_check_try = np.arange(1, 10 * K_check_try + 1)
                x_col_check_try = x_check_try.reshape(-1, 1)
                d_row_try = domain_try.reshape(1, -1)
                floor_matrix_try = x_col_check_try // d_row_try
                M_vals_try = np.sum(clipped_try.reshape(1, -1) * floor_matrix_try, axis=1)
                
                if not np.any(M_vals_try > 1.0001):
                    return domain_try, values_try
        except Exception:
            pass
    
    # Fallback to Mobius
    K = 200000
    mu = _compute_mobius(K)
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain].astype(np.float64)
    return domain, values
