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

def _verify_feasibility(f1, f_vals, ks, max_x):
    """Vectorized verification that M(x) <= 1.0001 for all x in [1, max_x]."""
    # M(x) = f1 * x + sum_j f_vals[j] * floor(x / ks[j])
    # Vectorize: for each x, compute all floor(x/k) at once
    x_range = np.arange(1, max_x + 1)
    # floor(x/k) for all x, k: shape (max_x, m)
    floors = np.floor(np.outer(x_range, 1.0 / ks)).astype(np.int64)
    # M(x) = f1 * x + floors @ f_vals
    M_vals = f1 * x_range + floors @ f_vals
    return np.all(M_vals <= 1.0001)

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600.0 - 10.0
    
    best_result = None
    best_score = -np.inf
    
    # Try increasing K_max values
    K_max_list = [200, 300, 400, 500]
    
    for K_max in K_max_list:
        if time.time() > deadline - 30:  # Safety margin
            break
        
        ks = np.arange(2, K_max + 1)
        m = len(ks)
        
        # Objective: maximize -sum f_k log(k)/k
        # c = -log(ks)/ks, so c_obj = -c = log(ks)/ks for minimization
        c_obj = np.log(ks) / ks
        
        # Constraints: -sum f_k {x/k} <= 1 for x in 1..10K_max
        max_x = 10 * K_max
        
        # Vectorized constraint matrix construction
        x_range = np.arange(1, max_x + 1)
        # frac_parts[x, j] = (x / ks[j]) % 1
        frac_parts = np.mod(np.outer(x_range, 1.0 / ks), 1.0)
        A_ub = -frac_parts
        b_ub = np.ones(max_x)
        
        # Bounds: -10 <= f_k <= 10
        bounds = [(-10, 10)] * m
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                f_vals = np.clip(res.x, -10, 10)
                
                # f(1) is determined by sum f_k/k = 0
                S = np.sum(f_vals / ks)
                f1 = np.clip(-S, -10, 10)
                
                # Compute objective value
                obj_val = -np.sum(f_vals * np.log(ks) / ks)
                
                # Verify feasibility with vectorized check
                if time.time() > deadline - 15:
                    break
                    
                feasible = _verify_feasibility(f1, f_vals, ks, max_x)
                
                if feasible:
                    domain = np.concatenate(([1], ks))
                    values = np.concatenate(([f1], f_vals))
                    
                    if obj_val > best_score:
                        best_score = obj_val
                        best_result = (domain, values)
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: try with smaller K
    K_max = 50
    ks = np.arange(2, K_max + 1)
    m = len(ks)
    c_obj = np.log(ks) / ks
    
    max_x = 10 * K_max
    x_range = np.arange(1, max_x + 1)
    frac_parts = np.mod(np.outer(x_range, 1.0 / ks), 1.0)
    A_ub = -frac_parts
    b_ub = np.ones(max_x)
    bounds = [(-10, 10)] * m
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            f_vals = np.clip(res.x, -10, 10)
            S = np.sum(f_vals / ks)
            f1 = np.clip(-S, -10, 10)
            domain = np.concatenate(([1], ks))
            values = np.concatenate(([f1], f_vals))
            return domain, values
    except Exception:
        pass
    
    # Fallback to Mobius
    K = 200000
    mu = _compute_mobius(K)
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain].astype(np.float64)
    return domain, values
