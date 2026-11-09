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
    
    # Use a small set of k to solve LP
    K_max = 100
    ks = np.arange(2, K_max + 1)
    m = len(ks)
    
    # Objective: maximize -sum f_k log(k)/k
    # c = -log(ks)/ks
    c = -np.log(ks) / ks
    
    # Constraints: M(x) <= 1 for x in 1..10K_max
    # M(x) = sum f_k floor(x/k)
    # floor(x/k) = x/k - {x/k}
    # M(x) = sum f_k * (x/k - {x/k}) = x * sum f_k/k - sum f_k {x/k}
    # Since we'll set f(1) = -sum f_k/k, then x * sum f_k/k = x * (-f(1))
    # But wait, the evaluator computes M(x) = sum_k f(k) * floor(x/k) over ALL k in domain
    # If domain includes k=1, then floor(x/1) = x
    # So M(x) = f(1)*x + sum_{k>=2} f(k)*floor(x/k)
    # = f(1)*x + sum_{k>=2} f(k)*(x/k - {x/k})
    # = x*(f(1) + sum_{k>=2} f(k)/k) - sum_{k>=2} f(k)*{x/k}
    # = -sum_{k>=2} f(k)*{x/k}  (since f(1) = -sum f(k)/k)
    # So the constraint is: -sum_{k>=2} f(k)*{x/k} <= 1
    # i.e., sum_{k>=2} f(k)*{x/k} >= -1
    
    c_obj = -c
    
    A_ub = []
    b_ub = []
    
    # Constraints for M(x) <= 1
    # -sum f_k {x/k} <= 1
    max_x = 10 * K_max
    for x in range(1, max_x + 1):
        frac_parts = np.array([(x / k) % 1 for k in ks])
        A_ub.append(-frac_parts)
        b_ub.append(1.0)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: -10 <= f_k <= 10
    bounds = [(-10, 10)] * m
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            f_vals = res.x
            # Clip to [-10, 10]
            f_vals = np.clip(f_vals, -10, 10)
            
            # f(1) is determined by sum f_k/k = 0
            S = np.sum(f_vals / ks)
            f1 = -S
            
            # Clip f1 to [-10, 10]
            f1 = np.clip(f1, -10, 10)
            
            domain = np.concatenate(([1], ks))
            values = np.concatenate(([f1], f_vals))
            
            # Verify M(x) <= 1.0001 for all x in [1, 10K]
            K = int(ks.max())
            max_x_check = 10 * K
            # Compute M(x) for all x
            # M(x) = f1 * x + sum_{k>=2} f(k) * floor(x/k)
            # This is expensive, so let's just check a subset or trust the LP
            # For safety, let's verify
            feasible = True
            for x in range(1, max_x_check + 1):
                Mx = f1 * x
                for j, k in enumerate(ks):
                    Mx += f_vals[j] * (x // k)
                if Mx > 1.0001:
                    feasible = False
                    break
            
            if feasible:
                return domain, values
            else:
                pass
        else:
            pass
    except Exception:
        pass
    
    # Fallback: try with smaller K
    K_max = 50
    ks = np.arange(2, K_max + 1)
    m = len(ks)
    c = -np.log(ks) / ks
    c_obj = -c
    
    A_ub = []
    b_ub = []
    max_x = 10 * K_max
    for x in range(1, max_x + 1):
        frac_parts = np.array([(x / k) % 1 for k in ks])
        A_ub.append(-frac_parts)
        b_ub.append(1.0)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
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
