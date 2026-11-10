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

def _verify_feasibility(f1, f_vals, ks, deadline):
    """Vectorized verification that M(x) <= 1.0001 for all x in [1, 10*K]."""
    K = int(ks.max())
    max_x = 10 * K
    # M(x) = f1 * x + sum_{k>=2} f(k) * floor(x/k)
    # Vectorize: for each x, compute floor(x/k) for all k
    xs = np.arange(1, max_x + 1)
    # floor(x/k) shape: (max_x, m)
    floors = np.floor(xs[:, None] / ks[None, :]).astype(np.int64)
    # M(x) = f1 * x + floors @ f_vals
    M_vals = f1 * xs + floors @ f_vals
    return np.max(M_vals) <= 1.0001

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    def try_K(K_max):
        """Try solving LP for given K_max. Returns (domain, values) or None."""
        if time.time() > deadline:
            return None
        ks = np.arange(2, K_max + 1)
        m = len(ks)
        
        # Objective: maximize -sum f_k log(k)/k
        c_obj = -(-np.log(ks) / ks)  # minimize sum f_k log(k)/k, so negate
        # Actually: maximize -sum f_k log(k)/k => minimize sum f_k log(k)/k
        # linprog minimizes c^T x, so c = log(ks)/ks
        c_obj = np.log(ks) / ks
        
        # Constraints: -sum f_k {x/k} <= 1 for x in 1..10*K_max
        max_x = 10 * K_max
        A_ub_list = []
        b_ub_list = []
        
        for x in range(1, max_x + 1):
            frac_parts = (x / ks) % 1.0
            A_ub_list.append(-frac_parts)
            b_ub_list.append(1.0)
            
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        
        # Bounds: -10 <= f_k <= 10
        bounds = [(-10, 10)] * m
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                f_vals = np.clip(res.x, -10, 10)
                S = np.sum(f_vals / ks)
                f1 = np.clip(-S, -10, 10)
                
                # Verify with deadline check
                if time.time() > deadline:
                    return None
                if _verify_feasibility(f1, f_vals, ks, deadline):
                    domain = np.concatenate(([1], ks))
                    values = np.concatenate(([f1], f_vals))
                    return domain, values
        except Exception:
            pass
        return None
    
    # Try progressively larger K_max values
    for K_max in [100, 150, 200, 300, 500]:
        result = try_K(K_max)
        if result is not None:
            return result
    
    # Fallback to Mobius
    K = 200000
    mu = _compute_mobius(K)
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain].astype(np.float64)
    return domain, values
