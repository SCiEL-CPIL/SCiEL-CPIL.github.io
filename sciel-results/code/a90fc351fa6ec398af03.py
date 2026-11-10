import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

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

def _build_constraint_matrix_sparse(ks, max_x):
    """
    Build sparse constraint matrix for M(x) <= 1 constraints.
    A[x-1, k_idx] = floor(x/k) - x/k
    Each row has non-zero entries only where floor(x/k) != x/k,
    which happens when k does not divide x evenly.
    """
    m = len(ks)
    rows = []
    cols = []
    vals = []
    
    for x in range(1, max_x + 1):
        for idx, k in enumerate(ks):
            floor_val = x // k
            frac_val = x / k
            diff = floor_val - frac_val
            if abs(diff) > 1e-15:
                rows.append(x - 1)
                cols.append(idx)
                vals.append(diff)
    
    A = csr_matrix((vals, (rows, cols)), shape=(max_x, m))
    return A

def _verify_solution(domain, values):
    """Verify that the solution satisfies M(x) <= 1.0001 for all x in [1, 10K]."""
    clipped = np.clip(values, -10, 10)
    S = np.sum(clipped / domain)
    idx1 = np.where(domain == 1)[0]
    if len(idx1) > 0:
        clipped = clipped.copy()
        clipped[idx1[0]] = clipped[idx1[0]] - S
    else:
        clipped = np.insert(clipped, 0, -S)
        domain = np.insert(domain, 0, 1)
    
    K_check = int(domain.max())
    # Use chunked computation to avoid memory issues
    max_x_check = 10 * K_check
    chunk_size = max(1, min(10000, max_x_check // max(1, len(domain) // 1000)))
    
    for start in range(1, max_x_check + 1, chunk_size):
        end = min(start + chunk_size, max_x_check + 1)
        x_vals = np.arange(start, end)
        # M(x) = sum_k clipped[k] * floor(x/k)
        # Compute in chunks over x
        M_vals = np.zeros(len(x_vals))
        for i, k in enumerate(domain):
            M_vals += clipped[i] * (x_vals // k)
        if np.any(M_vals > 1.0001):
            return False
    return True

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600.0 - 15.0
    
    # Try increasing K_max values with sparse LP
    K_max_candidates = [1000, 500, 200, 100, 50]
    
    for K_max in K_max_candidates:
        if time.time() > deadline:
            break
            
        ks = np.arange(2, K_max + 1)
        m = len(ks)
        
        # Objective: maximize -sum f_k log(k)/k
        c = -np.log(ks) / ks
        c_obj = -c  # minimize negative of objective
        
        max_x = 10 * K_max
        
        # Build sparse constraint matrix
        # A[x-1, k_idx] = floor(x/k) - x/k
        # Constraint: sum_k f(k) * (floor(x/k) - x/k) <= 1
        # But we need M(x) = sum_k f(k) * floor(x/k) <= 1
        # Since f(1) = -sum f(k)/k, M(x) = f(1)*x + sum_{k>=2} f(k)*floor(x/k)
        # = -x*sum f(k)/k + sum_{k>=2} f(k)*floor(x/k)
        # = sum_{k>=2} f(k)*(floor(x/k) - x/k)
        # So constraint is: sum_{k>=2} f(k)*(floor(x/k) - x/k) <= 1
        
        try:
            A_ub = _build_constraint_matrix_sparse(ks, max_x)
        except Exception:
            # Fallback to dense if sparse fails
            try:
                x_vals = np.arange(1, max_x + 1)
                x_col = x_vals.reshape(-1, 1)
                k_row = ks.reshape(1, -1)
                floor_part = x_col // k_row
                frac_part = x_col.astype(np.float64) / k_row
                A_ub = (floor_part.astype(np.float64) - frac_part)
            except Exception:
                continue
        
        b_ub = np.ones(max_x)
        bounds = [(-10, 10)] * m
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'time_limit': 200})
            if res.success:
                f_vals = res.x
                f1 = -np.sum(f_vals / ks)
                
                domain = np.concatenate(([1], ks))
                values = np.concatenate(([f1], f_vals))
                
                # Verify
                if _verify_solution(domain, values):
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
