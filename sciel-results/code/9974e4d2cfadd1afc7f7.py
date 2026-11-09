import numpy as np
import time
import sys
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import logging

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin for 600s limit

    # Try increasing K
    # K=2000 seems reasonable for LP size
    # Let's start with K=2000
    K = 2000
    
    # Variables u_k for k = 2 .. K
    n_vars = K - 1
    ks = np.arange(2, K + 1)
    
    # Objective coefficients c_k = log(k)/k
    # We want to minimize sum(u_k * c_k)
    c = np.log(ks) / ks
    
    # Constraints:
    # sum_{k=2}^K u_k * {x/k} >= -1.0001 for x in 1..10K
    # This is A_ub * u <= b_ub form? No, linprog takes A_ub x <= b_ub and A_eq x = b_eq.
    # We have A_lb x >= b_lb.
    # We can convert A_lb x >= b_lb to -A_lb x <= -b_lb.
    # Or use A_ub for the negative version.
    
    max_x = 10 * K
    
    # Build matrix B where B[x, k_idx] = { (x+1) / (k_idx+2) } ?
    # x ranges from 1 to max_x.
    # k ranges from 2 to K.
    
    # Building dense matrix of size max_x x n_vars
    # max_x = 20000, n_vars = 1999.
    # 20000 * 1999 = ~40M elements.
    # This is feasible.
    
    # Efficient construction of B
    # B is row x, col k.
    # B[x-1, k-2] = (x / k) % 1.0
    
    # We can compute this using broadcasting or loops.
    # Loop over k is better for memory locality if we build column by column?
    # Or just use numpy.
    
    # x_vals = np.arange(1, max_x + 1)[:, None]
    # k_vals = ks[None, :]
    # B = (x_vals / k_vals) % 1.0
    
    # This creates a 20000 x 1999 array.
    # Let's try it.
    
    x_vals = np.arange(1, max_x + 1, dtype=np.float64)
    
    # We need to construct A_ub for linprog.
    # Constraint: sum u_k B_{x,k} >= -1.0001
    # -sum u_k B_{x,k} <= 1.0001
    # So A_ub = -B, b_ub = 1.0001
    
    # To save memory, we can construct A_ub directly.
    # But B is needed for the calculation.
    
    # Let's construct B in chunks if memory is a concern, but 40M floats is ~320MB.
    # Should be fine.
    
    try:
        # Compute fractional parts
        # x_vals shape (max_x, 1)
        # ks shape (1, n_vars)
        # B shape (max_x, n_vars)
        
        # To avoid huge temporary arrays, we can do:
        # B = np.zeros((max_x, n_vars))
        # for i, k in enumerate(ks):
        #    B[:, i] = (x_vals / k) % 1.0
        
        # This loop is 2000 iterations of 20000 ops. 40M ops. Fast in numpy.
        B = np.empty((max_x, n_vars), dtype=np.float64)
        for i, k in enumerate(ks):
            B[:, i] = (x_vals / k) % 1.0
            
        A_ub = -B
        b_ub = np.full(max_x, 1.0001)
        
        # Bounds
        bounds = [(-10.0, 10.0) for _ in range(n_vars)]
        
        # Solve
        # Use 'highs' method for speed and robustness
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status != 0:
            # Fallback to heuristic if LP fails
            # Use Mobius heuristic
            return _mobius_heuristic(K)
        
        u = res.x
        
        # Verify constraints to be safe (floating point errors)
        # M(x) = - sum u_k B_{x,k}
        # We need M(x) <= 1.0001
        # M = - B @ u
        # max_M = np.max(-B @ u)
        # if max_M > 1.0001:
        #     # Scale down
        #     u = u * (1.0001 / max_M) * 0.99
        
        domain = ks.astype(np.int64)
        values = u.astype(np.float64)
        
        return domain, values
        
    except Exception as e:
        print(f"LP failed: {e}", file=sys.stderr)
        return _mobius_heuristic(K)

def _mobius_heuristic(K):
    # Fallback
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, K + 1):
        if is_prime[i]:
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
                if (j // (i * i)) * i * i == j:
                    mu[j] = 0
    ks = np.arange(2, K + 1)
    mus = mu[2:K + 1]
    L_K = np.sum(mus / ks)
    max_x = 10 * K
    x_vals = np.arange(1, max_x + 1)
    S_K = np.zeros(max_x, dtype=np.float64)
    for k, m in zip(ks, mus):
        S_K += m * (x_vals // k)
    A = S_K - x_vals * L_K
    pos_A = A[A > 0]
    if len(pos_A) > 0:
        c_max = np.min(1.0 / pos_A)
    else:
        c_max = 1.0
    c = c_max * 0.9
    f_vals = c * mus
    return ks.astype(np.int64), f_vals.astype(np.float64)

