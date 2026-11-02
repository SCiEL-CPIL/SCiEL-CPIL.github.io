import numpy as np
import time
from scipy.optimize import linprog

N = 100

def construct():
    deadline = time.time() + 110
    
    # Variables: x (N), M (1), d (N*N)
    n_vars = N + 1 + N * N
    
    # Objective: minimize M
    c = np.zeros(n_vars)
    c[N] = 1.0  # M is at index N
    
    # Inequality constraints A_ub @ v <= b_ub
    # We will build A_ub as a list of rows (sparse might be better, but dense is okay for 10k vars? 10k x 20k is 200M floats, too big. Use sparse or build carefully.)
    # Actually, let's use sparse matrices or build the dense matrix carefully if memory allows.
    # 10,101 vars.
    # Constraints:
    # 1. x_i <= 1 (N constraints)
    # 2. -x_i <= 0 (N constraints)
    # 3. -sum(x) <= -N/2 (1 constraint)
    # 4. sum(x) <= N/2 (1 constraint)
    # 5. -sum(d_k) - 2M <= -N (N constraints)
    # 6. -x_{i+k} - x_i - d_{k,i} <= -1 (N*N constraints)
    # 7. x_{i+k} + x_i - d_{k,i} <= 1 (N*N constraints)
    
    # Total constraints ~ 2 + 2N + 2N^2 = 20,202.
    # Matrix size 20202 x 10101.
    # This is about 200 million elements. 8 bytes each -> 1.6 GB. Too big for dense.
    # Must use sparse or a different solver approach.
    # scipy.optimize.linprog supports sparse A_ub.
    
    import scipy.sparse as sp
    
    rows = []
    cols = []
    vals = []
    b_ub = []
    
    # 1. x_i <= 1
    for i in range(N):
        rows.append(len(b_ub))
        cols.append(i)
        vals.append(1.0)
        b_ub.append(1.0)
        
    # 2. -x_i <= 0
    for i in range(N):
        rows.append(len(b_ub))
        cols.append(i)
        vals.append(-1.0)
        b_ub.append(0.0)
        
    # 3. -sum(x) <= -N/2
    for i in range(N):
        rows.append(len(b_ub))
        cols.append(i)
        vals.append(-1.0)
    b_ub.append(-N / 2.0)
    
    # 4. sum(x) <= N/2
    for i in range(N):
        rows.append(len(b_ub))
        cols.append(i)
        vals.append(1.0)
    b_ub.append(N / 2.0)
    
    # 5. -sum(d_k) - 2M <= -N
    for k in range(N):
        row_idx = len(b_ub)
        # -2 M
        rows.append(row_idx)
        cols.append(N)
        vals.append(-2.0)
        # -sum d_k
        for i in range(N):
            d_idx = N + 1 + k * N + i
            rows.append(row_idx)
            cols.append(d_idx)
            vals.append(-1.0)
        b_ub.append(-N)
        
    # 6. -x_{(i+k)%N} - x_i - d_{k,i} <= -1
    for k in range(N):
        for i in range(N):
            row_idx = len(b_ub)
            j = (i + k) % N
            rows.append(row_idx)
            cols.append(j)
            vals.append(-1.0)
            rows.append(row_idx)
            cols.append(i)
            vals.append(-1.0)
            d_idx = N + 1 + k * N + i
            rows.append(row_idx)
            cols.append(d_idx)
            vals.append(-1.0)
            b_ub.append(-1.0)
            
    # 7. x_{(i+k)%N} + x_i - d_{k,i} <= 1
    for k in range(N):
        for i in range(N):
            row_idx = len(b_ub)
            j = (i + k) % N
            rows.append(row_idx)
            cols.append(j)
            vals.append(1.0)
            rows.append(row_idx)
            cols.append(i)
            vals.append(1.0)
            d_idx = N + 1 + k * N + i
            rows.append(row_idx)
            cols.append(d_idx)
            vals.append(-1.0)
            b_ub.append(1.0)
            
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(len(b_ub), n_vars))
    
    # Bounds for variables
    # x_i in [0, 1]
    # M in [0, infinity) (overlap is non-negative, max is N/2=50)
    # d in [0, infinity)
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
    bounds.append((0, N/2)) # M upper bound
    for _ in range(N * N):
        bounds.append((0, None)) # d upper bound is 2, but None is safer
        
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            h = res.x[:N]
            # Ensure exact sum due to floating point errors
            h = np.clip(h, 0, 1)
            # Project to sum N/2
            # Simple projection
            diff = N/2 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            # Re-project to be safe
            # Use a simple water filling or just return if close enough
            return h
        else:
            # Fallback to random if LP fails
            h = np.random.uniform(0, 1, N)
            h = h / np.sum(h) * (N/2)
            return h
    except Exception:
        h = np.random.uniform(0, 1, N)
        h = h / np.sum(h) * (N/2)
        return h
