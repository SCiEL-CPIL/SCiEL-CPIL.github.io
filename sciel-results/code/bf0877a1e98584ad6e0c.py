"""Matrix multiplication 3,3,7 tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize as opt

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _solve_3x3x2(rank=14, seed=42):
    """
    Find a rank-14 decomposition for 3x3 * 3x2 matrix multiplication.
    n=3, m=3, p=2.
    U: (9, rank)
    V: (6, rank)
    W: (6, rank)
    Condition: sum_r W[k*n+i, r] U[i*m+j1, r] V[j2*p+k, r] == delta(j1, j2)
    """
    n, m, p = 3, 3, 2
    np.random.seed(seed)
    
    # Flatten U, V, W into a single vector x
    # U: 9*rank, V: 6*rank, W: 6*rank
    total_vars = rank * (n*m + m*p + p*n)
    
    def residuals(x):
        U = x[:rank*n*m].reshape(rank, n*m).T
        V = x[rank*n*m:rank*(n*m+m*p)].reshape(rank, m*p).T
        W = x[rank*(n*m+m*p):].reshape(rank, p*n).T
        
        res = []
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        target = 1.0 if j1 == j2 else 0.0
                        actual = 0.0
                        # Vectorized sum over r
                        # W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r]
                        w_row = W[k*n+i]
                        u_row = U[i*m+j1]
                        v_row = V[j2*p+k]
                        actual = np.dot(w_row, u_row * v_row)
                        res.append(actual - target)
        return np.array(res)
    
    # Try a few seeds to ensure convergence
    best_result = None
    best_cost = np.inf
    
    for attempt in range(5):
        if _time_left() < 20:
            break
            
        x0 = np.random.randn(total_vars) * 0.1
        # Scale some variables differently if needed, but 0.1 is usually fine
        
        try:
            result = opt.least_squares(residuals, x0, method='trf', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            cost = np.sum(result.fun**2)
            if cost < best_cost:
                best_cost = cost
                best_result = result
                if cost < 1e-10:
                    break
        except Exception:
            continue
            
    if best_result is None or best_cost > 1e-8:
        return None
        
    x = best_result.x
    U = x[:rank*n*m].reshape(rank, n*m).T
    V = x[rank*n*m:rank*(n*m+m*p)].reshape(rank, m*p).T
    W = x[rank*(n*m+m*p):].reshape(rank, p*n).T
    
    # Convert to lists for the return interface
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    if _time_left() < 5:
        return _schoolbook(n, m, p)
    
    # Try to find rank-14 for 3x3 * 3x2
    decomp_3x2 = _solve_3x3x2(rank=14, seed=42)
    
    if decomp_3x2 is not None:
        U_3x2, V_3x2, W_3x2 = decomp_3x2
        rank_3x2 = 14
        rank_3x1 = 9 # Schoolbook for 3x3 * 3x1
        total_rank = rank_3x2 * 3 + rank_3x1
        
        U_total = [[0] * total_rank for _ in range(n * m)]
        V_total = [[0] * total_rank for _ in range(m * p)]
        W_total = [[0] * total_rank for _ in range(p * n)]
        
        col_offset = 0
        
        # Combine 3 blocks of 3x2
        for block in range(3):
            # Copy U
            for i in range(n * m):
                for r in range(rank_3x2):
                    U_total[i][col_offset + r] = U_3x2[i][r]
            
            # Copy V with shifted columns
            # V_3x2 shape (6, 14). Indices j*2+k.
            # Global V shape (21, total_rank). Indices j*7+k.
            for j in range(m):
                for local_k in range(2):
                    global_k = 2 * block + local_k
                    v_idx_global = j * p + global_k
                    v_idx_local = j * 2 + local_k
                    for r in range(rank_3x2):
                        V_total[v_idx_global][col_offset + r] = V_3x2[v_idx_local][r]
            
            # Copy W with shifted columns
            # W_3x2 shape (6, 14). Indices k*n+i.
            # Global W shape (21, total_rank). Indices k*n+i.
            for local_k in range(2):
                global_k = 2 * block + local_k
                for i in range(n):
                    w_idx_global = global_k * n + i
                    w_idx_local = local_k * n + i
                    for r in range(rank_3x2):
                        W_total[w_idx_global][col_offset + r] = W_3x2[w_idx_local][r]
            
            col_offset += rank_3x2
        
        # Block 4: 3x1 (schoolbook)
        # Rank 9.
        # U_3x1: (9, 9), V_3x1: (3, 9), W_3x1: (3, 9)
        # Schoolbook for 3x3 * 3x1:
        # r = i*m + j.
        # U[i*m+j, r] = 1 if r == i*m+j
        # V[j*1+k, r] = 1 if r == j (since k=0, V[j, r] = delta_{j,r}?)
        # Wait, for p=1, V index is j*1+0 = j.
        # Condition: sum_r W[0*n+i, r] U[i*m+j, r] V[j*1+0, r] = delta_{j, j'}
        # Target is delta_{j, j'}? No, the condition is for the whole block.
        # For the 3x1 block, the condition is:
        # sum_r W[i, r] U[i*m+j, r] V[j, r] = delta_{j, j'}
        # This is the identity matrix decomposition.
        # Standard schoolbook:
        # U[i*m+j, r] = 1 if r == i*m+j
        # V[j, r] = 1 if r == j? No, V has shape (3, 9).
        # If r = i*m+j, then V[j, r] should be 1 if j matches the j in r?
        # Let's check:
        # sum_r W[i, r] U[i*m+j, r] V[j, r]
        # If U[i*m+j, i*m+j'] = delta_{j,j'}, V[j, i*m+j'] = delta_{j,j'}, W[i, i*m+j'] = delta_{i,i'}?
        # This doesn't look right.
        
        # Let's use the standard schoolbook construction for 3x3x1.
        # Rank 9.
        # For each (i, j), we have one term.
        # U[i*m+j, r] = 1 if r == i*m+j
        # V[j*p+k, r] = 1 if r == j (k=0) -> V[j, r] = 1 if r == j?
        # No, r ranges 0..8.
        # Let r = i*m + j.
        # U[i*m+j, r] = 1.
        # V[j, r] = 1. (Since p=1, index is j*1+0 = j)
        # W[i, r] = 1. (Since p=1, index is 0*n+i = i)
        
        # Check condition:
        # sum_r W[i, r] U[i*m+j1, r] V[j2, r]
        # Only r = i*m+j1 is non-zero in U.
        # Term: W[i, i*m+j1] * U[i*m+j1, i*m+j1] * V[j2, i*m+j1]
        # = 1 * 1 * (1 if j2 == j1 else 0)
        # = delta_{j1, j2}.
        # This works.
        
        for i in range(n):
            for j in range(m):
                r_local = i * m + j
                r_global = col_offset + r_local
                
                U_total[i * m + j][r_global] = 1
                V_total[j * p + 6][r_global] = 1  # k=6 is the 7th column (index 6)
                W_total[6 * n + i][r_global] = 1
        
        return {"U": U_total, "V": V_total, "W": W_total}
        
    # Fallback to schoolbook 63
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
