import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16
SCHOOBOOK_RANK = N * M * P  # 32
DEADLINE = time.time() + 140

def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    # Build a set of valid (a,b,c) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True

def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def strassen_decomposition_228():
    """
    Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}.
    """
    n, m, p = N, M, P
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    for block_idx in range(4):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            for i in range(4):
                U[i][col] = u[i]
                
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * P + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]
                    
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * N + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]
                    
    return U, V, W

def _build_target_tensor():
    """Build the flattened target tensor T."""
    n, m, p = N, M, P
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    return T

def _tensor_error(params, T_flat, nm, mp, pn, rank):
    """Calculate the error for a given set of parameters."""
    # Unpack params into U, V, W
    size_u = nm * rank
    size_v = mp * rank
    U_flat = params[:size_u]
    V_flat = params[size_u:size_u+size_v]
    W_flat = params[size_u+size_v:]
    
    U = U_flat.reshape((nm, rank))
    V = V_flat.reshape((mp, rank))
    W = W_flat.reshape((pn, rank))
    
    # Compute tensor contraction
    # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is a 3-way tensor product.
    # We can compute it efficiently using einsum or loops.
    # For small dimensions, loops are fine, but einsum is faster.
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    
    error = np.sum((T_approx - T_flat.reshape((nm, mp, pn)))**2)
    return error

def _try_rank_search(target_rank, T, T_flat, nm, mp, pn, max_iters=5000):
    """Attempt to find a decomposition of the given rank."""
    if time.time() > DEADLINE:
        return None
        
    rank = target_rank
    size_u = nm * rank
    size_v = mp * rank
    total_size = size_u + size_v + size_v # W has same size as V here (16xrank)
    
    # Random initialization
    np.random.seed(42) # Fixed seed for reproducibility in this attempt, or random?
    # Let's use a few random restarts
    for attempt in range(3):
        if time.time() > DEADLINE - 10:
            break
            
        params0 = np.random.randn(total_size) * 0.1
        
        # Optimization
        # Using L-BFGS-B
        res = minimize(
            _tensor_error, 
            params0, 
            args=(T_flat, nm, mp, pn, rank),
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < 1e-6:
            # Extract solution
            U_flat = res.x[:size_u]
            V_flat = res.x[size_u:size_u+size_v]
            W_flat = res.x[size_u+size_v:]
            
            U = U_flat.reshape((nm, rank))
            V = V_flat.reshape((mp, rank))
            W = W_flat.reshape((pn, rank))
            
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Convert to list of lists for verify
            U_list = [[float(x) for x in row] for row in U_rounded]
            V_list = [[float(x) for x in row] for row in V_rounded]
            W_list = [[float(x) for x in row] for row in W_rounded]
            
            if verify(U_list, V_list, W_list):
                return U_list, V_list, W_list
                
    return None

def construct():
    """Entry point: return the best decomposition found."""
    T = _build_target_tensor()
    T_flat = T.flatten()
    nm, mp, pn = NM, MP, PN
    
    # Try rank 24 first
    if time.time() < DEADLINE - 30:
        res_24 = _try_rank_search(24, T, T_flat, nm, mp, pn)
        if res_24:
            return {"U": res_24[0], "V": res_24[1], "W": res_24[2]}
            
    # Try rank 25
    if time.time() < DEADLINE - 20:
        res_25 = _try_rank_search(25, T, T_flat, nm, mp, pn)
        if res_25:
            return {"U": res_25[0], "V": res_25[1], "W": res_25[2]}
            
    # Try rank 26
    if time.time() < DEADLINE - 10:
        res_26 = _try_rank_search(26, T, T_flat, nm, mp, pn)
        if res_26:
            return {"U": res_26[0], "V": res_26[1], "W": res_26[2]}
            
    # Fallback to rank 28
    U, V, W = strassen_decomposition_228()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
