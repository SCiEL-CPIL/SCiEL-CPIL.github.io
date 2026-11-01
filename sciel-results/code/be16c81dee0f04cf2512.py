"""
Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication.
Uses block Strassen algorithm to achieve rank 28, with search for lower rank.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16

def verify_numpy(U, V, W, n=N, m=M, p=P):
    """Fast verification using NumPy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Compute reconstructed tensor
    # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Using einsum
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exact match (with tolerance for floating point)
    return np.allclose(T_recon, T, atol=1e-9, rtol=1e-9)

def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    
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
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
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
    
    # Strassen terms for M_{2,2,2}
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

def try_rank_reduction(U_init, V_init, W_init, target_rank, max_iter=500, tol=1e-8):
    """
    Try to reduce rank by zeroing out columns and checking validity.
    Uses greedy elimination.
    """
    U = np.array(U_init, dtype=float).copy()
    V = np.array(V_init, dtype=float).copy()
    W = np.array(W_init, dtype=float).copy()
    
    current_rank = U.shape[1]
    
    for _ in range(current_rank - target_rank):
        # Try removing each column
        best_col = -1
        best_score = float('inf')
        
        for col in range(current_rank):
            # Temporarily remove column
            U_test = np.delete(U, col, axis=1)
            V_test = np.delete(V, col, axis=1)
            W_test = np.delete(W, col, axis=1)
            
            # Check if still valid
            if verify_numpy(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                # Column can be removed
                U = U_test
                V = V_test
                W = W_test
                current_rank -= 1
                break
        
        if best_col == -1 and current_rank > target_rank:
            break
    
    if current_rank <= target_rank:
        return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    """
    Entry point: return the best decomposition found.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    best_decomp = None
    best_rank = 32
    
    # Try rank 28 (Strassen)
    try:
        U, V, W = strassen_decomposition_228()
        if verify_numpy(U, V, W):
            best_rank = 28
            best_decomp = {"U": U, "V": V, "W": W}
            # Try to reduce further
            if time.time() < deadline - 5:
                reduced = try_rank_reduction(U, V, W, target_rank=24)
                if reduced is not None:
                    U_r, V_r, W_r = reduced
                    if verify_numpy(U_r, V_r, W_r):
                        best_rank = len(U_r[0])
                        best_decomp = {"U": U_r, "V": V_r, "W": W_r}
        else:
            # Fallback
            U, V, W = schoolbook()
            best_decomp = {"U": U, "V": V, "W": W}
    except Exception as e:
        print(f"Strassen decomposition failed: {e}")
        U, V, W = schoolbook()
        best_decomp = {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        return best_decomp
    
    return best_decomp
