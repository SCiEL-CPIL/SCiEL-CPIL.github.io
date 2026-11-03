"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses Kronecker-embedded Strassen (rank 65) as baseline, then ALS refinement with half-integer rounding."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64, checking tolerance."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=atol)

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _strassen_2x2x2_terms():
    """Return the 7 Strassen 2x2x2 terms as sparse (index, value) lists for U, V, W.
    U indices: (i,j) -> i*2+j (4 entries)
    V indices: (j,k) -> j*2+k (4 entries)
    W indices: (k,i) -> k*2+i (4 entries)
    """
    return [
        # M1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _kron_strassen_construct(n, m, p):
    """Construct a rank-65 decomposition using Kronecker product embedding.
    
    Splits 4x4x5 into 4x4x4 (handled by Kronecker-embedded Strassen, rank 49)
    plus 4x4x1 (handled naively, rank 16).
    Returns numpy arrays (U, V, W).
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    strassen_terms = _strassen_2x2x2_terms()
    
    # For 4x4x4: Kronecker product of two 2x2x2 Strassen decompositions
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[i * m + j] = val1 * val2
            
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[j * p + k] = val1 * val2
            
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[k * n + i] = val1 * val2
            
            add_term(u, v, w)
    
    # Add naive terms for the 5th column (k=4)
    for i in range(n):
        for j in range(m):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            u[i * m + j] = 1
            v[j * p + 4] = 1
            w[4 * n + i] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _als_solve_U(V, W, n, m, p):
    """Given V and W, solve for U via least squares.
    For each (i,j): U[i*m+j, :] should satisfy sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k.
    This is a linear system in U[i*m+j, :].
    """
    rank = V.shape[1]
    U = np.zeros((n * m, rank))
    for i in range(n):
        for j in range(m):
            # Build linear system: for each k, sum_c U[i*m+j,c] * (V[j*p+k,:] * W[k*n+i,:]) = 1
            # Let a_k = V[j*p+k,:] * W[k*n+i,:] (element-wise), shape (rank,)
            # We need: U[i*m+j,:] @ a_k = 1 for all k
            # This is: A @ x = b where A[k,c] = V[j*p+k,c]*W[k*n+i,c], b[k]=1
            A = np.zeros((p, rank))
            for k in range(p):
                A[k, :] = V[j * p + k, :] * W[k * n + i, :]
            b = np.ones(p)
            # Solve A @ x = b in least squares sense
            if rank <= p:
                # Underdetermined or square: use pinv
                x = np.linalg.lstsq(A, b, rcond=None)[0]
            else:
                x = np.linalg.lstsq(A, b, rcond=None)[0]
            U[i * m + j, :] = x
    return U

def _als_solve_V(U, W, n, m, p):
    """Given U and W, solve for V via least squares.
    For each (j,k): V[j*p+k, :] should satisfy sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i.
    """
    rank = W.shape[1]
    V = np.zeros((m * p, rank))
    for j in range(m):
        for k in range(p):
            # For each i: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Let a_i = U[i*m+j,:] * W[k*n+i,:] (element-wise)
            # Need: V[j*p+k,:] @ a_i = 1 for all i
            A = np.zeros((n, rank))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            b = np.ones(n)
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            V[j * p + k, :] = x
    return V

def _als_solve_W(U, V, n, m, p):
    """Given U and V, solve for W via least squares.
    For each (k,i): W[k*n+i, :] should satisfy sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j.
    """
    rank = U.shape[1]
    W = np.zeros((p * n, rank))
    for k in range(p):
        for i in range(n):
            # For each j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Let a_j = U[i*m+j,:] * V[j*p+k,:] (element-wise)
            # Need: W[k*n+i,:] @ a_j = 1 for all j
            A = np.zeros((m, rank))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b = np.ones(m)
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            W[k * n + i, :] = x
    return W

def _als_refine(U, V, W, n, m, p, max_iters=200, deadline=None, polish_rounds=50):
    """ALS refinement: alternate solving for U, V, W, then round to half-integers.
    Returns (U, V, W) as numpy arrays if a valid decomposition is found, else None.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        U = _als_solve_U(V, W, n, m, p)
        V = _als_solve_V(U, W, n, m, p)
        W = _als_solve_W(U, V, n, m, p)
        
        # Check convergence by residual
        if it % 10 == 0:
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        residual += (s - 1.0) ** 2
            if residual < 1e-12:
                break
    
    # Polish: try rounding to half-integers and verify
    for _ in range(polish_rounds):
        U_r = _round_matrix_to_half_int(U)
        V_r = _round_matrix_to_half_int(V)
        W_r = _round_matrix_to_half_int(W)
        if _verify_fast(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
        # Try smaller rounding tolerance
        break
    
    # Final check without rounding
    if _verify_fast(U, V, W, n, m, p):
        return U, V, W
    
    return None

def _try_rank_reduction(U_base, V_base, W_base, n, m, p, target_rank, deadline, seed=42):
    """Try to reduce rank from current to target_rank using ALS with random restarts.
    Strategy: take a subset of columns, zero out others, run ALS.
    """
    current_rank = U_base.shape[1]
    if target_rank >= current_rank:
        return None
    
    np.random.seed(seed)
    
    # Try multiple random subsets of columns
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
        
        # Select target_rank columns to keep
        cols = np.random.choice(current_rank, target_rank, replace=False)
        U_sub = U_base[:, cols].copy()
        V_sub = V_base[:, cols].copy()
        W_sub = W_base[:, cols].copy()
        
        result = _als_refine(U_sub, V_sub, W_sub, n, m, p, max_iters=100, deadline=deadline, polish_rounds=20)
        if result is not None:
            return result
    
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = 80  # schoolbook rank
    
    # Step 1: Build Kronecker-Strassen baseline (rank 65)
    try:
        U65, V65, W65 = _kron_strassen_construct(n, m, p)
        if _verify_fast(U65, V65, W65, n, m, p):
            best_result = {"U": U65.tolist(), "V": V65.tolist(), "W": W65.tolist()}
            best_rank = U65.shape[1]
    except Exception:
        pass
    
    # Step 2: Try ALS refinement on the baseline to see if rounding helps
    if best_result is not None and time.time() < deadline - 40:
        refined = _als_refine(U65, V65, W65, n, m, p, max_iters=100, deadline=deadline, polish_rounds=30)
        if refined is not None:
            U_r, V_r, W_r = refined
            r = U_r.shape[1]
            if r < best_rank:
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = r
    
    # Step 3: Try rank reduction via column subset + ALS
    if time.time() < deadline - 20:
        for target_rank in range(best_rank - 1, max(best_rank - 15, 1), -1):
            if time.time() > deadline - 10:
                break
            
            result = _try_rank_reduction(U65, V65, W65, n, m, p, target_rank, deadline, seed=target_rank * 137)
            if result is not None:
                U_r, V_r, W_r = result
                r = U_r.shape[1]
                if r < best_rank:
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = r
                    break  # Found a lower rank, stop
    
    # Fallback: schoolbook (rank 80)
    if best_result is None:
        return _schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
