"""
Optimized rank-77 decomposition for 4x4 x 4x6 matrix multiplication.
Uses Kronecker product of Strassen 2x2x2 (rank 7) and 2x2x3 (rank 11).
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _get_strassen_2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition (rank 7).
    Indices: row = i*2 + j for U/V (where j is col index of B in 2x2), 
    W index row = k*2 + i.
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_2x2x3_hybrid():
    """
    Returns U, V, W for 2x2x3 multiplication.
    U: 4x11, V: 6x11, W: 6x11.
    First 7 columns are Strassen for cols 0,1.
    Last 4 columns are schoolbook for col 2.
    """
    U1, V1, W1 = _get_strassen_2x2()
    
    U2 = np.zeros((4, 11), dtype=float)
    V2 = np.zeros((6, 11), dtype=float) # 2 rows, 3 cols -> 6 entries
    W2 = np.zeros((6, 11), dtype=float) # 3 rows, 2 cols -> 6 entries (k*2+i)
    
    # Copy Strassen part (cols 0,1 of B map to 0,1 and 3,4 in V2; 0,1,2,3 in W2)
    # B indices in V2: row = j*3 + k.
    # j=0, k=0 -> 0; j=0, k=1 -> 1; j=1, k=0 -> 3; j=1, k=1 -> 4.
    # W indices in W2: row = k*2 + i.
    # k=0, i=0 -> 0; k=0, i=1 -> 1; k=1, i=0 -> 2; k=1, i=1 -> 3.
    
    for c in range(7):
        U2[:, c] = U1[:, c]
        V2[0, c] = V1[0, c]
        V2[1, c] = V1[1, c]
        V2[3, c] = V1[2, c]
        V2[4, c] = V1[3, c]
        
        W2[0, c] = W1[0, c]
        W2[1, c] = W1[1, c]
        W2[2, c] = W1[2, c]
        W2[3, c] = W1[3, c]
        
    # Schoolbook for k=2 (col 2 of B)
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Term 7: A[0,0]B[0,2]
    U2[0, 7] = 1
    V2[2, 7] = 1 # j=0, k=2 -> 0*3+2=2
    W2[4, 7] = 1 # k=2, i=0 -> 2*2+0=4
    
    # Term 8: A[0,1]B[1,2]
    U2[1, 8] = 1
    V2[5, 8] = 1 # j=1, k=2 -> 1*3+2=5
    W2[4, 8] = 1 # k=2, i=0 -> 2*2+0=4
    
    # Term 9: A[1,0]B[0,2]
    U2[2, 9] = 1
    V2[2, 9] = 1 # j=0, k=2 -> 0*3+2=2
    W2[5, 9] = 1 # k=2, i=1 -> 2*2+1=5
    
    # Term 10: A[1,1]B[1,2]
    U2[3, 10] = 1
    V2[5, 10] = 1 # j=1, k=2 -> 1*3+2=5
    W2[5, 10] = 1 # k=2, i=1 -> 2*2+1=5
    
    return U2, V2, W2

def _kron_decomposition(n, m, p):
    """
    Constructs the full 4x4x6 decomposition using Kronecker products of 2x2x2 and 2x2x3.
    """
    U1, V1, W1 = _get_strassen_2x2()
    U2, V2, W2 = _get_2x2x3_hybrid()
    
    rank1 = U1.shape[1] # 7
    rank2 = U2.shape[1] # 11
    total_rank = rank1 * rank2 # 77
    
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((24, total_rank), dtype=float)
    W = np.zeros((24, total_rank), dtype=float)
    
    # Indices for 4x4, 4x6, 4x6
    # A (4x4): i = 2*i1 + i2, j = 2*j1 + j2. Row = i*4 + j
    # B (4x6): j = 2*j1 + j2, k = 3*k1 + k2. Row = j*6 + k
    # C (4x6): i = 2*i1 + i2, k = 3*k1 + k2. Row = k*4 + i
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c = c1 * rank2 + c2
            
            # U: (i1, j1) from U1, (i2, j2) from U2
            # U1 row: i1*2 + j1. U2 row: i2*2 + j2.
            # Full row: (i1*2 + i2)*4 + (j1*2 + j2)
            u1_vals = np.nonzero(U1[:, c1])[0]
            u2_vals = np.nonzero(U2[:, c2])[0]
            for idx1 in u1_vals:
                val1 = U1[idx1, c1]
                i1 = idx1 // 2
                j1 = idx1 % 2
                for idx2 in u2_vals:
                    val2 = U2[idx2, c2]
                    i2 = idx2 // 2
                    j2 = idx2 % 2
                    row = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                    U[row, c] = val1 * val2
            
            # V: (j1, k1) from V1, (j2, k2) from V2
            # V1 row: j1*2 + k1. V2 row: j2*3 + k2.
            # Full row: (j1*2 + j2)*6 + (k1*3 + k2)
            v1_vals = np.nonzero(V1[:, c1])[0]
            v2_vals = np.nonzero(V2[:, c2])[0]
            for idx1 in v1_vals:
                val1 = V1[idx1, c1]
                j1 = idx1 // 2
                k1 = idx1 % 2
                for idx2 in v2_vals:
                    val2 = V2[idx2, c2]
                    j2 = idx2 // 3
                    k2 = idx2 % 3
                    row = (j1 * 2 + j2) * 6 + (k1 * 3 + k2)
                    V[row, c] = val1 * val2
            
            # W: (k1, i1) from W1, (k2, i2) from W2
            # W1 row: k1*2 + i1. W2 row: k2*2 + i2.
            # Full row: (k1*3 + k2)*4 + (i1*2 + i2)
            w1_vals = np.nonzero(W1[:, c1])[0]
            w2_vals = np.nonzero(W2[:, c2])[0]
            for idx1 in w1_vals:
                val1 = W1[idx1, c1]
                k1 = idx1 // 2
                i1 = idx1 % 2
                for idx2 in w2_vals:
                    val2 = W2[idx2, c2]
                    k2 = idx2 // 2
                    i2 = idx2 % 2
                    row = (k1 * 3 + k2) * 4 + (i1 * 2 + i2)
                    W[row, c] = val1 * val2
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies the decomposition using random test cases.
    """
    np.random.seed(12345)
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            # alpha_r = sum(A * U_r)
            # beta_r = sum(B * V_r)
            # C += alpha_r * beta_r * W_r^T (transposed because W is p*n)
            
            # Using vectorized sum for speed
            alpha = np.sum(A * U[:, r].reshape(n, m))
            beta = np.sum(B * V[:, r].reshape(m, p))
            if alpha != 0 and beta != 0:
                C_computed += alpha * beta * W[:, r].reshape(p, n).T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _greedy_eliminate(U, V, W, n, m, p, deadline):
    """
    Attempts to reduce rank by greedily eliminating columns.
    Uses the Kronecker structure to efficiently check eliminability.
    """
    rank = U.shape[1]
    keep = [True] * rank
    
    # Exploit Kronecker structure: columns are (c1, c2) pairs
    # rank1 = 7 (from 2x2x2 Strassen), rank2 = 11 (from 2x2x3 hybrid)
    rank1 = 7
    rank2 = 11
    
    # First, check for redundant columns in each factor
    # A column (c1, c2) is eliminable if there exist coefficients such that
    # the tensor contribution matches. We can check this more efficiently.
    
    # Build coefficient tensors for all columns
    # T[r, i, j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    total_entries = n * m * p
    
    # Precompute flattened tensor contributions
    T = np.zeros((rank, total_entries))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U[i*m+j, r]
                if u_val == 0:
                    continue
                for k in range(p):
                    idx = (i*m + j)*p + k
                    T[r, idx] = u_val * V[j*p+k, r] * W[k*n+i, r]
    
    # Greedy elimination: try to eliminate each column
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        eliminated = False
        for c in range(rank):
            if not keep[c]:
                continue
            if time.time() > deadline - 5:
                break
            
            # Check if column c's tensor is in span of other kept columns
            kept_cols = [r for r in range(rank) if keep[r] and r != c]
            if not kept_cols:
                continue
            
            # Use SVD-based rank check: is T[c] in span of {T[r] for r in kept_cols}?
            # This is equivalent to checking if rank([T[c]; T[kept_cols]]) == rank(T[kept_cols])
            
            # Efficient check: compute projection of T[c] onto orthogonal complement of span(T[kept_cols])
            A = T[kept_cols].T  # shape: (total_entries, len(kept_cols))
            b = T[c]
            
            # Solve least squares: minimize ||A @ alpha - b||
            try:
                alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            except Exception:
                continue
            
            # Check residual
            residual = np.linalg.norm(A @ alpha - b)
            norm_b = np.linalg.norm(b)
            
            if norm_b > 1e-10 and residual / norm_b < 1e-8:
                # Column c is eliminable
                keep[c] = False
                eliminated = True
                break
        
        if not eliminated:
            break
    
    # Build reduced matrices
    kept_indices = [i for i in range(rank) if keep[i]]
    new_rank = len(kept_indices)
    
    if new_rank < rank:
        U_new = U[:, kept_indices]
        V_new = V[:, kept_indices]
        W_new = W[:, kept_indices]
        return U_new, V_new, W_new
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 110.0
    
    try:
        U, V, W = _kron_decomposition(n, m, p)
        
        if time.time() > deadline:
            raise TimeoutError("Time limit exceeded during construction")
        
        # Try greedy elimination
        U, V, W = _greedy_eliminate(U, V, W, n, m, p, deadline)
        
        if time.time() > deadline:
            raise TimeoutError("Time limit exceeded after elimination")
            
        # Additional verification with more test cases
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook if verification fails (should not happen)
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        # Fallback to schoolbook on any error
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
