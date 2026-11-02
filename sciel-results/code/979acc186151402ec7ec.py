"""Combined program: correct Strassen 2x2 blocking with verification and fallback."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix-multiplication tensor T[a, b, c] where
    a = i*m + j (A index), b = j'*p + k (B index), c = i'*n + k' (C index)
    T[a, b, c] = 1 iff i=i', j=j', k=k'
    """
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * n + k
                T[a, b, c] = 1.0
    return T


def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T"""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)


def _get_strassen_2x2_terms():
    """
    Returns U, V, W matrices for the rank-7 Strassen decomposition of 2x2 matrix multiplication.
    U shape: (4, 7) - coefficients for A = [[a, b], [c, d]]
    V shape: (4, 7) - coefficients for B = [[e, f], [g, h]]
    W shape: (4, 7) - coefficients for C = [[y, z], [w, x]]
    
    The decomposition: C[i,k] = sum_r U[i,r] * V[j,r] * W[k,r] summed over appropriate indices.
    More precisely: for each term r, the contribution to C is:
      C += W[:,r] * (U[:,r] dot A_flat) * (V[:,r] dot B_flat)
    
    Using the standard Strassen algorithm:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C[0,0] = M1 - M4 + M5 + M6
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M3 + M7
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # A_flat = [a, b, c, d] (row-major: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d)
    # B_flat = [e, f, g, h] (row-major: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h)
    # C_flat = [y, z, w, x] (row-major: C[0,0]=y, C[0,1]=z, C[1,0]=w, C[1,1]=x)
    
    # M1 = (a+d)(e+h)
    # U: a=1, d=1 -> U[0]=1, U[3]=1
    # V: e=1, h=1 -> V[0]=1, V[3]=1
    # W: contributes to C[0,0] and C[1,1] -> W[0]=1, W[3]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    # U: c=1, d=1 -> U[2]=1, U[3]=1
    # V: e=1 -> V[0]=1
    # W: contributes to C[1,0] -> W[2]=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = a(f-h)
    # U: a=1 -> U[0]=1
    # V: f=1, h=-1 -> V[1]=1, V[3]=-1
    # W: contributes to C[0,1] -> W[1]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = d(g-e)
    # U: d=1 -> U[3]=1
    # V: g=1, e=-1 -> V[2]=1, V[0]=-1
    # W: contributes to C[0,0] with -1, C[1,0] with +1 -> W[0]=-1, W[2]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5 = (a+b)h
    # U: a=1, b=1 -> U[0]=1, U[1]=1
    # V: h=1 -> V[3]=1
    # W: contributes to C[0,0] and C[0,1] -> W[0]=1, W[1]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    # U: c=1, a=-1 -> U[2]=1, U[0]=-1
    # V: e=1, f=1 -> V[0]=1, V[1]=1
    # W: contributes to C[0,0] -> W[0]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    
    # M7 = (b-d)(g+h)
    # U: b=1, d=-1 -> U[1]=1, U[3]=-1
    # V: g=1, h=1 -> V[2]=1, V[3]=1
    # W: contributes to C[1,1] -> W[3]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W


def _verify_strassen():
    """Verify the Strassen decomposition is correct by testing on random matrices."""
    U_str, V_str, W_str = _get_strassen_2x2_terms()
    # Test with several random 2x2 matrices
    for trial in range(100):
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        C_expected = A @ B
        # Compute C from decomposition
        # C_flat[i] = sum_r W[i, r] * (U[:, r] dot A_flat) * (V[:, r] dot B_flat)
        A_flat = A.flatten()  # [a, b, c, d]
        B_flat = B.flatten()  # [e, f, g, h]
        C_flat = np.zeros(4)
        for r in range(7):
            u_val = np.dot(U_str[:, r], A_flat)
            v_val = np.dot(V_str[:, r], B_flat)
            C_flat += W_str[:, r] * u_val * v_val
        C_computed = C_flat.reshape(2, 2)
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def construct():
    n, m, p = DIMENSIONS
    
    # First verify Strassen is correct
    if not _verify_strassen():
        # Fall back to schoolbook
        rank0 = n * m * p
        U_fb = np.zeros((n * m, rank0), dtype=np.float64)
        V_fb = np.zeros((m * p, rank0), dtype=np.float64)
        W_fb = np.zeros((p * n, rank0), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    U_str, V_str, W_str = _get_strassen_2x2_terms()
    
    # 2x4 * 4x6:
    # A is 2x4, B is 4x6, C is 2x6
    # Split A into two 2x2 blocks: A1 (cols 0-1), A2 (cols 2-3)
    # Split B into two 4x3 halves, then each into three 2x2 blocks:
    #   B1 (rows 0-1): B1_1 (cols 0-1), B1_2 (cols 2-3), B1_3 (cols 4-5)
    #   B2 (rows 2-3): B2_1 (cols 0-1), B2_2 (cols 2-3), B2_3 (cols 4-5)
    # C = A1*B1 + A2*B2
    # C[:, 0:2] = A1*B1_1 + A2*B2_1
    # C[:, 2:4] = A1*B1_2 + A2*B2_2
    # C[:, 4:6] = A1*B1_3 + A2*B2_3
    
    total_rank = 6 * 7  # 42
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    # For each of the 6 block pairs, assign 7 Strassen terms
    # Block pair (block_a, block_b) where block_a in {0,1} (A1 or A2), block_b in {0,1,2} (B_1, B_2, B_3)
    # A block 0: rows 0-1, cols 0-1 of A -> U indices: i*m+j for i in 0..1, j in 0..1 -> 0,1,4,5
    # A block 1: rows 0-1, cols 2-3 of A -> U indices: i*m+j for i in 0..1, j in 2..3 -> 2,3,6,7
    # B block (row_offset, col_offset): rows row_offset..row_offset+1, cols col_offset..col_offset+1
    #   B1_1: rows 0-1, cols 0-1 -> V indices: j*p+k for j in 0..1, k in 0..1 -> 0,1,6,7
    #   B1_2: rows 0-1, cols 2-3 -> V indices: j*p+k for j in 0..1, k in 2..3 -> 2,3,8,9
    #   B1_3: rows 0-1, cols 4-5 -> V indices: j*p+k for j in 0..1, k in 4..5 -> 4,5,10,11
    #   B2_1: rows 2-3, cols 0-1 -> V indices: j*p+k for j in 2..3, k in 0..1 -> 12,13,18,19
    #   B2_2: rows 2-3, cols 2-3 -> V indices: j*p+k for j in 2..3, k in 2..3 -> 14,15,20,21
    #   B2_3: rows 2-3, cols 4-5 -> V indices: j*p+k for j in 2..3, k in 4..5 -> 16,17,22,23
    # C block col_offset: cols col_offset..col_offset+1 of C
    #   C cols 0-1: W indices: k*n+i for k in 0..1, i in 0..1 -> 0,2,1,3
    #   C cols 2-3: W indices: k*n+i for k in 2..3, i in 0..1 -> 4,6,5,7
    #   C cols 4-5: W indices: k*n+i for k in 4..5, i in 0..1 -> 8,10,9,11
    
    # Define index mappings
    # A block 0 (A1): U row indices [0, 1, 4, 5] corresponding to [a, b, c, d]
    # A block 1 (A2): U row indices [2, 3, 6, 7] corresponding to [a, b, c, d]
    U_block0_rows = [0, 1, 4, 5]  # a, b, c, d
    U_block1_rows = [2, 3, 6, 7]  # a, b, c, d
    
    # B blocks: V row indices [e, f, g, h]
    V_B1_1_rows = [0, 1, 6, 7]    # e, f, g, h
    V_B1_2_rows = [2, 3, 8, 9]    # e, f, g, h
    V_B1_3_rows = [4, 5, 10, 11]  # e, f, g, h
    V_B2_1_rows = [12, 13, 18, 19]  # e, f, g, h
    V_B2_2_rows = [14, 15, 20, 21]  # e, f, g, h
    V_B2_3_rows = [16, 17, 22, 23]  # e, f, g, h
    
    # C blocks: W row indices [y, z, w, x]
    W_C0_rows = [0, 2, 1, 3]  # y, z, w, x (cols 0-1)
    W_C1_rows = [4, 6, 5, 7]  # y, z, w, x (cols 2-3)
    W_C2_rows = [8, 10, 9, 11]  # y, z, w, x (cols 4-5)
    
    # 6 block pairs:
    # (A1, B1_1) -> C cols 0-1
    # (A2, B2_1) -> C cols 0-1
    # (A1, B1_2) -> C cols 2-3
    # (A2, B2_2) -> C cols 2-3
    # (A1, B1_3) -> C cols 4-5
    # (A2, B2_3) -> C cols 4-5
    
    block_configs = [
        # (U_rows, V_rows, W_rows, term_offset)
        (U_block0_rows, V_B1_1_rows, W_C0_rows, 0),   # A1 * B1_1
        (U_block1_rows, V_B2_1_rows, W_C0_rows, 7),   # A2 * B2_1
        (U_block0_rows, V_B1_2_rows, W_C1_rows, 14),  # A1 * B1_2
        (U_block1_rows, V_B2_2_rows, W_C1_rows, 21),  # A2 * B2_2
        (U_block0_rows, V_B1_3_rows, W_C2_rows, 28),  # A1 * B1_3
        (U_block1_rows, V_B2_3_rows, W_C2_rows, 35),  # A2 * B2_3
    ]
    
    for U_rows, V_rows, W_rows, offset in block_configs:
        for r in range(7):
            for idx in range(4):
                U[U_rows[idx], offset + r] = U_str[idx, r]
                V[V_rows[idx], offset + r] = V_str[idx, r]
                W[W_rows[idx], offset + r] = W_str[idx, r]
    
    # Verify the full decomposition
    T = _build_tensor(n, m, p)
    if _check_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank0 = n * m * p
        U_fb = np.zeros((n * m, rank0), dtype=np.float64)
        V_fb = np.zeros((m * p, rank0), dtype=np.float64)
        W_fb = np.zeros((p * n, rank0), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
