"""Matrix multiplication 4,4,8 - low rank tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2_matrices():
    """Return U, V, W matrices (4x7) for 2x2x2 Strassen decomposition.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # m1 = (a+d)(e+h) -> C00:+1, C11:+1
    U[0,0] = 1; U[3,0] = 1   # a=A[0,0], d=A[1,1]
    V[0,0] = 1; V[3,0] = 1   # e=B[0,0], h=B[1,1]
    W[0,0] = 1; W[3,0] = 1   # C[0,0], C[1,1]
    
    # m2 = (c+d)e -> C10:+1, C11:-1
    U[2,1] = 1; U[3,1] = 1   # c=A[1,0], d=A[1,1]
    V[0,1] = 1               # e=B[0,0]
    W[2,1] = 1; W[3,1] = -1  # C[1,0], C[1,1]
    
    # m3 = a(f-h) -> C00:-1, C01:+1
    U[0,2] = 1               # a=A[0,0]
    V[1,2] = 1; V[3,2] = -1  # f=B[0,1], h=B[1,1]
    W[0,2] = -1; W[1,2] = 1  # C[0,0], C[0,1]
    
    # m4 = d(g-e) -> C00:+1, C10:+1
    U[3,3] = 1               # d=A[1,1]
    V[2,3] = 1; V[0,3] = -1  # g=B[1,0], e=B[0,0]
    W[0,3] = 1; W[2,3] = 1   # C[0,0], C[1,0]
    
    # m5 = (a+b)h -> C01:+1, C11:-1
    U[0,4] = 1; U[1,4] = 1   # a=A[0,0], b=A[0,1]
    V[3,4] = 1               # h=B[1,1]
    W[1,4] = 1; W[3,4] = -1  # C[0,1], C[1,1]
    
    # m6 = (c-a)(e+f) -> C10:+1, C11:+1
    U[2,5] = 1; U[0,5] = -1  # c=A[1,0], a=A[0,0]
    V[0,5] = 1; V[1,5] = 1   # e=B[0,0], f=B[0,1]
    W[2,5] = 1; W[3,5] = 1   # C[1,0], C[1,1]
    
    # m7 = (b-d)(g+h) -> C00:+1, C01:+1
    U[1,6] = 1; U[3,6] = -1  # b=A[0,1], d=A[1,1]
    V[2,6] = 1; V[3,6] = 1   # g=B[1,0], h=B[1,1]
    W[0,6] = 1; W[1,6] = 1   # C[0,0], C[0,1]
    
    return U, V, W

def _build_4x4x4_decomposition():
    """Build rank-49 decomposition for 4x4 * 4x4 using recursive Strassen.
    Returns U (16x49), V (16x49), W (16x49).
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    S_U, S_V, S_W = _strassen_2x2_matrices()
    
    # 7 block-level Strassen terms for 4x4:
    # Each term: (A_blocks, B_blocks, C_blocks) where blocks are (I,J) indices
    # A block (I,J): entries at A[I*2+i, J*2+j] for i,j in {0,1}
    # B block (J,K): entries at B[J*2+j, K*2+k] for j,k in {0,1}
    # C block (I,K): entries at C[I*2+i, K*2+k] for i,k in {0,1}
    
    block_terms = [
        # m1 = (A00+A11)(B00+B11) -> C00:+1, C11:+1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m2 = (A10+A11)B00 -> C10:+1, C11:-1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # m3 = A00(B01-B11) -> C00:-1, C01:+1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): -1, (0,1): 1}),
        # m4 = A11(B10-B00) -> C00:+1, C10:+1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # m5 = (A00+A01)B11 -> C01:+1, C11:-1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (1,1): -1}),
        # m6 = (A10-A00)(B00+B01) -> C10:+1, C11:+1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): 1}),
        # m7 = (A01-A11)(B10+B11) -> C00:+1, C01:+1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1}),
    ]
    
    for s, (A_blocks, B_blocks, C_blocks) in enumerate(block_terms):
        for t in range(7):
            col = s * 7 + t
            # U contribution
            for (I, J), a_coeff in A_blocks.items():
                for i in range(2):
                    for j in range(2):
                        u_row = (I * 2 + i) * m + (J * 2 + j)
                        U[u_row, col] += a_coeff * S_U[i * 2 + j, t]
            # V contribution
            for (J, K), b_coeff in B_blocks.items():
                for j in range(2):
                    for k in range(2):
                        v_row = (J * 2 + j) * p + (K * 2 + k)
                        V[v_row, col] += b_coeff * S_V[j * 2 + k, t]
            # W contribution
            for (I, K), c_coeff in C_blocks.items():
                for i in range(2):
                    for k in range(2):
                        w_row = (K * 2 + k) * n + (I * 2 + i)
                        W[w_row, col] += c_coeff * S_W[k * 2 + i, t]
    
    return U, V, W

def _build_4x4x8_decomposition():
    """Build rank-98 decomposition for 4x4 * 4x8.
    Split B into two 4x4 blocks, use 4x4x4 rank-49 for each.
    Returns U (16x98), V (32x98), W (32x98).
    """
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    U4, V4, W4 = _build_4x4x4_decomposition()
    
    for block in range(2):
        base_col = block * 49
        # U: same for both blocks
        U[:, base_col:base_col+49] = U4
        # V: B[j,k] with k offset by block*4
        for j in range(m):
            for k in range(4):
                V[j * p + (block * 4 + k), base_col:base_col+49] = V4[j * 4 + k, :]
        # W: C[i,k] with k offset by block*4, W index = k*n+i
        for i in range(n):
            for k in range(4):
                W[(block * 4 + k) * n + i, base_col:base_col+49] = W4[k * n + i, :]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    try:
        U, V, W = _build_4x4x8_decomposition()
        
        # Quick verification
        if time.time() < deadline - 10:
            np.random.seed(42)
            A = np.random.randn(4, 4)
            B = np.random.randn(4, 8)
            C_expected = A @ B
            
            U_np = U.astype(float)
            V_np = V.astype(float)
            W_np = W.astype(float)
            
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j]*U[i*m+j, r]) * (sum_j B[j,k]*V[j*p+k, r])
            # Compute efficiently with matrix operations
            # For each r, define:
            #   a_r[i] = sum_j A[i,j] * U[i*4+j, r]
            #   b_r[k] = sum_j B[j,k] * V[j*8+k, r]
            #   C[i,k] += W[k*4+i, r] * a_r[i] * b_r[k]
            
            # Reshape U to (4, 4, 98) where U_reshaped[i,j,r] = U[i*4+j, r]
            U_reshaped = U_np.reshape(4, 4, 98)  # (i, j, r)
            # a_r[i] = sum_j A[i,j] * U_reshaped[i,j,r]
            # = einsum('ij,ijr->ir', A, U_reshaped)
            a_terms = np.einsum('ij,ijr->ir', A, U_reshaped)  # (4, 98)
            
            # Reshape V to (4, 8, 98) where V_reshaped[j,k,r] = V[j*8+k, r]
            V_reshaped = V_np.reshape(4, 8, 98)  # (j, k, r)
            # b_r[k] = sum_j B[j,k] * V_reshaped[j,k,r]
            b_terms = np.einsum('jk,jkr->kr', B, V_reshaped)  # (8, 98)
            
            # Reshape W to (8, 4, 98) where W_reshaped[k,i,r] = W[k*4+i, r]
            W_reshaped = W_np.reshape(8, 4, 98)  # (k, i, r)
            
            # C[i,k] = sum_r W_reshaped[k,i,r] * a_terms[i,r] * b_terms[k,r]
            # = einsum('kir,ir,kr->ik', W_reshaped, a_terms, b_terms)
            C_computed = np.einsum('kir,ir,kr->ik', W_reshaped, a_terms, b_terms)
            
            error = np.max(np.abs(C_computed - C_expected))
            if error > 1e-9:
                # Fallback to schoolbook
                n, m, p = DIMENSIONS
                rank = n * m * p
                U = np.zeros((n * m, rank), dtype=int)
                V = np.zeros((m * p, rank), dtype=int)
                W = np.zeros((p * n, rank), dtype=int)
                column = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j, column] = 1
                            V[j * p + k, column] = 1
                            W[k * n + i, column] = 1
                            column += 1
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    except Exception:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
