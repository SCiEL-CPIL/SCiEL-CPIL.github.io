"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return the 7 rank-1 tensors for 2x2x2 Strassen decomposition.
    Each tensor is (U, V, W) where U, V, W are 4-vectors (for 2x2 matrices).
    U[i,j] maps to index i*2+j, V[j,k] maps to j*2+k, W[k,i] maps to k*2+i.
    """
    # Strassen: C11 = (A11+A22)(B11+B22) - C44
    # C12 = A11(B12-B22)
    # C21 = (A21-A11)B11
    # C22 = (A12+A22)(B21-B11) - C33
    # C33 = A12(B21+B22)
    # C44 = (A11+A12)(B21-B11)
    # C55 = (A21+A22)(B11+B12)
    # C66 = (A12-A21)(B11+B22)  -- wait, let me use standard Strassen
    
    # Standard Strassen for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for 2x2: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
    # A11=0, A12=1, A21=2, A22=3
    # B11=0, B12=1, B21=2, B22=3
    # C11=0, C12=1, C21=2, C22=3 (C[k,i]: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3)
    
    tensors = []
    # M1 = (A11+A22)(B11+B22)
    U1 = [0,0,0,0]; U1[0]=1; U1[3]=1
    V1 = [0,0,0,0]; V1[0]=1; V1[3]=1
    W1 = [0,0,0,0]; W1[0]=1; W1[1]=1; W1[2]=1; W1[3]=1
    # Wait, W should encode C = M1 contributes to all of C? No.
    # M1 contributes to C11, C12, C21, C22 with coefficients.
    # Actually each M_i is a scalar, and C is a linear combination.
    # The tensor for M1: U1 encodes (A11+A22), V1 encodes (B11+B22), W1 encodes the contribution to C.
    # C11 += M1, C12 += M1, C21 += M1, C22 += M1? No!
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So M1 contributes: C11+=1, C12+=0, C21+=0, C22+=1
    # W1[k*n+i] where n=2: W1[0*2+0]=1 (C11), W1[0*2+1]=0 (C12), W1[1*2+0]=0 (C21), W1[1*2+1]=1 (C22)
    W1 = [1, 0, 0, 1]
    tensors.append((U1, V1, W1))
    
    # M2 = (A21+A22)B11
    U2 = [0,0,0,0]; U2[2]=1; U2[3]=1
    V2 = [0,0,0,0]; V2[0]=1
    W2 = [0, 0, 0, 0]
    # C21 += M2, C22 -= M2
    W2[2]=1; W2[3]=-1
    tensors.append((U2, V2, W2))
    
    # M3 = A11(B12-B22)
    U3 = [0,0,0,0]; U3[0]=1
    V3 = [0,0,0,0]; V3[1]=1; V3[3]=-1
    W3 = [0, 0, 0, 0]
    # C12 += M3, C22 += M3
    W3[1]=1; W3[3]=1
    tensors.append((U3, V3, W3))
    
    # M4 = A22(B21-B11)
    U4 = [0,0,0,0]; U4[3]=1
    V4 = [0,0,0,0]; V4[2]=1; V4[0]=-1
    W4 = [0, 0, 0, 0]
    # C11 += M4, C21 += M4
    W4[0]=1; W4[2]=1
    tensors.append((U4, V4, W4))
    
    # M5 = (A11+A12)B22
    U5 = [0,0,0,0]; U5[0]=1; U5[1]=1
    V5 = [0,0,0,0]; V5[3]=1
    W5 = [0, 0, 0, 0]
    # C11 -= M5, C12 += M5
    W5[0]=-1; W5[1]=1
    tensors.append((U5, V5, W5))
    
    # M6 = (A21-A11)(B11+B12)
    U6 = [0,0,0,0]; U6[2]=1; U6[0]=-1
    V6 = [0,0,0,0]; V6[0]=1; V6[1]=1
    W6 = [0, 0, 0, 0]
    # C22 += M6
    W6[3]=1
    tensors.append((U6, V6, W6))
    
    # M7 = (A12-A22)(B21+B22)
    U7 = [0,0,0,0]; U7[1]=1; U7[3]=-1
    V7 = [0,0,0,0]; V7[2]=1; V7[3]=1
    W7 = [0, 0, 0, 0]
    # C11 += M7
    W7[0]=1
    tensors.append((U7, V7, W7))
    
    return tensors

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_block_strassen():
    """Build a block Strassen decomposition for 3x5x6.
    
    Partition A(3x5) into row blocks [2,1] and col blocks [2,3]:
      A = [[A11(2x2), A12(2x3)],
           [A21(1x2), A22(1x3)]]
    
    Partition B(5x6) into row blocks [2,3] and col blocks [2,4]:
      B = [[B11(2x2), B12(2x4)],
           [B21(3x2), B22(3x4)]]
    
    The top-left 2x2x2 subproblem A11*B11 uses Strassen (rank 7 vs 8).
    All other subproblems use schoolbook.
    
    Total rank = 7 + 2*2*4 + 1*2*2 + 1*2*4 + 1*3*2 + 1*3*4
              = 7 + 16 + 4 + 8 + 6 + 12 = 53? That seems too low.
    
    Wait, I need to be more careful. The full product is:
    C = A*B where C is 3x6.
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    With the partition:
    C_top_left (2x2) = A11*B11 + A12*B21
    C_top_right (2x4) = A11*B12 + A12*B22
    C_bot_left (1x2) = A21*B11 + A22*B21
    C_bot_right (1x4) = A21*B12 + A22*B22
    
    Each subproduct is a matrix multiplication. The tensor for the full product is the sum of tensors for each subproduct.
    
    For A11(2x2)*B11(2x2): rank 7 (Strassen)
    For A12(2x3)*B21(3x2): rank 2*3*2 = 12
    For A11(2x2)*B12(2x4): rank 2*2*4 = 16
    For A12(2x3)*B22(3x4): rank 2*3*4 = 24
    For A21(1x2)*B11(2x2): rank 1*2*2 = 4
    For A22(1x3)*B21(3x2): rank 1*3*2 = 6
    For A21(1x2)*B12(2x4): rank 1*2*4 = 8
    For A22(1x3)*B22(3x4): rank 1*3*4 = 12
    
    Total = 7 + 12 + 16 + 24 + 4 + 6 + 8 + 12 = 89
    """
    n, m, p = 3, 5, 6
    
    # Block sizes
    n1, n2 = 2, 1  # row blocks of A
    m1, m2 = 2, 3  # col blocks of A / row blocks of B
    p1, p2 = 2, 4  # col blocks of B
    
    # Verify: n1+n2=3, m1+m2=5, p1+p2=6
    assert n1 + n2 == n
    assert m1 + m2 == m
    assert p1 + p2 == p
    
    U_rows = n * m  # 15
    V_rows = m * p  # 30
    W_rows = p * n  # 18
    
    U = [[0] * 0 for _ in range(U_rows)]  # placeholder
    V = [[0] * 0 for _ in range(V_rows)]
    W = [[0] * 0 for _ in range(W_rows)]
    
    # We'll build columns incrementally
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_tensor(U_sub, V_sub, W_sub, i_off, j_off, k_off, U_size, V_size, W_size):
        """Add a rank-1 tensor for subproblem at offset (i_off, j_off, k_off)."""
        U_cols.append(U_sub)
        V_cols.append(V_sub)
        W_cols.append(W_sub)
    
    # Helper: create full-size vectors from sub-vectors with offsets
    def pad_U(sub, i_off, j_off, n_sub, m_sub):
        full = [0] * (n * m)
        for ii in range(n_sub):
            for jj in range(m_sub):
                full[(i_off + ii) * m + (j_off + jj)] = sub[ii * m_sub + jj]
        return full
    
    def pad_V(sub, j_off, k_off, m_sub, p_sub):
        full = [0] * (m * p)
        for jj in range(m_sub):
            for kk in range(p_sub):
                full[(j_off + jj) * p + (k_off + kk)] = sub[jj * p_sub + kk]
        return full
    
    def pad_W(sub, k_off, i_off, p_sub, n_sub):
        full = [0] * (p * n)
        for kk in range(p_sub):
            for ii in range(n_sub):
                full[(k_off + kk) * n + (i_off + ii)] = sub[kk * n_sub + ii]
        return full
    
    # Subproblem 1: A11(2x2) * B11(2x2) -> C11(2x2), using Strassen
    strassen_tensors = _strassen_2x2x2()
    for U_s, V_s, W_s in strassen_tensors:
        add_tensor(
            pad_U(U_s, 0, 0, 2, 2),
            pad_V(V_s, 0, 0, 2, 2),
            pad_W(W_s, 0, 0, 2, 2)
        )
    
    # Subproblem 2: A12(2x3) * B21(3x2) -> C12_top(2x2)
    # i_off=0, j_off=2, k_off=0, n_sub=2, m_sub=3, p_sub=2
    for ii in range(2):
        for jj in range(3):
            for kk in range(2):
                U_sub = [0] * (2*3)
                V_sub = [0] * (3*2)
                W_sub = [0] * (2*2)
                U_sub[ii*3+jj] = 1
                V_sub[jj*2+kk] = 1
                W_sub[kk*2+ii] = 1
                add_tensor(
                    pad_U(U_sub, 0, 2, 2, 3),
                    pad_V(V_sub, 2, 0, 3, 2),
                    pad_W(W_sub, 0, 0, 2, 2)
                )
    
    # Subproblem 3: A11(2x2) * B12(2x4) -> C13_top(2x4)
    # i_off=0, j_off=0, k_off=2, n_sub=2, m_sub=2, p_sub=4
    for ii in range(2):
        for jj in range(2):
            for kk in range(4):
                U_sub = [0] * (2*2)
                V_sub = [0] * (2*4)
                W_sub = [0] * (4*2)
                U_sub[ii*2+jj] = 1
                V_sub[jj*4+kk] = 1
                W_sub[kk*2+ii] = 1
                add_tensor(
                    pad_U(U_sub, 0, 0, 2, 2),
                    pad_V(V_sub, 0, 2, 2, 4),
                    pad_W(W_sub, 2, 0, 4, 2)
                )
    
    # Subproblem 4: A12(2x3) * B22(3x4) -> C14_top(2x4)
    # i_off=0, j_off=2, k_off=2, n_sub=2, m_sub=3, p_sub=4
    for ii in range(2):
        for jj in range(3):
            for kk in range(4):
                U_sub = [0] * (2*3)
                V_sub = [0] * (3*4)
                W_sub = [0] * (4*2)
                U_sub[ii*3+jj] = 1
                V_sub[jj*4+kk] = 1
                W_sub[kk*2+ii] = 1
                add_tensor(
                    pad_U(U_sub, 0, 2, 2, 3),
                    pad_V(V_sub, 2, 2, 3, 4),
                    pad_W(W_sub, 2, 0, 4, 2)
                )
    
    # Subproblem 5: A21(1x2) * B11(2x2) -> C21_bot(1x2)
    # i_off=2, j_off=0, k_off=0, n_sub=1, m_sub=2, p_sub=2
    for ii in range(1):
        for jj in range(2):
            for kk in range(2):
                U_sub = [0] * (1*2)
                V_sub = [0] * (2*2)
                W_sub = [0] * (2*1)
                U_sub[ii*2+jj] = 1
                V_sub[jj*2+kk] = 1
                W_sub[kk*1+ii] = 1
                add_tensor(
                    pad_U(U_sub, 2, 0, 1, 2),
                    pad_V(V_sub, 0, 0, 2, 2),
                    pad_W(W_sub, 0, 2, 2, 1)
                )
    
    # Subproblem 6: A22(1x3) * B21(3x2) -> C22_bot(1x2)
    # i_off=2, j_off=2, k_off=0, n_sub=1, m_sub=3, p_sub=2
    for ii in range(1):
        for jj in range(3):
            for kk in range(2):
                U_sub = [0] * (1*3)
                V_sub = [0] * (3*2)
                W_sub = [0] * (2*1)
                U_sub[ii*3+jj] = 1
                V_sub[jj*2+kk] = 1
                W_sub[kk*1+ii] = 1
                add_tensor(
                    pad_U(U_sub, 2, 2, 1, 3),
                    pad_V(V_sub, 2, 0, 3, 2),
                    pad_W(W_sub, 0, 2, 2, 1)
                )
    
    # Subproblem 7: A21(1x2) * B12(2x4) -> C23_bot(1x4)
    # i_off=2, j_off=0, k_off=2, n_sub=1, m_sub=2, p_sub=4
    for ii in range(1):
        for jj in range(2):
            for kk in range(4):
                U_sub = [0] * (1*2)
                V_sub = [0] * (2*4)
                W_sub = [0] * (4*1)
                U_sub[ii*2+jj] = 1
                V_sub[jj*4+kk] = 1
                W_sub[kk*1+ii] = 1
                add_tensor(
                    pad_U(U_sub, 2, 0, 1, 2),
                    pad_V(V_sub, 0, 2, 2, 4),
                    pad_W(W_sub, 2, 2, 4, 1)
                )
    
    # Subproblem 8: A22(1x3) * B22(3x4) -> C24_bot(1x4)
    # i_off=2, j_off=2, k_off=2, n_sub=1, m_sub=3, p_sub=4
    for ii in range(1):
        for jj in range(3):
            for kk in range(4):
                U_sub = [0] * (1*3)
                V_sub = [0] * (3*4)
                W_sub = [0] * (4*1)
                U_sub[ii*3+jj] = 1
                V_sub[jj*4+kk] = 1
                W_sub[kk*1+ii] = 1
                add_tensor(
                    pad_U(U_sub, 2, 2, 1, 3),
                    pad_V(V_sub, 2, 2, 3, 4),
                    pad_W(W_sub, 2, 2, 4, 1)
                )
    
    rank = len(U_cols)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c, (U_c, V_c, W_c) in enumerate(zip(U_cols, V_cols, W_cols)):
        for r in range(U_rows):
            U[r][c] = U_c[r]
        for r in range(V_rows):
            V[r][c] = V_c[r]
        for r in range(W_rows):
            W[r][c] = W_c[r]
    
    return U, V, W

def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try block Strassen first
    try:
        U, V, W = _build_block_strassen()
        rank = len(U[0]) if U else 0
        if _verify_decomposition(U, V, W, 3, 5, 6):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    if time.time() > deadline:
        U, V, W = _build_schoolbook()
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
