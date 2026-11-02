"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _strassen_2x2_core():
    """
    Strassen's algorithm for 2x2 matrix multiplication.
    Returns 7 tensors (each a 2x2x2 tensor) that decompose the 2x2x2x2x2x2 multiplication tensor.
    
    For A (2x2) * B (2x2) = C (2x2):
    m1 = (a11 + a22) * (b11 + b22)
    m2 = (a21 + a22) * b11
    m3 = a11 * (b12 - b22)
    m4 = a22 * (b21 - b11)
    m5 = (a11 + a12) * b22
    m6 = (a21 - a11) * (b11 + b12)
    m7 = (a12 - a22) * (b21 + b22)
    
    C11 = m1 + m4 - m5 + m7
    C12 = m3 + m5
    C21 = m2 + m4
    C22 = m1 - m2 + m3 + m6
    """
    # Each tensor is (i, j, k) -> coefficient for A[i,j] * B[j,k] -> C[i,k]
    # We represent each of the 7 rank-1 terms as (U_term, V_term, W_term)
    # where U_term is 2x2, V_term is 2x2, W_term is 2x2
    # and sum over terms of outer(U_term, V_term, W_term) = multiplication tensor
    
    # m1 = (a11+a22)(b11+b22)
    # U1: rows of A: row1=[1,0], row2=[0,1] -> actually U1[i,j] = 1 if i==j (diagonal)
    # Wait, let me be more careful.
    # m1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # U1[0,0]=1, U1[1,1]=1, others 0
    # V1[0,0]=1, V1[1,1]=1, others 0
    # W1[0,0]=1, W1[1,1]=1, others 0
    
    U1 = np.array([[1,0],[0,1]], dtype=float)
    V1 = np.array([[1,0],[0,1]], dtype=float)
    W1 = np.array([[1,0],[0,1]], dtype=float)
    
    # m2 = (A[1,0] + A[1,1]) * B[0,0]
    # U2[1,0]=1, U2[1,1]=1
    # V2[0,0]=1
    # W2[1,0]=1
    U2 = np.array([[0,0],[1,1]], dtype=float)
    V2 = np.array([[1,0],[0,0]], dtype=float)
    W2 = np.array([[0,0],[1,0]], dtype=float)
    
    # m3 = A[0,0] * (B[0,1] - B[1,1])
    # U3[0,0]=1
    # V3[0,1]=1, V3[1,1]=-1
    # W3[0,1]=1
    U3 = np.array([[1,0],[0,0]], dtype=float)
    V3 = np.array([[0,1],[0,-1]], dtype=float)
    W3 = np.array([[0,1],[0,0]], dtype=float)
    
    # m4 = A[1,1] * (B[1,0] - B[0,0])
    # U4[1,1]=1
    # V4[1,0]=1, V4[0,0]=-1
    # W4[1,0]=1
    U4 = np.array([[0,0],[0,1]], dtype=float)
    V4 = np.array([[-1,0],[1,0]], dtype=float)
    W4 = np.array([[0,0],[1,0]], dtype=float)
    
    # m5 = (A[0,0] + A[0,1]) * B[1,1]
    # U5[0,0]=1, U5[0,1]=1
    # V5[1,1]=1
    # W5[0,1]=1
    U5 = np.array([[1,1],[0,0]], dtype=float)
    V5 = np.array([[0,0],[0,1]], dtype=float)
    W5 = np.array([[0,1],[0,0]], dtype=float)
    
    # m6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # U6[1,0]=1, U6[0,0]=-1
    # V6[0,0]=1, V6[0,1]=1
    # W6[1,1]=1
    U6 = np.array([[-1,0],[1,0]], dtype=float)
    V6 = np.array([[1,1],[0,0]], dtype=float)
    W6 = np.array([[0,0],[0,1]], dtype=float)
    
    # m7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # U7[0,1]=1, U7[1,1]=-1
    # V7[1,0]=1, V7[1,1]=1
    # W7[0,0]=1
    U7 = np.array([[0,1],[-1,0]], dtype=float)
    V7 = np.array([[0,0],[1,1]], dtype=float)
    W7 = np.array([[1,0],[0,0]], dtype=float)
    
    return [(U1,V1,W1), (U2,V2,W2), (U3,V3,W3), (U4,V4,W4), (U5,V5,W5), (U6,V6,W6), (U7,V7,W7)]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The multiplication tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = U.shape[1]
    # Build the multiplication tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = 1.0
    
    # Compute from decomposition
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U[i*m + j, r]
                if u_val != 0:
                    for k in range(p):
                        v_val = V[j*p + k, r]
                        if v_val != 0:
                            w_val = W[k*n + i, r]
                            T_decomp[i,j,k] += u_val * v_val * w_val
    
    return np.allclose(T_decomp, T, atol=tol)

def construct():
    """
    Construct a rank-26 decomposition for 3x3x3 matrix multiplication.
    
    Strategy: Use Strassen's algorithm (rank 7) for the top-left 2x2 submatrix,
    and standard multiplication for the remaining entries.
    
    The 3x3 matrices can be partitioned as:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    B = [[B11, B12, B13],
         [B21, B22, B23],
         [B31, B32, B33]]
    
    C = A * B
    
    We use Strassen for the 2x2 block in the top-left (indices 0,1 for both),
    and handle the third row/column with standard multiplication.
    
    Actually, let me think more carefully. The known rank-26 decomposition for 3x3
    uses Strassen for a 2x2 core and adds 19 more terms for the border.
    
    Let me use a different approach: directly construct the rank-26 decomposition
    using the known formula.
    
    For 3x3 matrix multiplication, one known rank-26 decomposition is based on
    extending Strassen's 2x2 algorithm. Let me construct it carefully.
    
    The multiplication tensor for 3x3 has 27 entries (i,j,k) for i,j,k in {0,1,2}.
    We need to decompose it into 26 rank-1 tensors.
    
    Let me use the approach where we treat the 3x3 as a 2x2 block with an extra row/col.
    """
    n, m, p = 3, 3, 3
    rank = 26
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen core for indices (0,1) x (0,1) x (0,1)
    strassen_terms = _strassen_2x2_core()
    
    # Map Strassen 2x2 to 3x3: indices 0,1 stay as 0,1
    for t_idx, (Ut, Vt, Wt) in enumerate(strassen_terms):
        for i in range(2):
            for j in range(2):
                U[i*m + j, t_idx] = Ut[i, j]
        for j in range(2):
            for k in range(2):
                V[j*p + k, t_idx] = Vt[j, k]
        for k in range(2):
            for i in range(2):
                W[k*n + i, t_idx] = Wt[k, i]
    
    # Now handle the border: entries involving index 2
    # We need to cover:
    # C[i,2] for i in {0,1,2}: sum_k A[i,k]*B[k,2]
    # C[2,j] for j in {0,1,2}: sum_k A[2,k]*B[k,j]
    # C[2,2]: sum_k A[2,k]*B[k,2]
    
    # Let's add terms for the third column of B and third row of A.
    
    # For C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]
    # For C[2,j] = A[2,0]*B[0,j] + A[2,1]*B[1,j] + A[2,2]*B[2,j]
    # For C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # Total border terms needed: 
    # C[0,2], C[1,2], C[2,2] from column 2 of B
    # C[2,0], C[2,1] from row 2 of A (C[2,2] already counted)
    # That's 5 terms if we do them naively, but we can be smarter.
    
    # Actually, let me count: the 27 total entries minus the 4 covered by Strassen (C[0,0], C[0,1], C[1,0], C[1,1]) = 23 entries.
    # But Strassen uses 7 terms for those 4 entries. So we have 26-7=19 terms for the remaining 23 entries.
    # That means we need some terms to cover multiple entries.
    
    # Let me just add standard multiplication terms for the border.
    # Border entries: all (i,j,k) where i=2 or j=2 or k=2, except those already in Strassen.
    
    # Strassen covers (i,j,k) for i,j,k in {0,1}. So border is everything else.
    # Number of border entries: 27 - 8 = 19.
    # So we need exactly 19 more terms for 19 border entries! Perfect.
    
    col = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already covered by Strassen
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # col should be 7 + 19 = 26
    assert col == rank, f"Expected col={rank}, got {col}"
    
    # Verify
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
