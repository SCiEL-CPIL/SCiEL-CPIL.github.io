"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    # Laderman rank-7 decomposition of 2x2x2, applied to 5 column-pair blocks
    # giving total rank 35 for 2x2x10 multiplication.
    #
    # Laderman's 7 terms for 2x2 matmul (C = A*B):
    # T1 = (a11+a22)(b11+b22)
    # T2 = (a11-a22)(b11-b22)
    # T3 = (a11+a12)(b22-b21)
    # T4 = (a22+a21)(b11-b12)
    # T5 = a11*b22
    # T6 = a22*b11
    # T7 = (a12+a22)(b11+b21)
    #
    # C11 = T1 + T2 + T3 - T4 + T5
    # C12 = T3 + T5 - T7
    # C21 = T4 - T5 + T7
    # C22 = -T1 + T2 + T4 + T6
    #
    # For 2x2x10: A is 2x2, B is 2x10, C is 2x10.
    # We split B into 5 blocks of 2 columns each: B[:, 2t:2t+2].
    # For each block t, we apply Laderman's 7 terms.
    # Total rank = 5 * 7 = 35.
    #
    # U has shape (n*m, rank) = (4, 35)
    # V has shape (m*p, rank) = (20, 35)
    # W has shape (p*n, rank) = (20, 35)
    #
    # Indexing:
    # U row: i*m + j  (i in 0..n-1, j in 0..m-1) -> i*2 + j
    # V row: j*p + k  (j in 0..m-1, k in 0..p-1) -> j*10 + k
    # W row: k*n + i  (k in 0..p-1, i in 0..n-1) -> k*2 + i
    #
    # Laderman terms for block t (columns 2t, 2t+1 of B):
    # Let b11 = B[0, 2t], b12 = B[1, 2t], b21 = B[0, 2t+1], b22 = B[1, 2t+1]
    # Wait, I need to be careful with the indexing.
    #
    # Actually, let me re-derive. B is 2x10. B[j, k] where j is row (0,1), k is column (0..9).
    # For block t (t=0..4), the two columns are k=2t and k=2t+1.
    # So b11 = B[0, 2t], b12 = B[0, 2t+1] -- wait no.
    #
    # Let me use standard matrix notation. B is m x p = 2 x 10.
    # B[j, k] where j=0,1 and k=0..9.
    # For block t, we consider B[:, 2t:2t+2] which is a 2x2 matrix:
    #   [B[0,2t],   B[0,2t+1]]
    #   [B[1,2t],   B[1,2t+1]]
    #
    # In the Laderman notation for C = A * B_block:
    # A = [[a11, a12], [a21, a22]]
    # B_block = [[b11, b12], [b21, b22]]
    # where b11 = B[0,2t], b12 = B[0,2t+1], b21 = B[1,2t], b22 = B[1,2t+1]
    #
    # The 7 terms:
    # T1 = (a11+a22)*(b11+b22) -> U: a11+a22, V: b11+b22
    # T2 = (a11-a22)*(b11-b22) -> U: a11-a22, V: b11-b22
    # T3 = (a11+a12)*(b22-b21) -> U: a11+a12, V: b22-b21
    # T4 = (a22+a21)*(b11-b12) -> U: a22+a21, V: b11-b12
    # T5 = a11*b22 -> U: a11, V: b22
    # T6 = a22*b11 -> U: a22, V: b11
    # T7 = (a12+a22)*(b11+b21) -> U: a12+a22, V: b11+b21
    #
    # C_block = C[:, 2t:2t+2] = [[C[0,2t], C[0,2t+1]], [C[1,2t], C[1,2t+1]]]
    # C11 = C[0,2t] = T1 + T2 + T3 - T4 + T5
    # C12 = C[0,2t+1] = T3 + T5 - T7
    # C21 = C[1,2t] = T4 - T5 + T7
    # C22 = C[1,2t+1] = -T1 + T2 + T4 + T6
    #
    # Now, U has rows indexed by (i,j) where i is row of A (0,1), j is col of A (0,1).
    # A[i,j]: i=0,j=0 -> a11; i=0,j=1 -> a12; i=1,j=0 -> a21; i=1,j=1 -> a22
    # U row index = i*m + j = i*2 + j
    #
    # V has rows indexed by (j,k) where j is row of B (0,1), k is col of B (0..9).
    # V row index = j*p + k = j*10 + k
    #
    # W has rows indexed by (k,i) where k is col of C (0..9), i is row of C (0,1).
    # W row index = k*n + i = k*2 + i
    #
    # For each block t and term r (0..6), we have a column c = t*7 + r.
    
    n, m, p = 2, 2, 10
    rank = 35
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Laderman terms: (U_coefficients, V_coefficients, W_coefficients)
    # Each term is a list of (row_index, coefficient) for U, V, W respectively.
    
    # U rows: 0=(0,0)=a11, 1=(0,1)=a12, 2=(1,0)=a21, 3=(1,1)=a22
    # For each block t, V rows: b11=(0,2t), b12=(0,2t+1), b21=(1,2t), b22=(1,2t+1)
    # V row indices: b11=0*10+2t=2t, b12=0*10+2t+1=2t+1, b21=1*10+2t=10+2t, b22=1*10+2t+1=11+2t
    # W rows: C11=(2t,0), C12=(2t+1,0), C21=(2t,1), C22=(2t+1,1)
    # W row indices: C11=2t*2+0=4t, C12=(2t+1)*2+0=4t+2, C21=2t*2+1=4t+1, C22=(2t+1)*2+1=4t+3
    
    # Term definitions: (U: list of (row, coeff)), (V: list of (row, coeff)), (W: list of (row, coeff))
    # T1 = (a11+a22)*(b11+b22)
    # T2 = (a11-a22)*(b11-b22)
    # T3 = (a11+a12)*(b22-b21)
    # T4 = (a22+a21)*(b11-b12)
    # T5 = a11*b22
    # T6 = a22*b11
    # T7 = (a12+a22)*(b11+b21)
    
    # W contributions:
    # C11 = T1 + T2 + T3 - T4 + T5
    # C12 = T3 + T5 - T7
    # C21 = T4 - T5 + T7
    # C22 = -T1 + T2 + T4 + T6
    
    for t in range(5):
        col_base = t * 7
        b11_idx = 2 * t
        b12_idx = 2 * t + 1
        b21_idx = 10 + 2 * t
        b22_idx = 10 + 2 * t + 1
        c11_idx = 4 * t
        c12_idx = 4 * t + 2
        c21_idx = 4 * t + 1
        c22_idx = 4 * t + 3
        
        # T1: U={a11:1, a22:1}, V={b11:1, b22:1}, W={C11:1, C22:-1}
        col = col_base + 0
        U[0][col] = 1   # a11
        U[3][col] = 1   # a22
        V[b11_idx][col] = 1
        V[b22_idx][col] = 1
        W[c11_idx][col] = 1
        W[c22_idx][col] = -1
        
        # T2: U={a11:1, a22:-1}, V={b11:1, b22:-1}, W={C11:1, C22:1}
        col = col_base + 1
        U[0][col] = 1
        U[3][col] = -1
        V[b11_idx][col] = 1
        V[b22_idx][col] = -1
        W[c11_idx][col] = 1
        W[c22_idx][col] = 1
        
        # T3: U={a11:1, a12:1}, V={b22:1, b21:-1}, W={C11:1, C12:1}
        col = col_base + 2
        U[0][col] = 1
        U[1][col] = 1
        V[b22_idx][col] = 1
        V[b21_idx][col] = -1
        W[c11_idx][col] = 1
        W[c12_idx][col] = 1
        
        # T4: U={a22:1, a21:1}, V={b11:1, b12:-1}, W={C11:-1, C21:1, C22:1}
        col = col_base + 3
        U[3][col] = 1
        U[2][col] = 1
        V[b11_idx][col] = 1
        V[b12_idx][col] = -1
        W[c11_idx][col] = -1
        W[c21_idx][col] = 1
        W[c22_idx][col] = 1
        
        # T5: U={a11:1}, V={b22:1}, W={C11:1, C12:1, C21:-1}
        col = col_base + 4
        U[0][col] = 1
        V[b22_idx][col] = 1
        W[c11_idx][col] = 1
        W[c12_idx][col] = 1
        W[c21_idx][col] = -1
        
        # T6: U={a22:1}, V={b11:1}, W={C22:1}
        col = col_base + 5
        U[3][col] = 1
        V[b11_idx][col] = 1
        W[c22_idx][col] = 1
        
        # T7: U={a12:1, a22:1}, V={b11:1, b21:1}, W={C12:-1, C21:1}
        col = col_base + 6
        U[1][col] = 1
        U[3][col] = 1
        V[b11_idx][col] = 1
        V[b21_idx][col] = 1
        W[c12_idx][col] = -1
        W[c21_idx][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
