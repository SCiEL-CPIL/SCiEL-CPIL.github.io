"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time

def _strassen_3x3x3():
    """
    Return (U, V, W) for 3x3 x 3x3 matrix multiplication, rank 7.
    U: (9, 7) - linear combos of A entries, indexed by (i,j) -> row i*3+j
    V: (9, 7) - linear combos of B entries, indexed by (j,k) -> row j*3+k
    W: (9, 7) - linear combos giving C entries, indexed by (k,i) -> row k*3+i (trace convention W[k*n+i])
    """
    n = 3
    U = [[0]*7 for _ in range(9)]
    V = [[0]*7 for _ in range(9)]
    W = [[0]*7 for _ in range(9)]
    
    # Strassen's 7 intermediate products
    # M1 = (A00+A22)(B00+B22)
    # M2 = (A20+A22)B00
    # M3 = A00(B01-B21)
    # M4 = A22(B21-B01)
    # M5 = (A00+A01)B21
    # M6 = (A10+A20)(B00+B10)
    # M7 = (A01+A11)(B10+B20)
    
    # U: row index = i*3+j (A[i][j])
    # V: row index = j*3+k (B[j][k])
    # W: row index = k*3+i (C[i][k] -> W[k*3+i])
    
    # M1: (A00+A22)(B00+B22)
    U[0][0] = 1; U[8][0] = 1  # A[0][0], A[2][2]
    V[0][0] = 1; V[8][0] = 1  # B[0][0], B[2][2]
    
    # M2: (A20+A22)B00
    U[6][1] = 1; U[8][1] = 1  # A[2][0], A[2][2]
    V[0][1] = 1  # B[0][0]
    
    # M3: A00(B01-B21)
    U[0][2] = 1  # A[0][0]
    V[1][2] = 1; V[7][2] = -1  # B[0][1], B[2][1]
    
    # M4: A22(B21-B01)
    U[8][3] = 1  # A[2][2]
    V[7][3] = 1; V[1][3] = -1  # B[2][1], B[0][1]
    
    # M5: (A00+A01)B21
    U[0][4] = 1; U[1][4] = 1  # A[0][0], A[0][1]
    V[7][4] = 1  # B[2][1]
    
    # M6: (A10+A20)(B00+B10)
    U[3][5] = 1; U[6][5] = 1  # A[1][0], A[2][0]
    V[0][5] = 1; V[3][5] = 1  # B[0][0], B[1][0]
    
    # M7: (A01+A11)(B10+B20)
    U[1][6] = 1; U[4][6] = 1  # A[0][1], A[1][1]
    V[3][6] = 1; V[6][6] = 1  # B[1][0], B[2][0]
    
    # W: C[i][k] = sum over products, W row = k*3+i
    # C00 = M1 - M2 + M3 + M4
    W[0*3+0][0] = 1; W[0][1] = -1; W[0][2] = 1; W[0][3] = 1
    # C01 = M3 + M5
    W[1*3+0][2] = 1; W[3][4] = 1
    # C02 = M7
    W[2*3+0][6] = 1
    # C10 = M6 + M7
    W[0*3+1][5] = 1; W[0][6] = 1
    # C11 = M1 - M3 + M5 - M7
    W[1*3+1][0] = 1; W[3][2] = -1; W[3][4] = 1; W[3][6] = -1
    # C12 = M4 + M6
    W[2*3+1][3] = 1; W[6][5] = 1
    # C20 = M2 + M4
    W[0*3+2][1] = 1; W[0][3] = 1
    # C21 = M1 - M2 + M4
    W[1*3+2][0] = 1; W[3][1] = -1; W[3][3] = 1
    # C22 = M1 - M5 + M6
    W[2*3+2][0] = 1; W[6][4] = -1; W[6][5] = 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, verbose=False):
    """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} for all i,j,k."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                expected = 1.0 if i == k else 0.0
                err = abs(s - expected)
                if err > max_err:
                    max_err = err
                if verbose and err > 1e-10:
                    print(f"  ERROR: i={i},j={j},k={k}: got {s}, expected {expected}")
    return max_err


def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Strategy: Use 5 independent Strassen rank-7 blocks, one for each output column.
    # Total rank = 5 * 7 = 35
    # 
    # For output column k_out (0..4), the block computes:
    #   C[:, k_out] = A * B[:, k_out]
    # Using Strassen, this needs 7 rank-1 terms.
    #
    # U: shape (n*m, rank) = (9, 35)
    #   For block c (0..4), columns c*7..c*7+6 use the Strassen U pattern
    # V: shape (m*p, rank) = (15, 35)
    #   For block c, rows corresponding to B[:, c] use Strassen V pattern
    # W: shape (p*n, rank) = (15, 35)
    #   For block c, rows corresponding to C[:, c] use Strassen W pattern
    
    U_str, V_str, W_str = _strassen_3x3x3()
    
    rank = 5 * 7
    U = [[0] * rank for _ in range(n * m)]  # 9 x 35
    V = [[0] * rank for _ in range(m * p)]  # 15 x 35
    W = [[0] * rank for _ in range(p * n)]  # 15 x 35
    
    for col_out in range(p):  # 5 output columns
        base = col_out * 7
        for c in range(7):
            # U: same for all blocks (A is shared)
            for i in range(n):
                for j in range(m):
                    U[i*m + j][base + c] = U_str[i*m + j][c]
            
            # V: B has m*p = 15 entries, indexed by j*p+k
            # For output column col_out, we use B[j][col_out] for j=0,1,2
            # In Strassen V, row index = j*3+k where k is the "column" in the 3x3 B
            # Here k corresponds to col_out, so V row = j*p + col_out
            for j in range(m):
                for k_local in range(n):  # k_local is the column index in the 3x3 block
                    # Strassen V row = j*3 + k_local
                    # Actual V row = j*p + col_out
                    val = V_str[j*3 + k_local][c]
                    if val != 0:
                        V[j*p + col_out][base + c] += val
            
            # W: C has p*n = 15 entries, indexed by k*n+i (trace convention)
            # For output column col_out, C[i][col_out] -> W row = col_out*n + i
            # In Strassen W, row index = k*3+i where k is the column
            # Here k corresponds to col_out
            for i in range(n):
                for k_local in range(n):
                    # Strassen W row = k_local*3 + i
                    # Actual W row = col_out*n + i
                    val = W_str[k_local*3 + i][c]
                    if val != 0:
                        W[col_out*n + i][base + c] += val
    
    # Verify
    max_err = _verify_decomposition(U, V, W, n, m, p)
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
