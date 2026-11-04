"""Matrix multiplication 2,4,4 — tensor slicing + Strassen-style rank reduction."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta[i,i']*delta[j,j']*delta[k,k'].
    Actually the tensor is T[i,j,k, i',j',k'] = delta[i,i']*delta[j,j']*delta[k,k']
    and the decomposition is T = sum_c U[:,c] (x) V[:,c] (x) W[:,c]
    where U is indexed by (i,j), V by (j,k), W by (k,i).
    So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    and we need T[i,j,k] = 1 if i==i' and j==j' and k==k' ... wait.
    
    Actually for matrix multiplication C = A*B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T[i,j,k] = 1 (it's the identity tensor for the multiplication).
    The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, let me re-read the convention. W[k*n+i, r] means W is indexed by (k, i) where
    k is the column of B (and row of C) and i is the row of A (and row of C).
    
    So the bilinear form is: C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    
    For this to equal sum_j A[i,j]*B[j,k], we need:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i,i']... 
    
    No wait. The tensor for matrix multiplication is:
    T_{i,j,k} = 1 for all i,j,k (it's just the product structure).
    Actually no. The multiplication tensor is T_{i,j,k} where the output C[i,k] = sum_j A[i,j] B[j,k].
    The tensor is T_{i,j,k} = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k]).
    
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    Wait, that's not right either. Let me think more carefully.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    If we write A[i,j] = sum_c U[i*m+j, c] * x_c and B[j,k] = sum_c V[j*p+k, c] * y_c,
    then C[i,k] = sum_c sum_c' W[k*n+i, c] * (sum_j U[i*m+j,c] x_c) * V[j*p+k,c'] y_c'
    
    Hmm, the standard tensor decomposition for matrix multiplication:
    The tensor T has T[i,j,k] = 1 (for the specific (i,j,k) triple corresponding to the product).
    Actually the multiplication tensor is T_{i,j,k} where i indexes rows of A, j indexes cols of A / rows of B, k indexes cols of B.
    T_{i,j,k} = 1 for the term A_{i,j} * B_{j,k} contributing to C_{i,k}.
    
    So T_{i,j,k} = 1 for all valid (i,j,k), and the decomposition is:
    T_{i,j,k} = sum_c U_{i,j}^{(c)} * V_{j,k}^{(c)} * W_{k,i}^{(c)}
    
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    So we need: for all i in [0,n), j in [0,m), k in [0,p):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen's 7-term decomposition for 2x2 * 2x2.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U[i*2+j, c] corresponds to coefficient of A[i,j] in M_c
    # V[j*2+k, c] corresponds to coefficient of B[j,k] in M_c
    # W[k*2+i, c] corresponds to coefficient of M_c in C[i,k]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00 += M1
    W[3][0] = 1  # C11 += M1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10 += M2
    W[3][1] = -1 # C11 -= M2
    
    # M3 = A00(B01-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1 # B11
    W[1][2] = 1  # C01 += M3
    W[3][2] = 1  # C11 += M3
    
    # M4 = A11(B10-B00)
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # B00
    W[0][3] = 1  # C00 += M4
    W[2][3] = 1  # C10 += M4
    
    # M5 = (A00+A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1 # C00 -= M5
    W[1][4] = 1  # C01 += M5
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[2][5] = 1  # B10
    W[3][5] = 1  # C11 += M6
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[1][6] = 1  # B01
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00 += M7
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Strategy: Use Strassen on each 2x2 block product.
    # A is 2x4 = [A1 | A2] where A1, A2 are 2x2
    # B is 4x4 = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = A*B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # 
    # Each term like A1*B11 is a 2x2 * 2x2 product, which needs rank 7 (Strassen).
    # We have 4 such products: A1*B11, A2*B21, A1*B12, A2*B22
    # Total rank with independent Strassen: 4*7 = 28
    #
    # Can we do better? Let's try to see if we can share some terms.
    # 
    # For now, implement the 4x Strassen decomposition with rank 28.
    
    # Strassen decomposition for 2x2*2x2
    sU, sV, sW = _strassen_2x2x2()
    
    # Build full decomposition with rank 28
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 8 x 28
    V = [[0]*rank for _ in range(m*p)]  # 16 x 28
    W = [[0]*rank for _ in range(p*n)]  # 8 x 28
    
    # Block indices:
    # A[i, j] where i in {0,1}, j in {0,1,2,3}
    #   A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B[j, k] where j in {0,1,2,3}, k in {0,1,2,3}
    #   B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    #   B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C[i, k] where i in {0,1}, k in {0,1,2,3}
    
    # Four Strassen instances:
    # Instance 0: A1 * B11 -> contributes to C[:, 0:2]
    #   A1[i, j] = A[i, j] for i in {0,1}, j in {0,1}
    #   B11[j, k] = B[j, k] for j in {0,1}, k in {0,1}
    #   C[i, k] += result[i, k] for i in {0,1}, k in {0,1}
    
    # Instance 1: A2 * B21 -> contributes to C[:, 0:2]
    #   A2[i, j] = A[i, j+2] for i in {0,1}, j in {0,1}
    #   B21[j, k] = B[j+2, k] for j in {0,1}, k in {0,1}
    #   C[i, k] += result[i, k] for i in {0,1}, k in {0,1}
    
    # Instance 2: A1 * B12 -> contributes to C[:, 2:4]
    #   A1[i, j] = A[i, j] for i in {0,1}, j in {0,1}
    #   B12[j, k] = B[j, k+2] for j in {0,1}, k in {0,1}
    #   C[i, k+2] += result[i, k] for i in {0,1}, k in {0,1}
    
    # Instance 3: A2 * B22 -> contributes to C[:, 2:4]
    #   A2[i, j] = A[i, j+2] for i in {0,1}, j in {0,1}
    #   B22[j, k] = B[j+2, k+2] for j in {0,1}, k in {0,1}
    #   C[i, k+2] += result[i, k] for i in {0,1}, k in {0,1}
    
    for inst in range(4):
        base = inst * 7
        # Determine which A block and B block
        if inst == 0:  # A1 * B11
            a_row_offset = 0  # A[i, j]
            b_col_offset = 0  # B[j, k]
            c_col_offset = 0  # C[i, k]
        elif inst == 1:  # A2 * B21
            a_row_offset = 2  # A[i, j+2]
            b_col_offset = 2  # B[j+2, k]
            c_col_offset = 0  # C[i, k]
        elif inst == 2:  # A1 * B12
            a_row_offset = 0  # A[i, j]
            b_col_offset = 0  # B[j, k+2]
            c_col_offset = 2  # C[i, k+2]
        else:  # inst == 3: A2 * B22
            a_row_offset = 2  # A[i, j+2]
            b_col_offset = 2  # B[j+2, k+2]
            c_col_offset = 2  # C[i, k+2]
        
        for c in range(7):
            # U: for each (i, j_local) in 2x2 block of A
            for i in range(2):
                for j_local in range(2):
                    val = sU[i*2 + j_local][c]
                    if val != 0:
                        U[i * m + (j_local + a_row_offset)][base + c] += val
            
            # V: for each (j_local, k_local) in 2x2 block of B
            for j_local in range(2):
                for k_local in range(2):
                    val = sV[j_local*2 + k_local][c]
                    if val != 0:
                        V[(j_local + b_col_offset) * p + (k_local + b_col_offset)][base + c] += val
            
            # W: for each (k_local, i) in output
            for k_local in range(2):
                for i in range(2):
                    val = sW[k_local*2 + i][c]
                    if val != 0:
                        W[(k_local + c_col_offset) * n + i][base + c] += val
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
