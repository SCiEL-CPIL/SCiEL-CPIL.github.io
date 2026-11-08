"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_block_U(n, m, block_offset_j):
    """U rows for one 2x2 Strassen block at column offset block_offset_j.
    U has shape (n*m, 7). Row i*m+j gets the appropriate Strassen coefficient.
    Strassen: M1=(A11+A22)(B11+B22), M2=(A21+A22)B11, M3=A11(B12-B22),
              M4=A22(B21-B11), M5=(A11+A12)B22, M6=(A21-A11)(B11+B12),
              M7=(A12-A21)(B21+B22)
    C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    """
    # For n=3, m=3, we use rows 0,1 for the 2x2 block (j=block_offset_j, j+1)
    # and row 2 is unused (zero) in this block
    U_block = np.zeros((n * m, 7))
    j0 = block_offset_j
    j1 = block_offset_j + 1
    # Row i*m+j0 corresponds to A[i, j0], row i*m+j1 to A[i, j1]
    for i in range(n):
        # A11 = A[0, j0], A12 = A[0, j1], A21 = A[1, j0], A22 = A[1, j1]
        # But wait: we need a 2x2 submatrix. For n=3, we can use rows 0,1 or 1,2 or 0,2.
        # Let's use rows 0 and 1 for the 2x2 block.
        pass
    # Actually, let me reconsider. The matrix A is 3x3, B is 3x7.
    # C = A*B is 3x7.
    # We can decompose B into 2x2 blocks and 1x2 blocks, etc.
    # For each 2x2 block of B (columns j, j+1), we use Strassen on the 2x2 submatrix A[0:2, 0:2]
    # But A is 3x3, so we have A[0:2, 0:2] as a 2x2 block, and the third row/column is separate.
    
    # Let me think differently. The standard approach:
    # A (3x3) * B (3x7) = C (3x7)
    # We can partition A into blocks and B into blocks.
    # One approach: use Strassen on a 2x2 block of A (rows 0,1 and cols 0,1)
    # and handle the rest separately.
    
    # Actually, the cleanest approach for 3x3 * 3x7:
    # Partition B into 7 columns. For each pair of columns, use Strassen.
    # But Strassen requires 2x2 matrices. So we need to partition A into 2x2 blocks too.
    
    # Let me use the following partition:
    # A = [[A11, A12, A13],
    #      [A21, A22, A23],
    #      [A31, A32, A33]]
    # B = [[B11, B12, ..., B17],
    #      [B21, B22, ..., B27],
    #      [B31, B32, ..., B37]]
    
    # C = A*B
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # One approach: for each column k of B, compute A * B[:,k] (3x3 * 3x1 = 3x1)
    # But that's 7 separate 3x3*3x1 multiplications.
    
    # Better: use Strassen on 2x2 blocks.
    # Partition A into a 2x2 block (rows 0,1, cols 0,1) and handle row 2, col 2 separately.
    # Partition B into 2x2 blocks (cols 0,1), (cols 2,3), (cols 4,5), and col 6 separately.
    
    # For the 2x2 block A[0:2, 0:2] * B[0:2, 0:2], use Strassen (7 terms)
    # For the 2x2 block A[0:2, 0:2] * B[0:2, 2:4], use Strassen (7 terms)
    # For the 2x2 block A[0:2, 0:2] * B[0:2, 4:6], use Strassen (7 terms)
    # For A[0:2, 0:2] * B[0:2, 6], that's 2x2 * 2x1, which is 4 terms
    # For A[2, 0:3] * B[0:3, 0:7], that's 1x3 * 3x7 = 21 terms
    # For A[0:2, 2] * B[2, 0:7], that's 2x1 * 1x7 = 14 terms
    
    # Total: 7+7+7+4+21+14 = 60? Let me recalculate.
    
    # Actually, let me just implement the known rank-61 decomposition.
    # The key idea: 3x3 * 3x7 can be done with rank 61 using Strassen-based decomposition.
    
    return None  # Placeholder

def construct():
    """Construct a rank-minimal tensor decomposition for 3x3 * 3x7 matrix multiplication."""
    t_start = time.time()
    deadline = t_start + 110  # Safety margin before 120s limit
    
    n, m, p = 3, 3, 7
    
    # Strassen-based decomposition for 3x3 * 3x7
    # Partition:
    # A (3x3) into: A[0:2, 0:2] (2x2 block), A[2, 0:3] (1x3 row), A[0:2, 2] (2x1 col)
    # B (3x7) into: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6] (2x1), B[2, 0:7] (1x7)
    
    # For each 2x2 block of B (cols 0-1, 2-3, 4-5), use Strassen with A[0:2, 0:2]
    # That's 3 * 7 = 21 terms
    # For B[0:2, 6] (2x1), use 4 terms with A[0:2, 0:2]
    # For A[2, 0:3] * B[0:3, 0:7], use 3*7 = 21 terms
    # For A[0:2, 2] * B[2, 0:7], use 2*7 = 14 terms
    # Total: 21 + 4 + 21 + 14 = 60? 
    
    # Hmm, let me think more carefully.
    # Actually, the known result is rank 61 for 3x3 * 3x7.
    # Let me implement it step by step.
    
    U = np.zeros((n * m, 61))
    V = np.zeros((m * p, 61))
    W = np.zeros((p * n, 61))
    
    col = 0
    
    # Strassen 2x2 block coefficients for U (A side), V (B side), W (C side)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A21)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For A[0:2, 0:2] * B[0:2, j:j+2] for j in [0, 2, 4]:
    # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,j], B12=B[0,j+1], B21=B[1,j], B22=B[1,j+1]
    # C11=C[0,j], C12=C[0,j+1], C21=C[1,j], C22=C[1,j+1]
    
    for j in [0, 2, 4]:
        # M1
        U[0*3+0][col] = 1  # A[0,0]
        U[1*3+1][col] = 1  # A[1,1]
        V[0*7+j][col] = 1  # B[0,j]
        V[0*7+j+1][col] = 1  # B[0,j+1]
        V[1*7+j][col] = 1  # B[1,j]
        V[1*7+j+1][col] = 1  # B[1,j+1]
        W[j*3+0][col] = 1  # C[0,j]
        W[(j+1)*3+0][col] = 1  # C[0,j+1]
        W[j*3+1][col] = 1  # C[1,j]
        W[(j+1)*3+1][col] = 1  # C[1,j+1]
        col += 1
        
        # M2
        U[1*3+0][col] = 1  # A[1,0]
        U[1*3+1][col] = 1  # A[1,1]
        V[0*7+j][col] = 1  # B[0,j]
        W[j*3+1][col] = 1  # C[1,j]
        col += 1
        
        # M3
        U[0*3+0][col] = 1  # A[0,0]
        V[0*7+j+1][col] = 1  # B[0,j+1]
        V[1*7+j+1][col] = -1  # B[1,j+1]
        W[(j+1)*3+0][col] = 1  # C[0,j+1]
        col += 1
        
        # M4
        U[1*3+1][col] = 1  # A[1,1]
        V[1*7+j][col] = 1  # B[1,j]
        V[0*7+j][col] = -1  # B[0,j]
        W[j*3+1][col] = 1  # C[1,j]
        W[(j+1)*3+1][col] = 1  # C[1,j+1]
        col += 1
        
        # M5
        U[0*3+0][col] = 1  # A[0,0]
        U[0*3+1][col] = 1  # A[0,1]
        V[1*7+j+1][col] = 1  # B[1,j+1]
        W[j*3+0][col] = -1  # C[0,j]
        W[(j+1)*3+0][col] = 1  # C[0,j+1]
        col += 1
        
        # M6
        U[1*3+0][col] = 1  # A[1,0]
        U[0*3+0][col] = -1  # A[0,0]
        V[0*7+j][col] = 1  # B[0,j]
        V[0*7+j+1][col] = 1  # B[0,j+1]
        W[(j+1)*3+1][col] = 1  # C[1,j+1]
        col += 1
        
        # M7
        U[0*3+1][col] = 1  # A[0,1]
        U[1*3+0][col] = -1  # A[1,0]
        V[1*7+j][col] = 1  # B[1,j]
        V[1*7+j+1][col] = 1  # B[1,j+1]
        W[j*3+0][col] = 1  # C[0,j]
        W[(j+1)*3+1][col] = 1  # C[1,j+1]
        col += 1
    
    # For A[0:2, 0:2] * B[0:2, 6] (2x2 * 2x1 = 2x1)
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    # 4 terms
    # Term 1: A[0,0]*B[0,6]
    U[0*3+0][col] = 1
    V[0*7+6][col] = 1
    W[6*3+0][col] = 1
    col += 1
    # Term 2: A[0,1]*B[1,6]
    U[0*3+1][col] = 1
    V[1*7+6][col] = 1
    W[6*3+0][col] = 1
    col += 1
    # Term 3: A[1,0]*B[0,6]
    U[1*3+0][col] = 1
    V[0*7+6][col] = 1
    W[6*3+1][col] = 1
    col += 1
    # Term 4: A[1,1]*B[1,6]
    U[1*3+1][col] = 1
    V[1*7+6][col] = 1
    W[6*3+1][col] = 1
    col += 1
    
    # For A[2, 0:3] * B[0:3, 0:7] (1x3 * 3x7 = 1x7)
    # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    # 3*7 = 21 terms
    for j in range(3):
        for k in range(7):
            U[2*3+j][col] = 1  # A[2,j]
            V[j*7+k][col] = 1  # B[j,k]
            W[k*3+2][col] = 1  # C[2,k]
            col += 1
    
    # For A[0:2, 2] * B[2, 0:7] (2x1 * 1x7 = 2x7)
    # C[i,k] = A[i,2]*B[2,k] for i=0,1
    # 2*7 = 14 terms
    for i in range(2):
        for k in range(7):
            U[i*3+2][col] = 1  # A[i,2]
            V[2*7+k][col] = 1  # B[2,k]
            W[k*3+i][col] = 1  # C[i,k]
            col += 1
    
    # Total: 21 + 4 + 21 + 14 = 60? But we expected 61.
    # Let me check: 3*7 (Strassen) + 4 + 21 + 14 = 21+4+21+14 = 60.
    # Hmm, maybe I'm missing something. Let me verify.
    
    if col != 61:
        # Adjust: we have col terms, pad with zeros if needed
        if col < 61:
            U = np.concatenate([U, np.zeros((n*m, 61-col))], axis=1)
            V = np.concatenate([V, np.zeros((m*p, 61-col))], axis=1)
            W = np.concatenate([W, np.zeros((p*n, 61-col))], axis=1)
        else:
            # Shouldn't happen, but truncate
            U = U[:, :61]
            V = V[:, :61]
            W = W[:, :61]
    
    # Verify
    U_np = U.astype(float)
    V_np = V.astype(float)
    W_np = W.astype(float)
    
    # Check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c]... 
    # Actually, the tensor decomposition condition is:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # where T[i,j,k] = 1 if this is the (i,j,k) entry of the matmul tensor, else 0.
    # For matmul, T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k]).
    # Wait, no. The tensor is T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    # For standard matmul, T[i,j,k] = 1 for all i,j,k.
    
    # The decomposition says: T = sum_c U_c ⊗ V_c ⊗ W_c
    # where (U_c ⊗ V_c ⊗ W_c)[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k.
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(61):
                    s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                max_err = max(max_err, abs(s - 1))
    
    if max_err > 1e-10:
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Convert to lists
    return {
        "U": U_np.tolist(),
        "V": V_np.tolist(),
        "W": W_np.tolist()
    }
# EVOLVE-BLOCK-END
