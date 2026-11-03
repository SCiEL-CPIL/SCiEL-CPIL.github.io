import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _verify(U, V, W, n=2, m=4, p=6):
    """Verify that the decomposition correctly computes C = A @ B."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Multiple random tests for robustness
    for seed in [42, 123, 999, 7777, 31415]:
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        A_flat = A.flatten()  # (n*m,) row-major
        B_flat = B.flatten()  # (m*p,) row-major
        
        z = U.T @ A_flat  # (rank,)
        y = V.T @ B_flat  # (rank,)
        
        C_flat = W @ (z * y)  # (p*n,)
        C_computed = C_flat.reshape(p, n).T  # (n, p)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Structural change: Use a direct tensor decomposition approach.
    # 
    # Key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can decompose this using a "row-block" Strassen approach:
    # Split A's 4 columns into two groups of 2: A_L = A[:,0:2], A_R = A[:,2:4]
    # Split B's 4 rows into two groups of 2: B_T = B[0:2,:], B_B = B[2:4,:]
    # 
    # C = A_L @ B_T + A_R @ B_B
    # 
    # Each term (2x2) @ (2x6) can be computed with rank at most 12 naively,
    # but we can use a Strassen-like approach on the 2x2 part.
    #
    # Alternative: Use the 7-term Strassen on the 2x2 block structure where
    # we view A as 2 blocks of (1x2) and B as 2 blocks of (2x3).
    #
    # Let me use a cleaner formulation:
    # A = [a00 a01 | a02 a03]  (2x4)
    #     [a10 a11 | a12 a13]
    # B = [b00 b01 b02 | b03 b04 b05]  (4x6)
    #     [b10 b11 b12 | b13 b14 b15]
    #     [b20 b21 b22 | b23 b24 b25]
    #     [b30 b31 b32 | b33 b34 b35]
    #
    # C[0,k] = a00*b0k + a01*b1k + a02*b2k + a03*b3k
    # C[1,k] = a10*b0k + a11*b1k + a12*b2k + a13*b3k
    #
    # Strassen for 2x2 block multiply (blocks are 1x2 and 2x3):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # Where A11=A[0,0:2], A12=A[0,2:4], A21=A[1,0:2], A22=A[1,2:4]
    #       B11=B[0:2,0:3], B12=B[0:2,3:6], B21=B[2:4,0:3], B22=B[2:4,3:6]
    #
    # Each M is (1x2)@(2x3) = 1x3 vector
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For each M = X @ Y where X is 1x2 and Y is 2x3:
    # M[k] = X[0]*Y[0,k] + X[1]*Y[1,k] for k=0,1,2
    # This requires 6 scalar multiplications per M.
    # Total: 7 * 6 = 42 terms.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define Strassen steps:
    # Each step: (X_coeffs, Y_coeffs, C_targets)
    # X_coeffs: list of (a_row, a_col, coeff) for forming the 1x2 vector X
    #   X[0] = sum of terms where a_col is in {0,2} (first col of each 2-col block)
    #   X[1] = sum of terms where a_col is in {1,3} (second col of each 2-col block)
    # Y_coeffs: list of (b_row, b_col, coeff) for forming the 2x3 matrix Y
    #   Y[0,k] = sum of terms where b_row in {0,1} and b_col%3==k
    #   Y[1,k] = sum of terms where b_row in {2,3} and b_col%3==k
    # C_targets: list of (c_row, c_col_block, sign)
    #   c_col = c_col_block * 3 + k
    
    # A indices: A[i,j] where i in {0,1}, j in {0,1,2,3}
    # A11 = A[0,0:2] -> (0,0),(0,1)
    # A12 = A[0,2:4] -> (0,2),(0,3)
    # A21 = A[1,0:2] -> (1,0),(1,1)
    # A22 = A[1,2:4] -> (1,2),(1,3)
    
    # B indices: B[j,k] where j in {0,1,2,3}, k in {0,...,5}
    # B11 = B[0:2,0:3] -> rows 0,1 cols 0,1,2
    # B12 = B[0:2,3:6] -> rows 0,1 cols 3,4,5
    # B21 = B[2:4,0:3] -> rows 2,3 cols 0,1,2
    # B22 = B[2:4,3:6] -> rows 2,3 cols 3,4,5
    
    steps = []
    
    # M1 = (A11+A22)(B11+B22)
    # X = A11 + A22: X[0] = A[0,0]+A[1,2], X[1] = A[0,1]+A[1,3]
    X1 = [(0,0,1), (1,2,1), (0,1,1), (1,3,1)]  # (a_row, a_col, coeff)
    Y1 = [(0,0,1),(0,1,1),(0,2,1), (1,0,1),(1,1,1),(1,2,1),
          (2,0,1),(2,1,1),(2,2,1), (3,0,1),(3,1,1),(3,2,1)]
    C1_targets = [(0,0,1), (1,1,1)]  # C[0,0:3] += M1, C[1,3:6] += M1
    steps.append((X1, Y1, C1_targets))
    
    # M2 = (A21+A22)B11
    # X = A21+A22: X[0] = A[1,0]+A[1,2], X[1] = A[1,1]+A[1,3]
    X2 = [(1,0,1), (1,2,1), (1,1,1), (1,3,1)]
    Y2 = [(0,0,1),(0,1,1),(0,2,1), (1,0,1),(1,1,1),(1,2,1)]
    C2_targets = [(1,0,1), (1,1,-1)]  # C[1,0:3] += M2, C[1,3:6] -= M2
    steps.append((X2, Y2, C2_targets))
    
    # M3 = A11(B12-B22)
    # X = A11: X[0] = A[0,0], X[1] = A[0,1]
    X3 = [(0,0,1), (0,1,1)]
    Y3 = [(0,3,1),(0,4,1),(0,5,1), (1,3,1),(1,4,1),(1,5,1),
          (2,3,-1),(2,4,-1),(2,5,-1), (3,3,-1),(3,4,-1),(3,5,-1)]
    C3_targets = [(0,1,1), (1,1,1)]  # C[0,3:6] += M3, C[1,3:6] += M3
    steps.append((X3, Y3, C3_targets))
    
    # M4 = A22(B21-B11)
    # X = A22: X[0] = A[1,2], X[1] = A[1,3]
    X4 = [(1,2,1), (1,3,1)]
    Y4 = [(2,0,1),(2,1,1),(2,2,1), (3,0,1),(3,1,1),(3,2,1),
          (0,0,-1),(0,1,-1),(0,2,-1), (1,0,-1),(1,1,-1),(1,2,-1)]
    C4_targets = [(0,0,1), (1,0,1)]  # C[0,0:3] += M4, C[1,0:3] += M4
    steps.append((X4, Y4, C4_targets))
    
    # M5 = (A11+A12)B22
    # X = A11+A12: X[0] = A[0,0]+A[0,2], X[1] = A[0,1]+A[0,3]
    X5 = [(0,0,1), (0,2,1), (0,1,1), (0,3,1)]
    Y5 = [(2,3,1),(2,4,1),(2,5,1), (3,3,1),(3,4,1),(3,5,1)]
    C5_targets = [(0,0,-1), (0,1,1)]  # C[0,0:3] -= M5, C[0,3:6] += M5
    steps.append((X5, Y5, C5_targets))
    
    # M6 = (A21-A11)(B11+B12)
    # X = A21-A11: X[0] = A[1,0]-A[0,0], X[1] = A[1,1]-A[0,1]
    X6 = [(1,0,1), (0,0,-1), (1,1,1), (0,1,-1)]
    Y6 = [(0,0,1),(0,1,1),(0,2,1), (1,0,1),(1,1,1),(1,2,1),
          (0,3,1),(0,4,1),(0,5,1), (1,3,1),(1,4,1),(1,5,1)]
    C6_targets = [(1,1,1)]  # C[1,3:6] += M6
    steps.append((X6, Y6, C6_targets))
    
    # M7 = (A12-A22)(B21+B22)
    # X = A12-A22: X[0] = A[0,2]-A[1,2], X[1] = A[0,3]-A[1,3]
    X7 = [(0,2,1), (1,2,-1), (0,3,1), (1,3,-1)]
    Y7 = [(2,0,1),(2,1,1),(2,2,1), (3,0,1),(3,1,1),(3,2,1),
          (2,3,1),(2,4,1),(2,5,1), (3,3,1),(3,4,1),(3,5,1)]
    C7_targets = [(0,0,1)]  # C[0,0:3] += M7
    steps.append((X7, Y7, C7_targets))
    
    col = 0
    for X_coeffs, Y_coeffs, C_targets in steps:
        # Build X[0] and X[1] coefficient vectors for U
        # X[0] = sum of terms where a_col in {0,2} (first col of 2-col block)
        # X[1] = sum of terms where a_col in {1,3} (second col of 2-col block)
        U_X0 = np.zeros(n * m)
        U_X1 = np.zeros(n * m)
        for a_row, a_col, coeff in X_coeffs:
            if a_col % 2 == 0:  # first col of block
                U_X0[a_row * m + a_col] += coeff
            else:  # second col of block
                U_X1[a_row * m + a_col] += coeff
        
        # Build Y[0,k] and Y[1,k] coefficient vectors for V
        # Y[0,k] = sum of terms where b_row in {0,1} and b_col%3==k
        # Y[1,k] = sum of terms where b_row in {2,3} and b_col%3==k
        Y0 = [np.zeros(m * p) for _ in range(3)]
        Y1 = [np.zeros(m * p) for _ in range(3)]
        for b_row, b_col, coeff in Y_coeffs:
            k = b_col % 3
            if b_row < 2:  # first row group
                Y0[k][b_row * p + b_col] += coeff
            else:  # second row group
                Y1[k][b_row * p + b_col] += coeff
        
        # For each output column k (0,1,2 within the 3-col block):
        for k in range(3):
            # Term: X[0] * Y[0,k]
            U[:, col] = U_X0
            V[:, col] = Y0[k]
            W_col = np.zeros(p * n)
            for c_row, c_col_block, sign in C_targets:
                c_col = c_col_block * 3 + k
                W_col[c_col * n + c_row] += sign
            W[:, col] = W_col
            col += 1
            
            # Term: X[1] * Y[1,k]
            U[:, col] = U_X1
            V[:, col] = Y1[k]
            W_col = np.zeros(p * n)
            for c_row, c_col_block, sign in C_targets:
                c_col = c_col_block * 3 + k
                W_col[c_col * n + c_row] += sign
            W[:, col] = W_col
            col += 1
        
        if time.time() > deadline - 10:
            break
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist()):
        # Fallback to naive rank-48
        rank = 48
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
