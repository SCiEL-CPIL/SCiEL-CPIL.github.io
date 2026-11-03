import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _verify(U, V, W, n=2, m=4, p=6):
    """Verify that the decomposition correctly computes C = A @ B."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # C[k*n+i] = sum_r W[k*n+i, r] * (sum_{i',j} U[i'*m+j, r] * A[i',j]) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
    # For verification, we need: for each r, the contribution to C[i,k] is
    # W[k*n+i, r] * (U[:,r] . A_flat) * (V[:,r] . B_flat)
    # This should equal A[i,k] * B[k,j] summed over j... 
    
    # More precisely, for the decomposition to work:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j} U[i'*m+j, r] * A[i',j]) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
    # should equal sum_j A[i,j] * B[j,k]
    
    # Let's check with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute via decomposition
    A_flat = A.flatten()  # (n*m,)
    B_flat = B.flatten()  # (m*p,)
    
    # z_r = U[:, r] . A_flat
    z = U.T @ A_flat  # (rank,)
    y = V.T @ B_flat  # (rank,)
    
    # C_flat[i] = sum_r W[i, r] * z_r * y_r
    C_flat = W @ (z * y)  # (p*n,)
    C_computed = C_flat.reshape(p, n).T  # (n, p)
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strategy: Treat 2x4x6 as 2x2 block multiplication where:
    # A = [[A11, A12], [A21, A22]] with each block being 1x2
    # B = [[B11, B12], [B21, B22]] with each block being 2x3
    # C = [[C11, C12], [C21, C22]] with each block being 1x3
    #
    # Using Strassen's 7-step algorithm for 2x2 block matrices:
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11       -> C21 += M2, C22 -= M2
    # M3 = A11(B12-B22)       -> C12 += M3, C22 += M3
    # M4 = A22(B21-B11)       -> C11 += M4, C21 += M4
    # M5 = (A11+A12)B22       -> C11 -= M5, C12 += M5
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    #
    # Each Mi is a product of a (1x2) vector and a (2x3) matrix = (1x3) result.
    # Computing (1x2)*(2x3) requires 2*3 = 6 scalar multiplications naively.
    # Total rank = 7 * 6 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Block definitions:
    # A11 = A[0, 0:2], A12 = A[0, 2:4]
    # A21 = A[1, 0:2], A22 = A[1, 2:4]
    # B11 = B[0:2, 0:3], B12 = B[0:2, 3:6]
    # B21 = B[2:4, 0:3], B22 = B[2:4, 3:6]
    # C11 = C[0, 0:3], C12 = C[0, 3:6]
    # C21 = C[1, 0:3], C22 = C[1, 3:6]
    
    # For each Strassen step, we have:
    # X = linear_combination of A entries -> 1x2 vector
    # Y = linear_combination of B entries -> 2x3 matrix
    # Product Z = X * Y -> 1x3 vector
    # Each element Z[k] = sum_j X[j] * Y[j,k] requires one scalar mult per (j,k)
    # So 2*3 = 6 terms per step.
    
    # Define the 7 Strassen steps with their A-coefficients and B-coefficients
    # A-coeffs: dict mapping (i, j) -> coeff for forming X
    # B-coeffs: dict mapping (j, k) -> coeff for forming Y
    # C_targets: list of (block_row, block_col, sign) where block_row in {0,1}, block_col in {0,1}
    
    steps = [
        # M1 = (A11+A22)(B11+B22)
        (
            {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},  # X = [A00+A12, A01+A13]
            {(0,0): 1, (0,1): 1, (0,2): 1,             # Y = B11 + B22
             (1,0): 1, (1,1): 1, (1,2): 1,
             (2,0): 1, (2,1): 1, (2,2): 1,
             (3,0): 1, (3,1): 1, (3,2): 1},
            [(0, 0, 1), (1, 1, 1)]  # C11 += M1, C22 += M1
        ),
        # M2 = (A21+A22)B11
        (
            {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},  # X = [A10+A12, A11+A13]
            {(0,0): 1, (0,1): 1, (0,2): 1,             # Y = B11
             (1,0): 1, (1,1): 1, (1,2): 1},
            [(1, 0, 1), (1, 1, -1)]  # C21 += M2, C22 -= M2
        ),
        # M3 = A11(B12-B22)
        (
            {(0,0): 1, (0,1): 1},                       # X = [A00, A01]
            {(0,3): 1, (0,4): 1, (0,5): 1,             # Y = B12 - B22
             (1,3): 1, (1,4): 1, (1,5): 1,
             (2,3): -1, (2,4): -1, (2,5): -1,
             (3,3): -1, (3,4): -1, (3,5): -1},
            [(0, 1, 1), (1, 1, 1)]  # C12 += M3, C22 += M3
        ),
        # M4 = A22(B21-B11)
        (
            {(1,2): 1, (1,3): 1},                       # X = [A12, A13]
            {(2,0): 1, (2,1): 1, (2,2): 1,             # Y = B21 - B11
             (3,0): 1, (3,1): 1, (3,2): 1,
             (0,0): -1, (0,1): -1, (0,2): -1,
             (1,0): -1, (1,1): -1, (1,2): -1},
            [(0, 0, 1), (1, 0, 1)]  # C11 += M4, C21 += M4
        ),
        # M5 = (A11+A12)B22
        (
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},  # X = [A00+A02, A01+A03]
            {(2,3): 1, (2,4): 1, (2,5): 1,             # Y = B22
             (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 0, -1), (0, 1, 1)]  # C11 -= M5, C12 += M5
        ),
        # M6 = (A21-A11)(B11+B12)
        (
            {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1}, # X = [A10-A00, A11-A01]
            {(0,0): 1, (0,1): 1, (0,2): 1,             # Y = B11 + B12
             (1,0): 1, (1,1): 1, (1,2): 1,
             (0,3): 1, (0,4): 1, (0,5): 1,
             (1,3): 1, (1,4): 1, (1,5): 1},
            [(1, 1, 1)]  # C22 += M6
        ),
        # M7 = (A12-A22)(B21+B22)
        (
            {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1}, # X = [A02-A12, A03-A13]
            {(2,0): 1, (2,1): 1, (2,2): 1,             # Y = B21 + B22
             (3,0): 1, (3,1): 1, (3,2): 1,
             (2,3): 1, (2,4): 1, (2,5): 1,
             (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 0, 1)]  # C11 += M7
        ),
    ]
    
    col = 0
    for A_coeffs, B_coeffs, C_targets in steps:
        # X is a 1x2 vector: X[j] = sum_{(i,k) in A_coeffs} coeff * A[i,k]
        # Y is a 2x3 matrix: Y[j,k] = sum_{(j',k') in B_coeffs} coeff * B[j',k']
        # Z = X * Y is 1x3: Z[k] = X[0]*Y[0,k] + X[1]*Y[1,k]
        
        # For each k in 0,1,2: Z[k] = X[0]*Y[0,k] + X[1]*Y[1,k]
        # We need 2 terms per k (one for X[0]*Y[0,k] and one for X[1]*Y[1,k])
        # Total 6 terms per step
        
        for k in range(3):
            # Term 1: X[0] * Y[0,k]
            U_col = np.zeros(n * m)
            V_col = np.zeros(m * p)
            W_col = np.zeros(p * n)
            
            # X[0] = sum of A_coeffs where the "column index within block" is 0
            # A11 cols: 0,1. A12 cols: 2,3. A21 cols: 0,1. A22 cols: 2,3.
            # X[0] uses the first column of each block: A[i,0] for i=0,1 and A[i,2] for i=1
            # Actually, X[0] = sum of coeffs where j % 2 == 0 (first col of 2-col block)
            # X[1] = sum of coeffs where j % 2 == 1 (second col of 2-col block)
            
            for (i, j), c in A_coeffs.items():
                if j % 2 == 0:  # First column of block -> X[0]
                    U_col[i * m + j] += c
            
            # Y[0,k] = sum of B_coeffs where row is in first row of block (j%2==0) and col within block == k
            for (j, k2), c in B_coeffs.items():
                if j % 2 == 0 and k2 % 3 == k:  # First row of block, k-th col within block
                    V_col[j * p + k2] += c
            
            # W: Z[0] contributes to C targets at column k
            for block_row, block_col, sign in C_targets:
                i_C = block_row
                k_C = block_col * 3 + k
                W_col[k_C * n + i_C] += sign
            
            U[:, col] = U_col
            V[:, col] = V_col
            W[:, col] = W_col
            col += 1
            
            # Term 2: X[1] * Y[1,k]
            U_col = np.zeros(n * m)
            V_col = np.zeros(m * p)
            W_col = np.zeros(p * n)
            
            for (i, j), c in A_coeffs.items():
                if j % 2 == 1:  # Second column of block -> X[1]
                    U_col[i * m + j] += c
            
            for (j, k2), c in B_coeffs.items():
                if j % 2 == 1 and k2 % 3 == k:  # Second row of block, k-th col within block
                    V_col[j * p + k2] += c
            
            for block_row, block_col, sign in C_targets:
                i_C = block_row
                k_C = block_col * 3 + k
                W_col[k_C * n + i_C] += sign
            
            U[:, col] = U_col
            V[:, col] = V_col
            W[:, col] = W_col
            col += 1
        
        if time.time() > deadline - 20:
            break
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist()):
        # Fallback to naive rank-48 if verification fails
        rank = 48
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
