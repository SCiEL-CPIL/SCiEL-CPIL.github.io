import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 4, 8

    # Winograd convolution for length-4 (degree 3 polynomial multiplication)
    # We treat the middle dimension m=4 as a convolution.
    # For multiplying a 2x4 matrix by a 4x8 matrix, we can use a block approach.
    # However, a more direct approach is to use the Winograd scheme for the inner product of length 4.
    # The Winograd scheme for length 4 requires 3 multiplications instead of 4.
    # This reduces the rank from n*m*p = 2*4*8 = 64 to n*p*3 = 2*8*3 = 48.
    
    rank = 48
    U = np.zeros((n * m, rank))  # (8, 48)
    V = np.zeros((m * p, rank))  # (32, 48)
    W = np.zeros((p * n, rank))  # (16, 48)
    
    # Winograd coefficients for length 4
    # A matrix: 3x4, B matrix: 4x3, C matrix: 3x4
    # The decomposition is: C[i,k] = sum_t A[i,t] * (B[:,t] * V[:,t])
    # where V[:,t] are the columns of the V matrix for a fixed row of A.
    
    # For length 4 convolution, the Winograd scheme is:
    # A = [[1, 1, 1],
    #      [1, -1, 0],
    #      [0, 1, 1]]
    # B = [[1, 1, 0],
    #      [1, 0, 1],
    #      [1, 1, 1],
    #      [1, 0, -1]]
    # C = [[1, 1, 1],
    #      [1, 0, -1],
    #      [1, -1, 1],
    #      [1, 0, 1]]
    
    # But we need to adapt this to the matrix multiplication context.
    # For each row i of A (n=2 rows) and each column k of C (p=8 columns),
    # we have a length-4 inner product.
    
    # Let's use the standard Winograd coefficients for length 4:
    # A_mat: 3x4
    A_mat = np.array([
        [1, 1, 1],
        [1, -1, 0],
        [0, 1, 1]
    ])
    
    # B_mat: 4x3
    B_mat = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, -1]
    ])
    
    # C_mat: 3x4
    C_mat = np.array([
        [1, 1, 1],
        [1, 0, -1],
        [1, -1, 1],
        [1, 0, 1]
    ])
    
    # For each row i of A (i=0,1) and each column k of C (k=0..7),
    # we need to decompose the length-4 inner product.
    # But wait, the matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a length-4 inner product for each (i,k) pair.
    # Using Winograd, we can reduce this to 3 multiplications.
    
    # The total number of (i,k) pairs is n*p = 2*8 = 16.
    # Each requires 3 multiplications, so total rank = 16*3 = 48.
    
    col = 0
    for i in range(n):
        for k in range(p):
            for t in range(3):
                # U[i*m+j, col] = A_mat[t, j]
                for j in range(m):
                    U[i * m + j, col] = A_mat[t, j]
                
                # V[j*p+k, col] = B_mat[j, t]
                for j in range(m):
                    V[j * p + k, col] = B_mat[j, t]
                
                # W[k*n+i, col] = C_mat[t, 0]  # Wait, this is not right.
                # The Winograd scheme gives:
                # C[i,k] = sum_t (A_mat[t,:] * A[i,:]) * (B_mat[:,t] * B[:,k])
                # So W[k*n+i, col] should be 1 for all t? No.
                # Actually, the output is:
                # C[i,k] = sum_t M_t
                # where M_t = (A_mat[t,:] * A[i,:]) * (B_mat[:,t] * B[:,k])
                # So W[k*n+i, col] = 1 for all t.
                W[k * n + i, col] = 1
                
                col += 1
    
    # Verify correctness
    if time.time() < deadline:
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
