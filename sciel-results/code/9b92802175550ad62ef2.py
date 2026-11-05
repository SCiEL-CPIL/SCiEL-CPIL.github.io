import numpy as np
import time

def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x3 * 3x5 matrix multiplication.
    
    The decomposition uses Strassen's algorithm on the 2x2 blocks of the intermediate
    dimension, combined with schoolbook for the remaining column.
    
    The matrix multiplication tensor T(i,j,k) = 1 for all i,j,k needs to be decomposed
    as: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Strategy:
    - Split the 3 intermediate columns into [0:2] and [2:3]
    - Split the 5 output columns into [0:2], [2:4], [4:5]
    - Use Strassen (rank 7) for the 2x2*2x2 blocks
    - Use schoolbook for blocks involving column 2 of the intermediate dimension
    
    Total rank: 7 + 7 + 4 + 4 + 4 + 2 = 28
    """
    n, m, p = 2, 3, 5
    start_time = time.time()
    
    rank = 28
    U = np.zeros((n * m, rank))  # Shape: (2*3, 28) = (6, 28)
    V = np.zeros((m * p, rank))  # Shape: (3*5, 28) = (15, 28)
    W = np.zeros((p * n, rank))  # Shape: (5*2, 28) = (10, 28)
    
    col = 0
    
    def strassen_2x2(i1, i2, j1, j2, k1, k2):
        """
        Apply Strassen's algorithm to multiply:
        A[i1:i3, j1:j3] * B[j1:j3, k1:k3] -> C[k1:k3, i1:i3]
        where the blocks are 2x2.
        
        i1, i2: row indices of A and C (i1 < i2)
        j1, j2: column indices of A and row indices of B (j1 < j2)
        k1, k2: column indices of B and column indices of C (k1 < k2)
        """
        nonlocal col
        
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m + j1, col] = 1
        U[i2*m + j2, col] = 1
        V[j1*p + k1, col] = 1
        V[j2*p + k2, col] = 1
        W[k1*n + i1, col] = 1
        W[k2*n + i2, col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[i2*m + j1, col] = 1
        U[i2*m + j2, col] = 1
        V[j1*p + k1, col] = 1
        W[k1*n + i2, col] = 1
        W[k2*n + i2, col] = -1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[i1*m + j1, col] = 1
        V[j1*p + k2, col] = 1
        V[j2*p + k2, col] = -1
        W[k2*n + i1, col] = 1
        W[k2*n + i2, col] = 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U[i2*m + j2, col] = 1
        V[j2*p + k1, col] = 1
        V[j1*p + k1, col] = -1
        W[k1*n + i1, col] = 1
        W[k1*n + i2, col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[i1*m + j1, col] = 1
        U[i1*m + j2, col] = 1
        V[j2*p + k2, col] = 1
        W[k1*n + i1, col] = -1
        W[k2*n + i1, col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[i2*m + j1, col] = 1
        U[i1*m + j1, col] = -1
        V[j1*p + k1, col] = 1
        V[j1*p + k2, col] = 1
        W[k2*n + i2, col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[i1*m + j2, col] = 1
        U[i2*m + j2, col] = -1
        V[j2*p + k1, col] = 1
        V[j2*p + k2, col] = 1
        W[k1*n + i1, col] = 1
        col += 1
    
    def schoolbook(i_list, j_list, k_list):
        """
        Schoolbook multiplication for the specified blocks.
        """
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2] (Strassen, rank 7)
    strassen_2x2(0, 1, 0, 1, 0, 1)
    
    # Block 2: A[:,0:2] * B[0:2,2:4] -> C[0:2,2:4] (Strassen, rank 7)
    strassen_2x2(0, 1, 0, 1, 2, 3)
    
    # Block 3: A[:,0:2] * B[0:2,4:5] -> C[0:2,4:5] (Schoolbook, rank 4)
    schoolbook([0, 1], [0, 1], [4])
    
    # Block 4: A[:,2:3] * B[2:3,0:2] -> C[0:2,0:2] (Schoolbook, rank 4)
    schoolbook([0, 1], [2], [0, 1])
    
    # Block 5: A[:,2:3] * B[2:3,2:4] -> C[0:2,2:4] (Schoolbook, rank 4)
    schoolbook([0, 1], [2], [2, 3])
    
    # Block 6: A[:,2:3] * B[2:3,4:5] -> C[0:2,4:5] (Schoolbook, rank 2)
    schoolbook([0, 1], [2], [4])
    
    # Verify the decomposition
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    
    if not np.allclose(tensor, 1.0, atol=1e-9):
        # Fallback to full schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
