import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    rank = 30
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # We'll use the naive decomposition for the 3x3x3 core (rank 27)
    # and add 3 more terms for the 7th column
    # Total rank: 27 + 3 = 30
    
    # For the core 3x3x3 part (columns 0-5 of the 7-column output):
    # For each (i, j, k) where k in 0..5, we add a term
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(6):  # Only first 6 columns
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # For the 7th column (k=6), we need to handle it separately
    # The 7th column of the output is: for each i, output[i, 6] = sum_j A[i,j] * B[j,6]
    # We can decompose this using 3 terms (one for each row of A)
    for i in range(n):
        # Term: A[i, :] dot B[:, 6]
        # U[i*m + j, col] should be 1 for all j? No, that would give sum_j A[i,j] * 1
        # We need: sum_j A[i,j] * B[j,6]
        # So U[i*m+j, col] = 1 for all j, V[j*p+6, col] = 1 for all j, W[6*n+i, col] = 1
        for j in range(m):
            U[i*m + j, col] = 1
        for j in range(m):
            V[j*p + 6, col] = 1
        W[6*n + i, col] = 1
        col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
