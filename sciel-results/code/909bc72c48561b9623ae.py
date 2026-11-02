import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use the block decomposition into 8 blocks of 2x2x3
    # Each block has rank 10 (known for M_2,2,3)
    # Total rank = 8 * 10 = 80
    
    # We need the rank-10 decomposition of M_2,2,3
    # U_2,2,3 (4x10), V_2,2,3 (6x10), W_2,2,3 (6x10)
    
    # Placeholder for rank-10 decomposition
    # If not available, fall back to schoolbook for sub-blocks (rank 12)
    # Total rank would be 8 * 12 = 96 (same as schoolbook)
    
    # Let's try to find or construct the rank-10.
    # Actually, I will use a known rank-10 decomposition.
    
    # For now, I will implement the 8-block structure with schoolbook sub-blocks
    # to ensure correctness, then try to optimize.
    
    rank = 96
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i_prime in range(2):
        for j_prime in range(2):
            for k_prime in range(2):
                # Block indices
                # i = 2*i_prime + i''
                # j = 2*j_prime + j''
                # k = 3*k_prime + k''
                
                for i_double in range(2):
                    for j_double in range(2):
                        for k_double in range(3):
                            i = 2 * i_prime + i_double
                            j = 2 * j_prime + j_double
                            k = 3 * k_prime + k_double
                            
                            U[i*m + j, col] = 1
                            V[j*p + k, col] = 1
                            W[k*n + i, col] = 1
                            col += 1
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
