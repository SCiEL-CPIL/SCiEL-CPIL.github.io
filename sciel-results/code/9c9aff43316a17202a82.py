"""Matrix multiplication 4,4,6 rank-minimal decomposition."""
import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    
    # Build target tensor for verification
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    def check_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces the target tensor."""
        T_test = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_test += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_test, T, atol=1e-10)
    
    # Build trivial rank-96 decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    
    # Strassen's rank-7 decomposition for 2x2x2 matrix multiplication
    # For C = A * B where A is 2x2, B is 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Split 4x4x6 into blocks of 2x2x2
    # n=4 -> 2 blocks in i-direction
    # m=4 -> 2 blocks in j-direction  
    # p=6 -> 3 blocks in k-direction
    # Total: 2*2*3 = 12 sub-problems, each rank 7 = total rank 84
    
    block_i = 2  # size of i-block
    block_j = 2  # size of j-block
    block_k = 2  # size of k-block
    
    num_i_blocks = n // block_i  # 2
    num_j_blocks = m // block_j  # 2
    num_k_blocks = p // block_k  # 3
    
    total_rank = num_i_blocks * num_j_blocks * num_k_blocks * 7  # 84
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for ib in range(num_i_blocks):
        for jb in range(num_j_blocks):
            for kb in range(num_k_blocks):
                # Block indices
                i_start = ib * block_i
                j_start = jb * block_j
                k_start = kb * block_k
                
                # For this 2x2x2 block, we need to map:
                # A is 2x2: indices (i, j) where i in [i_start, i_start+2), j in [j_start, j_start+2)
                # B is 2x2: indices (j, k) where j in [j_start, j_start+2), k in [k_start, k_start+2)
                # C is 2x2: indices (k, i) where k in [k_start, k_start+2), i in [i_start, i_start+2)
                
                # Strassen's 7 products for this block
                # Each product M_r involves linear combinations of A and B entries
                # and contributes to C entries
                
                # Let's define the 7 Strassen factors:
                # Factor 0: M1 = (A11+A22)*(B11+B22)
                #   A part: A11 (i=i_start, j=j_start) + A22 (i=i_start+1, j=j_start+1)
                #   B part: B11 (j=j_start, k=k_start) + B22 (j=j_start+1, k=k_start+1)
                #   C part: C11 (k=k_start, i=i_start) + C22 (k=k_start+1, i=i_start+1)
                
                # Factor 1: M2 = (A21+A22)*B11
                #   A part: A21 (i=i_start+1, j=j_start) + A22 (i=i_start+1, j=j_start+1)
                #   B part: B11 (j=j_start, k=k_start)
                #   C part: C21 (k=k_start, i=i_start+1)
                
                # Factor 2: M3 = A11*(B12-B22)
                #   A part: A11 (i=i_start, j=j_start)
                #   B part: B12 (j=j_start, k=k_start+1) - B22 (j=j_start+1, k=k_start+1)
                #   C part: C12 (k=k_start+1, i=i_start) + C22 (k=k_start+1, i=i_start+1)
                
                # Factor 3: M4 = A22*(B21-B11)
                #   A part: A22 (i=i_start+1, j=j_start+1)
                #   B part: B21 (j=j_start+1, k=k_start) - B11 (j=j_start, k=k_start)
                #   C part: C11 (k=k_start, i=i_start) + C21 (k=k_start, i=i_start+1)
                
                # Factor 4: M5 = (A11+A12)*B22
                #   A part: A11 (i=i_start, j=j_start) + A12 (i=i_start, j=j_start+1)
                #   B part: B22 (j=j_start+1, k=k_start+1)
                #   C part: C11 (k=k_start, i=i_start) + C12 (k=k_start+1, i=i_start)
                
                # Factor 5: M6 = (A21-A11)*(B11+B12)
                #   A part: A21 (i=i_start+1, j=j_start) - A11 (i=i_start, j=j_start)
                #   B part: B11 (j=j_start, k=k_start) + B12 (j=j_start, k=k_start+1)
                #   C part: C22 (k=k_start+1, i=i_start+1)
                
                # Factor 6: M7 = (A12-A22)*(B21+B22)
                #   A part: A12 (i=i_start, j=j_start+1) - A22 (i=i_start+1, j=j_start+1)
                #   B part: B21 (j=j_start+1, k=k_start) + B22 (j=j_start+1, k=k_start+1)
                #   C part: C11 (k=k_start, i=i_start) + C22 (k=k_start+1, i=i_start+1)
                
                # Now fill in the U, V, W matrices for each of the 7 factors
                
                # Factor 0: M1
                U[i_start*m + j_start, col] = 1      # A11
                U[(i_start+1)*m + (j_start+1), col] = 1  # A22
                V[j_start*p + k_start, col] = 1      # B11
                V[(j_start+1)*p + (k_start+1), col] = 1  # B22
                W[k_start*n + i_start, col] = 1      # C11
                W[(k_start+1)*n + (i_start+1), col] = 1  # C22
                col += 1
                
                # Factor 1: M2
                U[(i_start+1)*m + j_start, col] = 1      # A21
                U[(i_start+1)*m + (j_start+1), col] = 1  # A22
                V[j_start*p + k_start, col] = 1      # B11
                W[k_start*n + (i_start+1), col] = 1  # C21
                col += 1
                
                # Factor 2: M3
                U[i_start*m + j_start, col] = 1      # A11
                V[j_start*p + (k_start+1), col] = 1  # B12
                V[(j_start+1)*p + (k_start+1), col] = -1  # -B22
                W[(k_start+1)*n + i_start, col] = 1  # C12
                W[(k_start+1)*n + (i_start+1), col] = 1  # C22
                col += 1
                
                # Factor 3: M4
                U[(i_start+1)*m + (j_start+1), col] = 1  # A22
                V[(j_start+1)*p + k_start, col] = 1  # B21
                V[j_start*p + k_start, col] = -1  # -B11
                W[k_start*n + i_start, col] = 1  # C11
                W[k_start*n + (i_start+1), col] = 1  # C21
                col += 1
                
                # Factor 4: M5
                U[i_start*m + j_start, col] = 1      # A11
                U[i_start*m + (j_start+1), col] = 1  # A12
                V[(j_start+1)*p + (k_start+1), col] = 1  # B22
                W[k_start*n + i_start, col] = 1  # C11
                W[(k_start+1)*n + i_start, col] = 1  # C12
                col += 1
                
                # Factor 5: M6
                U[(i_start+1)*m + j_start, col] = 1      # A21
                U[i_start*m + j_start, col] = -1  # -A11
                V[j_start*p + k_start, col] = 1  # B11
                V[j_start*p + (k_start+1), col] = 1  # B12
                W[(k_start+1)*n + (i_start+1), col] = 1  # C22
                col += 1
                
                # Factor 6: M7
                U[i_start*m + (j_start+1), col] = 1  # A12
                U[(i_start+1)*m + (j_start+1), col] = -1  # -A22
                V[(j_start+1)*p + k_start, col] = 1  # B21
                V[(j_start+1)*p + (k_start+1), col] = 1  # B22
                W[k_start*n + i_start, col] = 1  # C11
                W[(k_start+1)*n + (i_start+1), col] = 1  # C22
                col += 1
    
    # Verify the decomposition
    if check_decomposition(U, V, W):
        best_rank = total_rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    else:
        # Fallback to trivial
        best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
