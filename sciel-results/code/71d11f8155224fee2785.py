"""Rewritten schoolbook decomposition for 3x3x7 matrix multiplication."""

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R = n * m * p
    
    # Initialize factor matrices with zeros
    # U has shape (n*m, R)
    # V has shape (m*p, R)
    # W has shape (p*n, R)
    
    # We construct the tensors based on the identity:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] has a 1 if we are computing the product of A[i,j] and B[j,k].
    # In the decomposition T = sum_r U[i,r] V[j,r] W[k,r],
    # For schoolbook, each term corresponds to a specific (i,j,k) triple.
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    # Pre-allocate lists of lists
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    # Iterate through all combinations of i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map indices to flattened row indices for U and V
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                
                # Set the coefficient for this rank-1 term
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
