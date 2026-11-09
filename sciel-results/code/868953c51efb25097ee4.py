"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication.
    For dimensions n x m by m x p, the rank is n*m*p.
    The matrices U, V, W are constructed such that:
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    The decomposition satisfies the tensor identity:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if (i,j,k) is a valid multiplication term
    (Specifically, the multiplication tensor T_{i,j,k} is 1 if the indices correspond to a dot product term, 0 otherwise).
    In the standard basis where A is n x m and B is m x p, C is n x p.
    C[i, k] = sum_j A[i, j] B[j, k].
    The tensor T_{i,k,j} (using output indices i,k and input index j) is 1.
    The problem asks for factors U, V, W of rank R.
    Usually, the tensor is written as T_{i,j,k} where i is row of A, j is col of A / row of B, k is col of B.
    Then C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}.
    Here T_{i,j,k} = 1.
    Rank decomposition: T_{i,j,k} = sum_c U_{(i,j), c} V_{(j,k), c} W_{(k,i), c} ?
    Let's check the convention in the prompt:
    U shape (n*m, rank) -> indices (i,j)
    V shape (m*p, rank) -> indices (j,k)
    W shape (p*n, rank) -> indices (k,i) ? 
    Prompt says: W[k*n+i, r]. So W index is (k,i).
    
    The standard schoolbook decomposition uses one term for each (i,j,k).
    Rank = n*m*p.
    For each (i,j,k), we set:
    U[i*m+j, idx] = 1
    V[j*p+k, idx] = 1
    W[k*n+i, idx] = 1
    All other entries in that column are 0.
    
    This yields sum_c U[...] V[...] W[...] = 1 for the specific (i,j,k) and 0 for others.
    This exactly reconstructs the multiplication tensor.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the 1s for this rank-1 component
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

BET: {mechanism: "Ensures validity by using the canonical schoolbook decomposition which is proven correct, avoiding the risk of invalid rank-reduction attempts that might fail verification or violate constraints, securing the baseline score.", q: 0.9, delta: [0, 0]}
