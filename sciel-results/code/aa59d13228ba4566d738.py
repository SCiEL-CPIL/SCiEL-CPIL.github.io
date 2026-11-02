"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 7 decomposition for 2x2 matrix multiplication
    # This is a known exact decomposition.
    # U shape: (n*m, 7)
    # V shape: (m*p, 7)
    # W shape: (p*n, 7)
    
    # Standard rank 7 decomposition matrices
    # Rows of U correspond to (i, j)
    # Rows of V correspond to (j, k)
    # Rows of W correspond to (k, i) based on the prompt's W[k*n+i] convention
    
    # The matrices below are for the standard tensor T_{i,j,k} where output is C_{i,k}.
    # We need to map the output indices to W[k*n+i].
    
    U_std = [
        [1, 1, 1, 1, 0, 0, 0], # i=0, j=0
        [1, 0, 0, 0, 1, 1, 0], # i=0, j=1
        [0, 1, 0, 0, 1, 0, 1], # i=1, j=0
        [0, 0, 1, 1, 0, 1, 1]  # i=1, j=1
    ]
    
    V_std = [
        [1, 1, 0, 0, 1, 0, 0], # j=0, k=0
        [0, 1, 1, 0, 0, 1, 0], # j=0, k=1
        [0, 0, 1, 1, 0, 0, 1], # j=1, k=0
        [1, 0, 0, 1, 1, 1, 1]  # j=1, k=1
    ]
    
    # W_std rows correspond to output (i, k)
    # Row 0: i=0, k=0
    # Row 1: i=0, k=1
    # Row 2: i=1, k=0
    # Row 3: i=1, k=1
    W_std = [
        [1, 0, 0, 0, 1, 1, 1], # i=0, k=0
        [1, 1, 0, 0, 0, 0, 1], # i=0, k=1
        [0, 1, 1, 0, 0, 1, 0], # i=1, k=0
        [0, 0, 1, 1, 1, 0, 0]  # i=1, k=1
    ]
    
    # The prompt requires W index to be k*n + i.
    # So W[0] corresponds to k=0, i=0 -> C_{0,0}
    # W[1] corresponds to k=0, i=1 -> C_{1,0}
    # W[2] corresponds to k=1, i=0 -> C_{0,1}
    # W[3] corresponds to k=1, i=1 -> C_{1,1}
    
    # Our W_std is indexed by (i, k).
    # We need to rearrange W_std rows to match W[k*n+i].
    # W[0] (k=0, i=0) should be W_std[0] (i=0, k=0)
    # W[1] (k=0, i=1) should be W_std[2] (i=1, k=0)
    # W[2] (k=1, i=0) should be W_std[1] (i=0, k=1)
    # W[3] (k=1, i=1) should be W_std[3] (i=1, k=1)
    
    W_adj = [
        W_std[0], # k=0, i=0
        W_std[2], # k=0, i=1
        W_std[1], # k=1, i=0
        W_std[3]  # k=1, i=1
    ]
    
    # Verification
    def verify(U, V, W):
        # Target: T_{i,j,k} = 1 if A_{i,j} B_{j,k} contributes to C_{k,i} (based on W index)
        # Wait, if W index is k*n+i, the output is C_{k,i}?
        # Standard MM: C_{i,k} = sum A_{i,j} B_{j,k}.
        # If the task is standard MM, the target is C_{i,k}.
        # But the W index is k*n+i.
        # This implies the output matrix is stored such that index k*n+i holds C_{k,i}.
        # This is a transposed output or column-major?
        # If C is stored as C[k][i], then C[k][i] = (AB)^T[k][i] = (BA)[k][i]?
        # No, (AB)[i][k] = sum A[i][j]B[j][k].
        # If we store result in W[k*n+i], we are storing C^T?
        # Let's assume the target tensor is defined by the indices in the prompt.
        # Target_{i,j, k*n+i} = 1 if A_{i,j} B_{j,k} is a valid term.
        # In standard MM, A_{i,j} B_{j,k} is always a valid term for C_{i,k}.
        # So Target_{i,j, k*n+i} = 1 for all i,j,k.
        
        # Let's check if our decomposition produces 1 for all i,j,k at the corresponding W index.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_idx = k * n + i
                    s = 0
                    for r in range(7):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[w_idx][r]
                    if s != 1:
                        return False
        return True

    if verify(U_std, V_std, W_adj):
        return {"U": U_std, "V": V_std, "W": W_adj}
    else:
        # Fallback to rank 8 if rank 7 fails verification
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
