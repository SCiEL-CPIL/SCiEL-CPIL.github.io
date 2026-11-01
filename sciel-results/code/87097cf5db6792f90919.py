import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    # U2, V2, W2 as lists of 2x2 matrices
    U2 = np.array([
        [[1, 0], [0, 1]], # 0
        [[0, 0], [1, 1]], # 1
        [[1, 0], [0, 0]], # 2
        [[0, 0], [0, 1]], # 3
        [[1, 1], [0, 0]], # 4
        [[-1, 0], [1, 0]],# 5
        [[0, 1], [0, -1]] # 6
    ])
    
    V2 = np.array([
        [[1, 0], [0, 1]], # 0
        [[1, 0], [0, 0]], # 1
        [[0, 1], [0, -1]],# 2
        [[-1, 0], [1, 0]],# 3
        [[0, 0], [0, 1]], # 4
        [[1, 1], [0, 0]], # 5
        [[0, 0], [1, 1]]  # 6
    ])
    
    W2 = np.array([
        [1, 0, 0, 1], # 0
        [0, 0, 1, -1],# 1
        [0, 1, 0, 1], # 2
        [1, 0, 1, 0], # 3
        [-1, 1, 0, 0],# 4
        [0, 0, 0, 1], # 5
        [1, 0, 0, 0]  # 6
    ])
    
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Tensor product construction
    for k1 in range(7):
        for k2 in range(7):
            r = k1 * 7 + k2
            
            # U factor: A is 4x4, split into 2x2 blocks
            # A[i,j] where i=2*i1+i2, j=2*j1+j2
            # U[r] corresponds to term (k1, k2)
            # The coefficient for A[i,j] in term r is U2[k1][i2][j2] * U2[k2][i1][j1]? 
            # Wait, the tensor product structure:
            # M_4 = sum_{r1, r2} (U2[r1] \otimes U2[r2]) (V2[r1] \otimes V2[r2]) (W2[r1] \otimes W2[r2])
            # The input A is a 4x4 matrix. We view it as a 2x2 matrix of 2x2 blocks.
            # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2.
            # The "Strassen" multiplication on the block level uses U2, V2, W2.
            # The "inner" multiplication on the 2x2 blocks uses U2, V2, W2 again.
            
            # So, for a given pair of indices (k1, k2) in the 7x7 grid:
            # The term is the product of the k1-th factor of the outer Strassen and the k2-th factor of the inner Strassen.
            
            # U matrix entry for A[i,j]:
            # i = 2*i1 + i2, j = 2*j1 + j2
            # The coefficient is U2[k1][i1][j1] * U2[k2][i2][j2]
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            coeff = U2[k1, i1, j1] * U2[k2, i2, j2]
                            U[i*m + j, r] += coeff
                            
            # V matrix entry for B[j,k]:
            # j = 2*j1 + j2, k = 2*k1_ + k2_ (using different names to avoid clash)
            # Let B indices be (row, col) = (j, k).
            # j = 2*j1 + j2, k = 2*k1_idx + k2_idx
            # The coefficient is V2[k1][j1][k1_idx] * V2[k2][j2][k2_idx]
            
            for j1 in range(2):
                for k1_idx in range(2):
                    for j2 in range(2):
                        for k2_idx in range(2):
                            j = 2*j1 + j2
                            k = 2*k1_idx + k2_idx
                            coeff = V2[k1, j1, k1_idx] * V2[k2, j2, k2_idx]
                            V[j*p + k, r] += coeff
                            
            # W matrix entry for C[i,k]:
            # i = 2*i1 + i2, k = 2*k1_idx + k2_idx
            # The coefficient is W2[k1][i1][k1_idx] * W2[k2][i2][k2_idx]
            # Note: W2 is stored as flat [row, col] for the 2x2 output block.
            # W2[m][u*2+v] corresponds to output element (u,v).
            
            for i1 in range(2):
                for k1_idx in range(2):
                    for i2 in range(2):
                        for k2_idx in range(2):
                            i = 2*i1 + i2
                            k = 2*k1_idx + k2_idx
                            coeff = W2[k1, i1*2 + k1_idx] * W2[k2, i2*2 + k2_idx]
                            W[k*n + i, r] += coeff
                            
    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
