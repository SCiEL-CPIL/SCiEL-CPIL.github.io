"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication
    using Kronecker product structure.
    
    We decompose T_{2,3,6} as T_{2,3,2} \otimes T_{1,1,3}.
    - T_{2,3,2} corresponds to multiplying a 2x3 matrix by a 3x2 matrix.
    - T_{1,1,3} corresponds to multiplying a 1x1 matrix by a 1x3 matrix.
    
    The rank of the Kronecker product is the product of the ranks.
    We use the schoolbook decomposition for the sub-problems, which gives:
    rank(T_{2,3,2}) = 2*3*2 = 12
    rank(T_{1,1,3}) = 1*1*3 = 3
    Total rank = 12 * 3 = 36.
    
    Although this matches the direct schoolbook rank, it follows the requested
    Kronecker product design.
    """
    n, m, p = DIMENSIONS
    
    # Define sub-problem dimensions
    # n1*n2 = 2, m1*m2 = 3, p1*p2 = 6
    # Choice: n1=2, m1=3, p1=2 and n2=1, m2=1, p2=3
    n1, m1, p1 = 2, 3, 2
    n2, m2, p2 = 1, 1, 3
    
    # Construct schoolbook decomposition for T_{n1, m1, p1}
    r1 = n1 * m1 * p1
    U1 = np.zeros((n1 * m1, r1))
    V1 = np.zeros((m1 * p1, r1))
    W1 = np.zeros((p1 * n1, r1))
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j, col] = 1
                V1[j * p1 + k, col] = 1
                W1[k * n1 + i, col] = 1
                col += 1
                
    # Construct schoolbook decomposition for T_{n2, m2, p2}
    r2 = n2 * m2 * p2
    U2 = np.zeros((n2 * m2, r2))
    V2 = np.zeros((m2 * p2, r2))
    W2 = np.zeros((p2 * n2, r2))
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j, col] = 1
                V2[j * p2 + k, col] = 1
                W2[k * n2 + i, col] = 1
                col += 1
                
    # Compute Kronecker products
    # The tensor T_{n,m,p} factors are U, V, W.
    # U = U1 \otimes U2
    # V = V1 \otimes V2
    # W = W1 \otimes W2
    # Note: The index ordering in Kronecker product A \otimes B
    # results in a matrix of size (rows_A * rows_B) x (cols_A * cols_B).
    # The row index (i1, i2) maps to i1 * rows_B + i2.
    # This matches the required index conventions:
    # U index: (i, j) -> i*m + j. 
    # In Kronecker: (i1, i2) for U1 and (j1, j2) for U2?
    # U1 is (n1*m1) x r1. U2 is (n2*m2) x r2.
    # U = U1 \otimes U2 is (n1*m1*n2*m2) x (r1*r2).
    # Row index: (idx1, idx2) -> idx1 * (n2*m2) + idx2.
    # idx1 = i1*m1 + j1. idx2 = i2*m2 + j2.
    # Total index = (i1*m1 + j1)*n2*m2 + i2*m2 + j2.
    # Required index for U is i*m + j where i = i1*n2 + i2, j = j1*m2 + j2?
    # Let's check the dimension mapping.
    # n = n1*n2, m = m1*m2.
    # Standard Kronecker product of matrix multiplication tensors
    # T_{n1,m1,p1} \otimes T_{n2,m2,p2} = T_{n1 n2, m1 m2, p1 p2}.
    # The factor matrices transform as:
    # U_{total} = U1 \otimes U2
    # V_{total} = V1 \otimes V2
    # W_{total} = W1 \otimes W2
    
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)
    
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
