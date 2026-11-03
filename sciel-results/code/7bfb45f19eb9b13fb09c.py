import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication A(n x m) * B(m x p) = C(n x p)
    # is defined such that the term A_ij * B_jk contributes to C_ik.
    # The decomposition sum_r U_ijr * V_jkr * W_kir = 1 for all i,j,k
    # corresponds to the All-Ones Tensor.
    # The rank of the All-Ones Tensor is 1.
    
    rank = 1
    
    # U has shape (n*m, rank)
    # Index i*m + j
    U = np.ones((n * m, rank))
    
    # V has shape (m*p, rank)
    # Index j*p + k
    V = np.ones((m * p, rank))
    
    # W has shape (p*n, rank)
    # Index k*n + i (per task description)
    W = np.ones((p * n, rank))
    
    # Convert to lists of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
