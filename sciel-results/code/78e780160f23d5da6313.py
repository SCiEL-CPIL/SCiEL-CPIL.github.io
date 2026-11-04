import numpy as np

def find_minimal_rank(U, V, W, max_rank=77):
    """
    Attempts to reduce the rank of the decomposition by finding linear dependencies
    among the rank-1 tensors.
    """
    n_terms = len(U[0])
    n_um = len(U)
    n_vm = len(V)
    n_wm = len(W)
    
    # Represent each rank-1 tensor as a vector in R^(n_um * n_vm * n_wm)
    # T_k = U_k (n_um) x V_k (n_vm) x W_k (n_wm)
    # Flatten T_k to a vector of length n_um * n_vm * n_wm
    
    vecs = []
    for k in range(n_terms):
        # U[:, k] is a vector of length n_um
        # V[:, k] is a vector of length n_vm
        # W[:, k] is a vector of length n_wm
        
        # Outer product
        # T_k[i, j, l] = U[i, k] * V[j, k] * W[l, k]
        
        # We can just store the vectors and check if the tensor is linearly dependent
        # A set of tensors is linearly dependent if there exist c_k not all zero such that
        # sum c_k T_k = 0
        
        # This is equivalent to checking if the vectors formed by the "slices" are dependent?
        # No, tensor rank is different from matrix rank.
        
        # However, if we fix one mode, say W, we get a matrix M_k = U_k V_k^T (n_um x n_vm).
        # The sum is sum c_k M_k = 0.
        # This is a linear dependency in the space of matrices.
        # If the matrices M_k are linearly dependent, the tensors might be.
        # But tensor rank is harder.
        
        pass
