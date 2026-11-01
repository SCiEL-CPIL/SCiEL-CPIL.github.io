import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # We use a vectorized construction using Kronecker products
    # to generate the factor matrices.
    
    # U picks A[i,j]. Index i*m + j.
    # V picks B[j,k]. Index j*p + k.
    # W picks C[k,i] (trace convention). Index k*n + i.
    
    # We can build the identity-like tensors and permute.
    # Term t corresponds to (i, j, k) where t = i*m*p + j*p + k? 
    # The seed uses t = i*m*p + j*p + k? 
    # Seed: 
    # for i:
    #   for j:
    #     for k:
    #       term++
    # So term = i*m*p + j*p + k.
    
    # Let's create the indices
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten to get term indices
    terms = i_idx.flatten() * (m * p) + j_idx.flatten() * p + k_idx.flatten()
    
    # U indices: i * m + j
    u_rows = (i_idx.flatten() * m + j_idx.flatten()).astype(int)
    
    # V indices: j * p + k
    v_rows = (j_idx.flatten() * p + k_idx.flatten()).astype(int)
    
    # W indices: k * n + i
    w_rows = (k_idx.flatten() * n + i_idx.flatten()).astype(int)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    U[u_rows, terms] = 1
    V[v_rows, terms] = 1
    W[w_rows, terms] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
