import numpy as np

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor of dimensions 3x4x6.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = 3, 4, 6
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Schoolbook decomposition
    # The tensor is T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki} (with appropriate output indexing)
    # U corresponds to A indices (i, j) -> index i*m + j
    # V corresponds to B indices (j, k) -> index j*p + k
    # W corresponds to C indices (k, i) -> index k*n + i (as per trace convention W[k*n+i])
    # Note: The standard bilinear form is C_{ik} = sum_j A_{ij} B_{jk}.
    # In the decomposition C_{ik} = sum_r U_{(i,j),r} A_{ij} V_{(j,k),r} B_{jk} W_{(k,i),r} ?
    # No, the standard decomposition is:
    # C_{ik} = sum_r (sum_{i',j'} U_{i'j', r} A_{i'j'}) (sum_{j',k'} V_{j'k', r} B_{j'k'}) W_{ki, r}
    # For this to equal sum_j A_{ij} B_{jk}, we need the "rank-1" terms to isolate specific i,j,k.
    # The schoolbook terms are:
    # For each (i,j,k):
    #   U has 1 at (i*m+j), 0 elsewhere.
    #   V has 1 at (j*p+k), 0 elsewhere.
    #   W has 1 at (k*n+i), 0 elsewhere.
    #   Then term r contributes: 1 * A_{ij} * 1 * B_{jk} * 1 * (output at k*n+i which is C_{ik})
    #   Summing over all i,j,k gives sum_{i,j,k} A_{ij} B_{jk} delta_{output, (k,i)}
    #   For a fixed output (k,i), only the term with that specific k and i contributes.
    #   The sum over j for that term is just A_{ij} B_{jk}.
    #   So yes, this works.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
