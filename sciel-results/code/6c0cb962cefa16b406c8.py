"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook (canonical) tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix. The rank is 3*3*7 = 63.
    
    While not rank-minimal (the optimal rank for M_{3,3,7} is likely lower,
    perhaps around 33-40), this is the only decomposition we can guarantee
    to construct correctly and verify within the time limit without relying
    on unverified complex algebraic identities.
    
    The interface requires:
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    Indexing convention for W: W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Populate the decomposition
    # The tensor for matrix multiplication is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For the schoolbook decomposition, each term in the sum corresponds to a specific (i,j,k) tuple.
    # T[i,j,k] should be 1 if we are computing element (i,k) of the product? 
    # Actually, the tensor M_{n,m,p} has indices (i, j, k) where:
    #   i in 0..n-1 (row of A)
    #   j in 0..m-1 (col of A / row of B)
    #   k in 0..p-1 (col of B)
    # The value is 1 if and only if we are looking at the component that contributes to C[i,k].
    # Specifically, (AB)[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T[i,j,k] is 1 for all valid i,j,k, and the decomposition sums up to the identity mapping?
    
    # Let's check the convention again.
    # "W[k*n+i,r]"
    # This suggests the third index of the tensor is mapped to a single index k*n+i.
    # So the tensor T has dimensions (n*m, m*p, p*n).
    # T[i*m+j, j*p+k, k*n+i] = 1 for the schoolbook case?
    # No, the standard decomposition is:
    # M_{n,m,p} = sum_{r=1}^{nmp} U^{(r)} ⊗ V^{(r)} ⊗ W^{(r)}
    # Where U^{(r)} is a vector in R^{n*m}, V^{(r)} in R^{m*p}, W^{(r)} in R^{p*n}.
    # For the schoolbook term corresponding to (i,j,k):
    #   U has a 1 at index i*m+j
    #   V has a 1 at index j*p+k
    #   W has a 1 at index k*n+i
    
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
                
    # Convert to lists to match the interface requirement if strict types are checked,
    # though NumPy arrays are usually accepted. The prompt says "factor matrices", 
    # and the seed used lists of lists. Let's return lists of lists to be safe and match seed style.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
