import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Use NumPy to construct the matrices efficiently
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # We iterate over the 'rank' dimension which corresponds to the
    # independent computations for each (i, j, k) triple in the trivial decomposition.
    # In the trivial decomposition, each rank column r corresponds to a unique (i, j, k).
    # U[i*m + j, r] = 1 if r corresponds to this (i, j, *)
    # V[j*p + k, r] = 1 if r corresponds to this (*, j, k)
    # W[k*n + i, r] = 1 if r corresponds to this (*, *, k, i) -> wait, W index is k*n + i
    
    # Let's map rank index r to (i, j, k)
    # r = i * (m * p) + j * p + k
    # This is a standard lexicographic mapping.
    
    # Vectorized construction:
    # Generate all (i, j, k) tuples
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate row indices for U, V, W
    # U row: i * m + j
    u_rows = i_flat * m + j_flat
    # V row: j * p + k
    v_rows = j_flat * p + k_flat
    # W row: k * n + i
    w_rows = k_flat * n + i_flat
    
    # The column index in U, V, W is simply the rank index r (0 to rank-1)
    # Since the mapping is 1-to-1 and covers all columns exactly once for each row-set?
    # No, for a specific (i, j), U[i*m+j, r] is 1 for all k.
    # Wait, the trivial decomposition is:
    # C_ik = sum_j A_ij B_jk
    # Tensor term T_{i,j,k}^{a,b,c} = delta_{a,i} delta_{b,j} delta_{c,k}
    # This corresponds to rank-1 terms.
    # Summing over j? No, the tensor is T_{a,b,c} = sum_j delta_{a,i} ... 
    # Actually, the tensor for matrix multiplication M_{n,m,p} is:
    # T = sum_{i,j,k} e_i^U \otimes e_j^V \otimes e_k^W ?? No.
    # The standard decomposition is:
    # C = A B
    # C_{ik} = \sum_{j=1}^m A_{ij} B_{jk}
    # This can be written as a sum of rank-1 tensors:
    # T = \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^p E_{i,j}^U \otimes E_{j,k}^V \otimes E_{k,i}^W
    # Where E_{i,j}^U is the basis vector for index (i*m+j) in U space?
    # Let's check the interface shapes:
    # U: (n*m, rank). Index i*m+j.
    # V: (m*p, rank). Index j*p+k.
    # W: (p*n, rank). Index k*n+i.
    
    # For each (i, j, k) tuple, we have a rank-1 component:
    # U[:, r] has a 1 at index i*m+j
    # V[:, r] has a 1 at index j*p+k
    # W[:, r] has a 1 at index k*n+i
    # And 0 elsewhere.
    
    # So for each r in 0..rank-1, we map r to (i, j, k).
    # r = i * (m*p) + j * p + k
    
    # We can fill U, V, W by scattering 1s.
    # U[u_rows, r] = 1
    # V[v_rows, r] = 1
    # W[w_rows, r] = 1
    
    # Using numpy advanced indexing:
    r_indices = np.arange(rank)
    
    # This sets U[u_rows[t], r_indices[t]] = 1 for all t
    # This is a sparse assignment.
    # We can use np.add.at or simply direct assignment if no duplicates in (row, col) pairs.
    # In this decomposition, each (row, col) pair is unique for a given matrix?
    # U: Row is i*m+j, Col is r (which includes k).
    # For a fixed (i,j), there are p different r values (different k).
    # So U[i*m+j, r] is 1 for r corresponding to that i,j and any k.
    # Are there duplicate assignments?
    # No, each r is unique. So we just set those positions to 1.
    
    U[u_rows, r_indices] = 1
    V[v_rows, r_indices] = 1
    W[w_rows, r_indices] = 1
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
