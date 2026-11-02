"""Rewritten program: greedy rank-reduction for 2x5 x 5x6 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T has indices (i, j, k) with i in [0,n), j in [0,m), k in [0,p)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k, and 0 otherwise.
    # This is a sparse tensor with exactly n*m*p non-zero entries, each equal to 1.
    
    # The schoolbook algorithm uses rank = n*m*p = 60.
    # We want to find a rank-minimal exact tensor decomposition.
    
    # Strategy: Use a greedy rank-reduction algorithm that iteratively identifies and removes
    # dependent terms via Gaussian elimination on the support of the tensor.
    
    # First, let's set up the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Now, we want to find the minimum number of rank-1 tensors that sum to T.
    # We'll use a greedy approach: start with all n*m*p rank-1 terms and try to eliminate
    # dependent ones.
    
    # The rank-1 terms are:
    # T_{i,j,k} = e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    # Let's represent each rank-1 term as a vector in the space R^{(n*m)*(m*p)*(p*n)}
    # by flattening the 3D tensor.
    
    # However, this space is too large (10 * 30 * 12 = 3600 dimensions). Instead,
    # we can use the fact that the tensor is sparse and work with its support.
    
    # Alternative approach: Use the fact that for small tensors, we can try to find
    # linear dependencies among the rank-1 terms by solving a linear system.
    
    # Let's try a different approach: use known decompositions or construct one manually.
    
    # For 2xM x MxP, there are known results. Let me try to construct a decomposition
    # with rank less than 60.
    
    # One idea: for each j, the sub-tensor involves U-indices {j, M+j}, V-indices {j*P, ..., j*P+P-1},
    # and W-indices that depend on i and k.
    
    # Let me try to use a Strassen-like approach for the 2x2 blocks within each j.
    
    # Actually, let's try a simpler approach: use the fact that we can sometimes combine
    # terms that share the same V-index.
    
    # For fixed (j,k), we have two terms:
    #   (e_j, e_{j*P+k}, e_{2k}) and (e_{M+j}, e_{j*P+k}, e_{2k+1})
    
    # These have different U and W indices, so they can't be directly combined.
    
    # Let me try yet another approach: use a greedy algorithm that tries to find
    # a subset of the rank-1 terms that spans the tensor.
    
    # I'll use a simple heuristic: start with all terms and try to remove each one
    # if the remaining terms can still span the tensor.
    
    # To check if the remaining terms can span the tensor, I'll flatten the tensor
    # and the rank-1 terms into vectors and check if the tensor is in the span.
    
    # This is computationally expensive, but for 60 terms in 3600 dimensions, it might work.
    
    # Let's flatten the tensor
    T_flat = T.flatten()
    
    # Generate all rank-1 terms as flattened vectors
    rank1_terms = []
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                term = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                term[i*m+j, j*p+k, k*n+i] = 1.0
                rank1_terms.append(term.flatten())
                indices.append((i, j, k))
    
    rank1_terms = np.array(rank1_terms).T  # Shape: (3600, 60)
    
    # Now, we want to find the minimum subset of columns that spans T_flat.
    # This is a set cover problem, which is NP-hard, but for 60 elements we can try
    # a greedy approach.
    
    # Greedy approach: start with all 60 terms and try to remove each one if the
    # remaining terms still span T_flat.
    
    # To check if T_flat is in the span of a set of vectors, we can check if
    # the augmented matrix [V | T_flat] has the same rank as V.
    
    # Let's implement this:
    
    def can_span(vecs, target):
        """Check if target is in the span of vecs (list of 1D arrays)."""
        if len(vecs) == 0:
            return np.allclose(target, 0)
        V = np.array(vecs).T  # Shape: (dim, num_vecs)
        augmented = np.column_stack([V, target])
        return np.linalg.matrix_rank(V) == np.linalg.matrix_rank(augmented)
    
    # Start with all 60 terms
    active = list(range(len(rank1_terms)))
    
    # Try to remove each term if possible
    changed = True
    while changed:
        changed = False
        for idx in list(active):
            # Try removing this term
            remaining = [i for i in active if i != idx]
            if can_span([rank1_terms[:, i] for i in remaining], T_flat):
                active.remove(idx)
                changed = True
    
    # Now, active contains the indices of the terms we need
    rank = len(active)
    
    # Construct the U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, idx in enumerate(active):
        i, j, k = indices[idx]
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
