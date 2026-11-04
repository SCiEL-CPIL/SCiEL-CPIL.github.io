import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Model matrix multiplication as a trilinear tensor T_{i,j,k} = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where i in [0,n), j in [0,m), k in [0,p)
    # The tensor has dimensions (n*m) x (m*p) x (p*n)
    # Rank is the minimum number of rank-1 tensors that sum to T

    # Start with schoolbook decomposition (rank = n*m*p = 48)
    # Each term is: U_c = e_{i,j}, V_c = e_{j,k}, W_c = e_{k,i}
    rank = n * m * p

    # Build factor matrices as numpy arrays
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    # Greedy iterative compression algorithm:
    # We try to reduce the rank by checking if any term can be expressed as a
    # linear combination of the others. This is done by flattening each term
    # into a vector in R^{(n*m)*(m*p)*(p*n)} and checking linear independence.
    #
    # For efficiency, we use a different approach: we check if the tensor
    # formed by removing one term is still in the span of the remaining terms.
    #
    # A more practical approach: use the fact that for small tensors, we can
    # check linear independence of the flattened terms.

    # Flatten each rank-1 term into a vector
    def flatten_term(u, v, w):
        """Flatten a rank-1 tensor u⊗v⊗w into a vector."""
        return np.outer(np.outer(u, v), w).flatten()

    # Build the list of flattened terms
    terms = []
    for c in range(rank):
        u_c = U[:, c]
        v_c = V[:, c]
        w_c = W[:, c]
        terms.append(flatten_term(u_c, v_c, w_c))

    # Check linear independence greedily
    # We maintain a basis and try to add each term
    basis = []
    keep = []

    for c in range(rank):
        t = terms[c]
        # Check if t is in the span of basis
        if not basis:
            basis.append(t)
            keep.append(c)
        else:
            # Form matrix with basis vectors as columns
            B = np.column_stack(basis)
            # Check if t is in column space of B
            # Use SVD for numerical stability
            try:
                U_svd, s, Vt = np.linalg.svd(B, full_matrices=False)
                # Project t onto column space
                proj = U_svd @ (U_svd.T @ t)
                residual = t - proj
                if np.linalg.norm(residual) < 1e-10:
                    # t is in the span, skip it
                    continue
                else:
                    # Add to basis
                    basis.append(t)
                    keep.append(c)
            except np.linalg.LinAlgError:
                basis.append(t)
                keep.append(c)

    new_rank = len(keep)

    if new_rank < rank:
        # Rebuild factor matrices with only the kept terms
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
    else:
        U_new = U
        V_new = V
        W_new = W

    # Convert to lists for return
    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
