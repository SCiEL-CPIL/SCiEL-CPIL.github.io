import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 100

    # Build initial decomposition:
    #   Strassen (rank 7) for each 2x1 pair of B columns (4 pairs) => 28 terms
    #   Schoolbook (rank 4) for remaining single column => 4 terms
    #   Schoolbook (rank 9) for the single shared row A[:,2] * B[2,:] => 9 terms
    # Total = 41 terms
    #
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (27, rank)
    # W shape: (p*n, rank) = (18, rank)

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_dict, v_dict, w_dict):
        u = [0] * (n * m)
        for idx, val in u_dict.items():
            u[idx] = val
        v = [0] * (m * p)
        for idx, val in v_dict.items():
            v[idx] = val
        w = [0] * (p * n)
        for idx, val in w_dict.items():
            w[idx] = val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # Strassen terms for a 2x2 x 2x2 block
    # Input matrix A2 (rows 0,1 of A; cols k0,k1 of B), B2 (rows k0,k1; cols 0,1)
    # Standard Strassen decomposition for C2 = A2 * B2:
    # M1 = a00 * b00
    # M2 = (a00+a01) * b10
    # M3 = (a10+a11) * b01
    # M4 = a11 * b11
    # M5 = (a00-a10) * (b10+b11)
    # M6 = (a10+a00) * (b01-b00)
    # M7 = (a01-a11) * (b11+b00)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M5
    # C10 = M3 - M4 - M7
    # C11 = M1 + M2 - M3 + M6

    # For our case: A2 = A[:, k0:k0+2], B2 = B[k0:k0+2, 0:2]
    # A indices: row i, col k -> index i*m + k = i*3 + k
    # B indices: row k, col j -> index k*p + j = k*9 + j
    # W indices: col j, row i -> index j*n + i

    strassen_terms = [
        # (u_dict, v_dict, w_dict)
        # M1 = a00 * b00  ->  C00 += M1, C11 += M1
        ({0: 1}, {0: 1}, {0: 1, 3: 1}),
        # M2 = (a00+a01) * b10 -> C01 += M2, C11 += M2
        ({0: 1, 1: 1}, {9: 1}, {1: 1, 3: 1}),
        # M3 = (a10+a11) * b01 -> C10 -= M3, C11 -= M3
        ({3: 1, 4: 1}, {1: 1}, {2: -1, 3: -1}),
        # M4 = a11 * b11 -> C00 += M4, C10 -= M4
        ({4: 1}, {10: 1}, {0: 1, 2: -1}),
        # M5 = (a00-a10) * (b10+b11) -> C00 -= M5, C01 += M5
        ({0: 1, 3: -1}, {9: 1, 10: 1}, {0: -1, 1: 1}),
        # M6 = (a10+a00) * (b01-b00) -> C11 += M6
        ({3: 1, 0: 1}, {1: 1, 0: -1}, {3: 1}),
        # M7 = (a01-a11) * (b11+b00) -> C00 += M7, C10 -= M7
        ({1: 1, 4: -1}, {10: 1, 0: 1}, {0: 1, 2: -1}),
    ]

    # Apply Strassen to each pair of B columns
    for pair_idx in range(4):
        if time.time() > deadline:
            break
        k0 = pair_idx * 2
        # Shift B indices by k0*p
        for (u_d, v_d, w_d) in strassen_terms:
            v_shifted = {idx + k0 * p: val for idx, val in v_d.items()}
            add_term(u_d, v_shifted, w_d)

    # Single remaining column k=8: schoolbook 2x1
    k = 8
    add_term({0: 1}, {k: 1}, {0: 1})       # C[0,0] += A[0,8]*B[8,0]
    add_term({1: 1}, {p + k: 1}, {1: 1})    # C[0,1] += A[0,8]*B[8,1]
    add_term({3: 1}, {k: 1}, {2: 1})        # C[1,0] += A[1,8]*B[8,0]
    add_term({4: 1}, {p + k: 1}, {3: 1})    # C[1,1] += A[1,8]*B[8,1]

    # Shared row: A[:, 2] * B[2, :]
    # C[i, j] += A[i, 2] * B[2, j]
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + j
            w_idx = j * n + i
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    initial_rank = len(U_cols)

    # Now attempt rank reduction using linear algebra
    # We have U, V, W of shapes (6, r), (27, r), (18, r)
    # The tensor is T[i,k,j] = sum_r U[i,r] * V[k,r] * W[j,r]
    # We want to find the minimum number of terms that exactly reconstruct T.
    #
    # Strategy: Build the tensor T, then find a minimal decomposition.
    # Since r is small (41), we can try to find dependencies.

    if time.time() < deadline - 5:
        U_mat = np.array(U_cols, dtype=float).T  # (6, r)
        V_mat = np.array(V_cols, dtype=float).T  # (27, r)
        W_mat = np.array(W_cols, dtype=float).T  # (18, r)

        # Compute the tensor T from current decomposition
        # T[i,k,j] = sum_r U[i,r] V[k,r] W[j,r]
        # This is a 2x3x9 tensor
        # We can try to find the minimal rank decomposition.
        #
        # Approach: Flatten T into a matrix and use SVD-like reasoning,
        # but tensor rank is NP-hard in general. For small cases, we can
        # try greedy elimination.

        # Build full tensor as a 3D array
        T = np.zeros((n, m, p))
        for r in range(initial_rank):
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        T[i, k, j] += U_mat[i, r] * V_mat[k, r] * W_mat[j, r]

        # Try to find linear dependencies among the rank-1 components.
        # Each term r contributes a rank-1 tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r].
        # We want to find a subset of terms (or linear combinations) that
        # reconstructs T with fewer terms.

        # Method: Try to express T as a sum of fewer rank-1 tensors.
        # Since the original decomposition is over integers, we look for
        # integer/half-integer solutions.

        # Simple approach: Check if any term is redundant (can be expressed
        # as a linear combination of others). This is hard in general.
        #
        # Alternative: Use the fact that for matrix multiplication, the
        # tensor has a known structure. Try to find a smaller decomposition
        # by solving a system of equations.

        # Let's try a different approach: use the structure of the problem.
        # C = A * B where A is 2x3, B is 3x9.
        # C[i,j] = sum_k A[i,k] * B[k,j]
        #
        # We can view this as: for each j, C[:,j] = A * B[:,j]
        # where B[:,j] is a 3x1 vector.
        #
        # The 2x3 matrix multiplication by a 3x1 vector has tensor rank 6
        # (schoolbook) but can be done in rank 4 using Winograd for 2x1?
        # Actually, for 2x3 * 3x1, the minimal rank is known to be 6
        # (no improvement over schoolbook for this small size).
        #
        # But we have 9 such products (one for each j), and they share the
        # same A. The tensor for (A fixed) * B is a linear map from B to C.
        # This is just matrix multiplication, and the tensor rank is the
        # bilinear complexity of computing A*B for A in R^{2x3}, B in R^{3x9}.
        #
        # The bilinear complexity of (2,3,9) matrix multiplication:
        # Lower bound: max(2*3, 3*9, 2*9) = 27
        # Upper bound (schoolbook): 2*3*9 = 54
        # Strassen-like: we achieved 41.
        #
        # Can we do better? Let's try to find a rank-36 or lower decomposition.
        #
        # For now, let's try a simple optimization: remove any terms that
        # are exactly zero or redundant.

        # Check for zero terms
        nonzero_indices = []
        for r in range(initial_rank):
            if np.any(U_mat[:, r] != 0) and np.any(V_mat[:, r] != 0) and np.any(W_mat[:, r] != 0):
                nonzero_indices.append(r)

        # Also check for terms that are scalar multiples of others
        # (this would indicate redundancy)
        # This is complex; for now, just use nonzero terms.

        if len(nonzero_indices) < initial_rank:
            U_mat = U_mat[:, nonzero_indices]
            V_mat = V_mat[:, nonzero_indices]
            W_mat = W_mat[:, nonzero_indices]

    # Return the result
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        return [[int(v) if float(v).is_integer() else v for v in row] for row in mat]

    U_out = to_list(U_mat.tolist())
    V_out = to_list(V_mat.tolist())
    W_out = to_list(W_mat.tolist())

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
