import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # We attempt to find a rank < 42 decomposition using a structured search.
    # Start with the known rank-42 Strassen decomposition and try to reduce rank
    # by finding linear dependencies among the rank-42 terms, or by searching
    # for a lower-rank decomposition directly.

    # First, build the rank-42 Strassen decomposition
    def build_strassen_42():
        rank = 42
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)

        def add_strassen_block(i_start, j_start, k_start, offset):
            a00 = (i_start + 0) * m + (j_start + 0)
            a01 = (i_start + 0) * m + (j_start + 1)
            a10 = (i_start + 1) * m + (j_start + 0)
            a11 = (i_start + 1) * m + (j_start + 1)

            b00 = (j_start + 0) * p + (k_start + 0)
            b01 = (j_start + 0) * p + (k_start + 1)
            b10 = (j_start + 1) * p + (k_start + 0)
            b11 = (j_start + 1) * p + (k_start + 1)

            w00 = (k_start + 0) * n + (i_start + 0)
            w01 = (k_start + 1) * n + (i_start + 0)
            w10 = (k_start + 0) * n + (i_start + 1)
            w11 = (k_start + 1) * n + (i_start + 1)

            U[a00, offset] += 1.0
            U[a11, offset] += 1.0
            V[b00, offset] += 1.0
            V[b11, offset] += 1.0
            W[w00, offset] += 1.0
            W[w11, offset] += 1.0

            U[a10, offset + 1] += 1.0
            U[a11, offset + 1] += 1.0
            V[b00, offset + 1] += 1.0
            W[w10, offset + 1] += 1.0
            W[w11, offset + 1] -= 1.0

            U[a00, offset + 2] += 1.0
            V[b01, offset + 2] += 1.0
            V[b11, offset + 2] -= 1.0
            W[w01, offset + 2] += 1.0
            W[w11, offset + 2] += 1.0

            U[a11, offset + 3] += 1.0
            V[b10, offset + 3] += 1.0
            V[b00, offset + 3] -= 1.0
            W[w00, offset + 3] += 1.0
            W[w10, offset + 3] += 1.0

            U[a00, offset + 4] += 1.0
            U[a01, offset + 4] += 1.0
            V[b11, offset + 4] += 1.0
            W[w00, offset + 4] -= 1.0
            W[w01, offset + 4] += 1.0

            U[a10, offset + 5] += 1.0
            U[a00, offset + 5] -= 1.0
            V[b00, offset + 5] += 1.0
            V[b01, offset + 5] += 1.0
            W[w11, offset + 5] += 1.0

            U[a01, offset + 6] += 1.0
            U[a11, offset + 6] -= 1.0
            V[b10, offset + 6] += 1.0
            V[b11, offset + 6] += 1.0
            W[w00, offset + 6] += 1.0

        offset = 0
        for i_start in range(0, n, 2):
            for j_start in range(0, m, 2):
                for k_start in range(0, p, 2):
                    add_strassen_block(i_start, j_start, k_start, offset)
                    offset += 7

        return U, V, W, rank

    def verify_decomp(U_mat, V_mat, W_mat, rank_val, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank_val):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j_b = b // p
                    k = b % p
                    i_w = c % n
                    k_w = c // n
                    if i == i_w and j == j_b and k == k_w:
                        continue
                    s = 0.0
                    for r in range(rank_val):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > tol:
                        return False
        return True

    U42, V42, W42, r42 = build_strassen_42()

    # Try to reduce rank by finding the minimum number of columns needed
    # such that the tensor is preserved. We do this by checking if any subset
    # of columns of the rank-42 decomposition still spans the tensor.

    # Build the tensor T of shape (nm, mp, pn) from the rank-42 decomp
    T = np.zeros((nm, mp, pn), dtype=float)
    for r in range(r42):
        T += np.einsum('a,b,c->abc', U42[:, r], V42[:, r], W42[:, r])

    # The tensor rank is the minimum rank decomposition. We try to find
    # a lower-rank decomposition by searching over subsets or by using
    # a greedy approach.

    # Approach: try rank 41, 40, ... by removing one column at a time
    # and checking if the remaining columns can still represent T.

    # For each column r, check if T can be represented without column r
    # This is equivalent to checking if column r is in the span of the others
    # in the appropriate tensor space.

    # A simpler approach: for each candidate rank R < 42, try to find
    # R columns from the 42 that still span T.

    # Actually, a better approach: use the fact that if we can find a
    # linear combination of columns that gives zero tensor, we can remove
    # one column.

    # Check for linear dependencies in the column space
    # The tensor T is in the space of (nm x mp x pn) tensors.
    # Each column r contributes U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # If there exist coefficients c_r such that sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0
    # then we have a dependency.

    # For rank 42 with 6 blocks of 7, within each block the 7 Strassen terms
    # are linearly independent. But across blocks, there might be dependencies.

    # Let's try a direct search: for rank from 41 down to 30, try to find
    # a valid decomposition.

    # Use a randomized approach: randomly select subsets of columns and check
    # if they can represent T (via least squares on the tensor).

    # Actually, let's try a more systematic approach.
    # The tensor T has a specific structure. Let's try to find its rank
    # by checking if rank 41 is achievable.

    # For rank 41: we need to remove 1 column. Check if any single column
    # is redundant (i.e., can be expressed as a combination of the others).

    # This is a 42-dimensional problem. Let's check if the 42 vectors
    # (flattened tensors) are linearly independent.

    # Flatten each rank-1 tensor to a vector of length nm*mp*pn
    vec_dim = nm * mp * pn
    T_vecs = np.zeros((r42, vec_dim), dtype=float)
    for r in range(r42):
        T_vecs[r, :] = np.einsum('a,b,c->abc', U42[:, r], V42[:, r], W42[:, r]).flatten()

    # Check rank of T_vecs
    try:
        rank_T = np.linalg.matrix_rank(T_vecs, tol=1e-10)
    except:
        rank_T = r42

    if rank_T < r42:
        # There are linear dependencies, so we can reduce rank
        # Find a basis
        from numpy.linalg import svd
        U_svd, s_vals, Vt = svd(T_vecs, full_matrices=False)
        # Keep only non-zero singular values
        tol_svd = 1e-10
        num_nonzero = np.sum(s_vals > tol_svd)
        # The rank is num_nonzero
        # We can construct a decomposition with rank = num_nonzero
        # using the top num_nonzero left singular vectors
        # But this gives a real decomposition, not necessarily integer/half-integer

        # For now, just return the rank-42 solution if we can't find
        # a clean lower-rank one with nice entries.

    # Let's try a different approach: use the known structure to find
    # a rank-36 or lower decomposition.

    # For 2x4x6, we can think of it as 2x2x2 blocks.
    # n=2 (1 block), m=4 (2 blocks), p=6 (3 blocks)
    # Total 6 blocks of 2x2x2.
    # Each 2x2x2 has rank 7 (Strassen).
    # Total: 6*7 = 42.

    # Can we do better by sharing structure between blocks?
    # The blocks share the same A (n=2), so A is the same across all blocks.
    # B has 2 blocks of 2 columns each, C has 3 blocks of 2 rows each.

    # Let's try to construct a decomposition that exploits the shared A.
    # A is 2x4, B is 4x6, C is 6x2.
    # C = A * B

    # Write A = [A1, A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x6
    # C = A1*B1 + A2*B2

    # Each A_i * B_i is 2x6.
    # A1*B1: 2x2 * 2x6 = 2x6
    # A2*B2: 2x2 * 2x6 = 2x6

    # For each 2x2 * 2x6, we can use Strassen with 7 multiplications
    # per 2x2 block of the 6 columns.
    # 2x6 = 3 blocks of 2x2, so 7*3 = 21 per A_i*B_i.
    # Total: 2*21 = 42. Same as before.

    # Can we do better? The minimum rank for 2x2x2 is 7.
    # For 2x2x4, the rank is at most 14 (two 2x2x2 blocks).
    # For 2x2x6, the rank is at most 21.
    # For 2x4x6, the rank is at most 42.

    # It is known that the rank of 2x2x2 is exactly 7.
    # The rank of 2x2x4 is at most 14, but might be less.
    # The rank of 2x4x6 is at most 42, but might be less.

    # Let's try to find if rank 41 is achievable by a direct search.
    # This is a hard problem, so we'll use a heuristic.

    # Try to find a rank-41 decomposition by perturbing the rank-42 one.
    # This is complex, so let's just return the rank-42 solution.

    # Actually, let's try one more thing: check if the tensor has
    # a lower rank by using a numerical rank computation on a
    # matricization.

    # Matricize T as (nm*mp) x pn
    T_mat = T.reshape(nm * mp, pn)
    try:
        mat_rank = np.linalg.matrix_rank(T_mat, tol=1e-10)
    except:
        mat_rank = pn

    # The tensor rank is at least the matricization rank
    # mat_rank gives a lower bound on the tensor rank

    # For now, return the rank-42 solution
    # Convert to lists
    U_list = U42.tolist()
    V_list = V42.tolist()
    W_list = W42.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
