import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def build_strassen_56():
        """8 independent 2x2 Strassen blocks, rank 56."""
        rank = 56
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for jg in range(2):
            for kg in range(4):
                j0 = jg * 2
                j1 = jg * 2 + 1
                k0 = kg * 2
                k1 = kg * 2 + 1
                strassen_defs = [
                    ([(0 * m + j0, 1)], [(j0 * p + k0, 1)], [(k0 * n + 0, 1)]),
                    ([(1 * m + j1, 1)], [(j1 * p + k1, 1)], [(k1 * n + 1, 1)]),
                    ([(0 * m + j0, 1), (0 * m + j1, 1)], [(j0 * p + k1, 1)], [(k1 * n + 0, 1)]),
                    ([(1 * m + j0, 1), (0 * m + j0, -1)], [(j0 * p + k0, 1), (j0 * p + k1, 1)], [(k0 * n + 0, 1)]),
                    ([(1 * m + j0, 1), (1 * m + j1, 1)], [(j1 * p + k0, 1)], [(k0 * n + 1, 1)]),
                    ([(0 * m + j1, 1), (1 * m + j1, -1)], [(j1 * p + k0, 1), (j1 * p + k1, 1)], [(k1 * n + 1, 1)]),
                    ([(1 * m + j1, 1), (0 * m + j0, 1), (1 * m + j0, -1), (0 * m + j1, -1)],
                     [(j0 * p + k1, 1), (j1 * p + k1, -1)], [(k0 * n + 0, 1)]),
                ]
                for u_coeffs, v_coeffs, w_coeffs in strassen_defs:
                    for idx, val in u_coeffs:
                        U[idx, col] += val
                    for idx, val in v_coeffs:
                        V[idx, col] += val
                    for idx, val in w_coeffs:
                        W[idx, col] += val
                    col += 1
        return U, V, W

    def check_decomp(U, V, W, tol=1e-6):
        T = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=tol)

    def tensor_from_factors(U, V, W):
        """Compute the tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]."""
        T = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return T

    def build_schoolbook():
        rank = n * m * p  # 64
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return U, V, W

    # First, get the baseline Strassen-56
    U_base, V_base, W_base = build_strassen_56()
    if not check_decomp(U_base, V_base, W_base):
        # Fallback to schoolbook
        U_s, V_s, W_s = build_schoolbook()
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    best_rank = 56
    best_U, best_V, best_W = U_base, V_base, W_base

    # Now attempt global elimination: try to reduce rank by finding redundant terms
    # Strategy: treat the 56 rank-1 tensors as vectors in a high-dim space.
    # The tensor T = sum_r t_r where t_r[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # We want to find a subset of size < 56 that still sums to the all-ones tensor.
    #
    # Approach: Randomly perturb the decomposition by combining terms, then check
    # if some terms become zero or can be merged.
    #
    # Alternative: Use the fact that the tensor is the identity tensor.
    # Try to find a decomposition with rank R < 56 by solving a system.
    #
    # Practical approach: Start with 56 terms, try to eliminate one at a time.
    # For each term r, check if t_r is in the span of {t_s : s != r} under the
    # constraint that coefficients must keep the total sum = T_target.
    # This is a linear algebra problem: can we find c_s such that
    #   sum_{s != r} c_s * t_s = T_target - t_r  (i.e., = sum_{s != r} t_s)
    # which is trivially true with c_s = 1. So that doesn't work directly.
    #
    # Better: Try to find a NEW decomposition with fewer terms by using
    # a different basis. Use random linear combinations of the 56 terms to
    # create new candidate terms, then greedily select a subset.
    #
    # Actually, the most promising approach for a structural change:
    # Use the known 2x4*4x4 result. If mu(2,4,4) < 28, then mu(2,4,8) <= 2*mu(2,4,4).
    # But we don't know mu(2,4,4) exactly.
    #
    # Let me try a different structural approach:
    # View 2x4*4x8 as computing 2 rows of a 4x8 product where the top 2 rows of A are given.
    # Use a 4x4 Strassen on the j-dimension (split j into {0,1} and {2,3}),
    # but share the k-dimension handling.
    #
    # Concretely: Split A into A_left (2x2, cols 0,1) and A_right (2x2, cols 2,3).
    # Split B into B_top (2x8, rows 0,1) and B_bottom (2x8, rows 2,3).
    # C = A_left * B_top + A_right * B_bottom
    #
    # Each 2x2 * 2x8 can be done with rank 4*7 = 28 (4 independent 2x2*2x2 Strassen).
    # Total: 56. Same as before.
    #
    # To do better, we need to share between the two 2x2*2x8 computations.
    # The sharing opportunity: both use the same B_top and B_bottom structure.
    #
    # Let me try a concrete rank-48 construction:
    # Use 6 Strassen-2x2 blocks for the 8 needed 2x2*2x2 sub-problems,
    # by merging two of them.
    #
    # The 8 sub-problems are: (i=0,1) x (j=0,1 or 2,3) x (k=0,1 or 2,3 or 4,5 or 6,7)
    # That's 2 * 2 * 4 = 16 sub-problems? No.
    #
    # Let me recount: C[i,k] = sum_j A[i,j] B[j,k]
    # i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    #
    # If I split j into two groups of 2, and k into four groups of 2:
    # C[i, k0:2] = A[i, j0:2] * B[j0:2, k0:2]  (for j-group 0, k-group 0)
    # C[i, k2:4] = A[i, j0:2] * B[j0:2, k2:4]  (for j-group 0, k-group 1)
    # etc.
    #
    # That's 2 (j-groups) * 4 (k-groups) = 8 independent 2x2*2x2 products.
    # Each rank 7, total 56.
    #
    # To reduce rank, I need to share bilinear terms between these 8 products.
    # The sharing is possible because the A-factors for different k-groups are the same!
    # For a fixed j-group, A[i, j0:2] is used in all 4 k-groups.
    #
    # So for j-group 0: A_left is shared across 4 k-groups.
    # The 4 products are: A_left * B[j0:2, k0:2], A_left * B[j0:2, k2:4], etc.
    #
    # This is like computing A_left * [B_col0 | B_col1 | B_col2 | B_col3] where each B_col is 2x2.
    # That's a 2x2 * 2x8 product.
    #
    # Rank of 2x2 * 2x8: at most 4*7 = 28 via independent Strassen.
    # Can we do better? The tensor for 2x2*2x8 is 2*2*8 = 32 dimensional.
    # Lower bound: at least 16 (trivial), and Strassen gives 28.
    #
    # Known: mu(2,2,4) = 14 (2 independent 2x2*2x2 Strassen).
    # mu(2,2,8) <= 4*7 = 28. Can we do 2x2*2x8 with rank < 28?
    #
    # Actually, 2x2*2x8 = 2x2 * (2x4 * 2) ... not helpful.
    #
    # Let me just try to implement a greedy elimination on the 56-term decomposition.
    # Randomly combine pairs of terms to see if the result is still valid with fewer terms.

    # Greedy elimination attempt
    if time_left() > 20:
        U_cur, V_cur, W_cur = U_base.copy(), V_base.copy(), W_base.copy()
        rank_cur = 56

        # Try to eliminate terms by finding linear dependencies
        # Represent each rank-1 tensor as a vector
        tensors = []
        for r in range(rank_cur):
            u = U_cur[:, r].reshape(n, m)
            v = V_cur[:, r].reshape(m, p)
            w = W_cur[:, r].reshape(p, n)
            t = np.einsum('ij,jk,ki->ijk', u, v, w)
            tensors.append(t)

        target = np.ones((n, m, p))

        # Try to find a subset of size < 56 that sums to target
        # This is a set cover / subset sum problem in high dimensions.
        # Use greedy: start with all 56, try to remove one at a time.
        # When removing term r, we need sum_{s in S\{r}} t_s = target.
        # This is equivalent to: t_r = sum_{s in S\{r}} t_s - (target - t_r) ... circular.
        #
        # Actually: if sum_{s in S} t_s = target, and we remove t_r,
        # we need sum_{s in S\{r}} t_s = target, which means t_r = 0.
        # So we can only remove terms that are zero tensors.
        #
        # Alternative: Replace two terms with one. If t_a + t_b = t_c (a new term),
        # we can replace {a, b} with {c}, reducing rank by 1.
        #
        # This is hard to find in general. Let me try a different approach:
        # Use random projections to find near-redundant terms.

        # Try random linear combinations to find a smaller decomposition
        np.random.seed(42)
        for attempt in range(50):
            if time_left() < 10:
                break

            # Create a new candidate decomposition by randomly combining terms
            new_U = np.zeros((U_dim, rank_cur))
            new_V = np.zeros((V_dim, rank_cur))
            new_W = np.zeros((W_dim, rank_cur))

            for r in range(rank_cur):
                new_U[:, r] = U_cur[:, r]
                new_V[:, r] = V_cur[:, r]
                new_W[:, r] = W_cur[:, r]

            # Try to merge two random terms
            if rank_cur > 2:
                a, b = np.random.choice(rank_cur, 2, replace=False)
                # Check if t_a + t_b can be represented as a single rank-1 tensor
                # This is rarely true, so this approach is unlikely to work.
                pass

        # If no improvement, stick with 56

    # Return the best found
    if check_decomp(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Ultimate fallback: schoolbook
    U_s, V_s, W_s = build_schoolbook()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
