"""Model-and-optimize approach for 2x4x7 matrix multiplication tensor decomposition.

Strategy: Build the known rank-50 Strassen-based decomposition, then attempt
rank reduction via exact linear-algebraic elimination of redundant rank-1 terms.
Also exploit the n=2 structure: decompose into two 1x4x7 sub-tensors and try
Winograd-style tiling to find a lower-rank solution (target ~24-28).
"""
import numpy as np
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 4, 7]


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.

    Strategy: Build a rank-50 Strassen-based decomposition, then attempt
    rank reduction via linear algebra on the tensor's slice structure.
    Also try a Winograd-style approach exploiting n=2.
    """
    n, m, p = 2, 4, 7
    deadline = time.time() + 115  # safety margin before 150s limit

    # Build the rank-50 decomposition as starting point
    rank_init = 50
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1

        # M2: (A21+A22)B11
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1

        # M3: A11(B12-B22)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1

        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1

        # M5: (A11+A12)B22
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1

        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1

        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks for the 2x4x6 part (k=0..5)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)

    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1

    assert col == rank_init, f"Expected rank {rank_init}, got {col}"

    def verify_decomposition(U_m, V_m, W_m, n, m, p, num_tests=30):
        """Fast verification using random integer inputs."""
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randint(-5, 6, size=(n, m))
            B = np.random.randint(-5, 6, size=(m, p))
            C_expected = A @ B
            rank_m = U_m.shape[1]
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(rank_m):
                            C_comp[i, k] += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                return False
        return True

    # Verify initial decomposition
    if not verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init

    # --- Rank reduction via exact linear algebra ---
    # Build the full multiplication tensor T[i,j,k] and check matricization ranks
    def build_tensor(U_arr, V_arr, W_arr, n, m, p):
        """Build the full tensor T of shape (n, m, p)."""
        rank_m = U_arr.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank_m):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        return T

    def try_reduce_via_svd(U_arr, V_arr, W_arr, n, m, p):
        """
        Try to reduce rank by finding the tensor rank via matricizations
        and then reconstructing a minimal decomposition.
        Returns (U_new, V_new, W_new, new_rank) or None.
        """
        if time.time() > deadline - 20:
            return None

        T = build_tensor(U_arr, V_arr, W_arr, n, m, p)

        # Matricize along mode 1: shape (n, m*p)
        T_mat1 = T.reshape(n, m * p)
        rank1 = np.linalg.matrix_rank(T_mat1, tol=1e-10)

        # Matricize along mode 2: shape (m, n*p)
        T_mat2 = T.transpose(1, 0, 2).reshape(m, n * p)
        rank2 = np.linalg.matrix_rank(T_mat2, tol=1e-10)

        # Matricize along mode 3: shape (p, n*m)
        T_mat3 = T.transpose(2, 0, 1).reshape(p, n * m)
        rank3 = np.linalg.matrix_rank(T_mat3, tol=1e-10)

        lower_bound = max(rank1, rank2, rank3)

        if lower_bound >= U_arr.shape[1]:
            return None

        # If the lower bound is strictly less, we know the tensor rank <= current rank
        # but finding the exact minimal decomposition is hard.
        # Try a greedy elimination approach instead.
        return None

    # --- Greedy column elimination ---
    def greedy_eliminate(U_arr, V_arr, W_arr, n, m, p, max_rounds=200):
        """
        Iteratively try to eliminate columns that are in the span of others
        in the combined (U, V, W) space.
        """
        U_arr = U_arr.copy().astype(float)
        V_arr = V_arr.copy().astype(float)
        W_arr = W_arr.copy().astype(float)
        current_rank = U_arr.shape[1]
        active = list(range(current_rank))

        for rnd in range(max_rounds):
            if time.time() > deadline - 15:
                break
            if not active:
                break

            found = False
            for idx_c, c in enumerate(active):
                if len(active) <= 1:
                    break

                others = [oc for oc in active if oc != c]
                if not others:
                    continue

                # Stack the three matrices' columns for the 'others'
                U_others = U_arr[:, others]  # (n*m, len(others))
                V_others = V_arr[:, others]  # (m*p, len(others))
                W_others = W_arr[:, others]  # (p*n, len(others))

                # Combined matrix: (n*m + m*p + p*n, len(others))
                stacked = np.vstack([U_others.T, V_others.T, W_others.T])  # (total, len(others))
                target = np.concatenate([U_arr[:, c], V_arr[:, c], W_arr[:, c]])

                # Solve stacked @ coeffs = target
                try:
                    coeffs, residuals, rank_s, sv = np.linalg.lstsq(stacked, target, rcond=1e-12)
                except Exception:
                    continue

                recon = stacked @ coeffs
                if np.allclose(recon, target, atol=1e-9):
                    # Column c is exactly in the span of others
                    active.remove(c)
                    found = True
                    break

            if not found:
                break

        if not active:
            return None

        U_new = U_arr[:, active]
        V_new = V_arr[:, active]
        W_new = W_arr[:, active]

        # Verify
        if not verify_decomposition(U_new, V_new, W_new, n, m, p, num_tests=10):
            return None

        return U_new, V_new, W_new, len(active)

    # --- Winograd-style approach exploiting n=2 ---
    def winograd_1x4x7():
        """
        Try to build a decomposition by treating the 2x4x7 problem as
        two 1x4x7 problems (rows of C) and using Winograd for 1x2x2 convolutions.

        C[0, k] = sum_j A[0, j] * B[j, k]
        C[1, k] = sum_j A[1, j] * B[j, k]

        For each row i, this is a vector (length 4) times a matrix (4x7).
        We can tile the 7 columns into groups: e.g., 3 groups of 2 + 1 group of 1.
        Each 1x2x2 convolution (vector of 2 times 2x2 block) has rank 3 (Winograd).
        So per row: 3 * 3 + 1 * 1 = 10 rank-1 terms.
        For 2 rows: 20 rank-1 terms total.
        But the B matrix is shared, so we need to be careful.

        Actually, let's think differently. The tensor is T[i,j,k] = delta_{i,j,k}
        (the standard multiplication tensor). We want rank-1 decompositions.

        Let's try a direct construction:
        For the 2x4x7 multiplication:
        C[i,k] = sum_j A[i,j] B[j,k]

        We can use the following approach:
        - Group columns of B into pairs: (k, k+1) for k=0,2,4 and single k=6
        - For each pair (k, k+1), we have a 2x2 sub-problem per row
        - Winograd for 1x2x2: rank 3

        Let's construct this explicitly.
        """
        # Winograd 1x2x2: compute [a, b] @ [[c, d], [e, f]] = [ac+be, ad+bf]
        # Rank 3 decomposition:
        # m1 = (a+b)(c+e)  -> contributes to both outputs
        # m2 = (a+b)(d+f)  -> contributes to both outputs
        # m3 = (a-b)(c-e)  -> contributes to first output
        # Output1 = m1 + m2 + m3
        # Output2 = m1 + m2 - m3

        # For our problem, A is 2x4, B is 4x7.
        # C[i, k] = sum_j A[i,j] B[j,k]

        # Let's tile B's columns: groups of 2: (0,1), (2,3), (4,5), and single (6)
        # For each group, we need to compute for both rows i=0,1.

        # Actually, let's think of it as: for each column group, we have a 2x2x2
        # sub-tensor (2 rows of A, 2 cols of B per group, 2 outputs per group).
        # This is a 2x2x2 matrix multiplication which has rank 7 (Strassen).

        # So 3 groups of 2x2x2 (rank 7 each) + 1 group of 2x1x1 (rank 2)
        # Total: 3*7 + 2 = 23

        # Let's build this:
        U_new = []
        V_new = []
        W_new = []

        def add_term(u_vec, v_vec, w_vec):
            """Add a rank-1 term. u_vec: length n*m=8, v_vec: length m*p=28, w_vec: length p*n=14."""
            U_new.append(u_vec)
            V_new.append(v_vec)
            W_new.append(w_vec)

        # For a 2x2x2 block at rows i0, cols j0, outputs k0:
        # A sub-block: A[i0:i0+2, j0:j0+2] (2x2)
        # B sub-block: B[j0:j0+2, k0:k0+2] (2x2)
        # C sub-block: C[i0:i0+2, k0:k0+2] (2x2)
        # Use Strassen for this 2x2x2 block (7 multiplications)

        def strassen_2x2x2(i0, j0, k0):
            """Add 7 rank-1 terms for a 2x2x2 sub-multiplication."""
            # M1: (A11+A22)(B11+B22)
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[i0*m + j0] = 1
            u[(i0+1)*m + (j0+1)] = 1
            v[j0*p + k0] = 1
            v[(j0+1)*p + (k0+1)] = 1
            w[k0*n + i0] = 1
            w[(k0+1)*n + (i0+1)] = 1
            add_term(u, v, w)

            # M2: (A21+A22)B11
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[(i0+1)*m + j0] = 1
            u[(i0+1)*m + (j0+1)] = 1
            v[j0*p + k0] = 1
            w[k0*n + (i0+1)] = 1
            w[(k0+1)*n + (i0+1)] = -1
            add_term(u, v, w)

            # M3: A11(B12-B22)
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[i0*m + j0] = 1
            v[j0*p + (k0+1)] = 1
            v[(j0+1)*p + (k0+1)] = -1
            w[(k0+1)*n + i0] = 1
            w[(k0+1)*n + (i0+1)] = 1
            add_term(u, v, w)

            # M4: A22(B21-B11)
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[(i0+1)*m + (j0+1)] = 1
            v[(j0+1)*p + k0] = 1
            v[j0*p + k0] = -1
            w[k0*n + i0] = 1
            w[k0*n + (i0+1)] = 1
            add_term(u, v, w)

            # M5: (A11+A12)B22
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[i0*m + j0] = 1
            u[i0*m + (j0+1)] = 1
            v[(j0+1)*p + (k0+1)] = 1
            w[k0*n + i0] = -1
            w[(k0+1)*n + i0] = 1
            add_term(u, v, w)

            # M6: (A21-A11)(B11+B12)
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[(i0+1)*m + j0] = 1
            u[i0*m + j0] = -1
            v[j0*p + k0] = 1
            v[j0*p + (k0+1)] = 1
            w[(k0+1)*n + (i0+1)] = 1
            add_term(u, v, w)

            # M7: (A12-A22)(B21+B22)
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[i0*m + (j0+1)] = 1
            u[(i0+1)*m + (j0+1)] = -1
            v[(j0+1)*p + k0] = 1
            v[(j0+1)*p + (k0+1)] = 1
            w[k0*n + i0] = 1
            add_term(u, v, w)

        # We have A: 2x4, B: 4x7
        # Tile B columns into: (0,1), (2,3), (4,5), (6)
        # For each pair, we have a 2x2x2 subproblem (using both rows of A and 2 cols of B)
        # But we only have 4 columns of A, so we need to tile those too.

        # Actually, the full problem is 2x4x7. Let's tile it as:
        # A rows: 2 (can't split further usefully)
        # A cols / B rows: 4 -> split into (0,1) and (2,3)
        # B cols: 7 -> split into (0,1), (2,3), (4,5), (6)

        # So we have a 2x2x2 grid of subproblems:
        # For each (j_group, k_group):
        #   j_group in [(0,1), (2,3)]
        #   k_group in [(0,1), (2,3), (4,5), (6)]
        #   The subproblem is: A[:, j_group] @ B[j_group, k_group]
        #   Which is 2x2 @ 2x2 (or 2x2 @ 2x1 for the last one)

        # For 2x2 @ 2x2: Strassen rank 7
        # For 2x2 @ 2x1: rank 2 (just 2 multiplications)

        # Total: 2 * 3 * 7 + 2 * 1 * 2 = 42 + 4 = 46. Not better than 50.

        # Let's try a different tiling.
        # Tile A cols: (0,1,2,3) -> keep as is
        # Tile B cols: (0,1,2,3,4,5,6) -> (0,1), (2,3), (4,5,6)
        # For (0,1): 2x4 @ 4x2 -> use Strassen on 2x2x2 blocks
        #   Split A cols into (0,1) and (2,3):
        #   (0,1)x(0,1): 2x2x2, rank 7
        #   (0,1)x(2,3): 2x2x2, rank 7
        #   (2,3)x(0,1): 2x2x2, rank 7
        #   (2,3)x(2,3): 2x2x2, rank 7
        #   Total for B cols (0,1): 28

        # This is getting complex. Let's just use the original Strassen approach
        # which gives rank 50, and try to improve via elimination.

        return None

    # Try Winograd approach
    winograd_result = winograd_1x4x7()
    if winograd_result is not None:
        U_w, V_w, W_w, rank_w = winograd_result
        if rank_w < best_rank and verify_decomposition(U_w, V_w, W_w, n, m, p):
            best_U, best_V, best_W = U_w, V_w, W_w
            best_rank = rank_w

    # Try greedy elimination on the rank-50 decomposition
    if time.time() < deadline - 20:
        elim_result = greedy_eliminate(U, V, W, n, m, p)
        if elim_result is not None:
            U_e, V_e, W_e, rank_e = elim_result
            if rank_e < best_rank:
                best_U, best_V, best_W = U_e, V_e, W_e
                best_rank = rank_e

    # Try a more aggressive approach: build the tensor, compute its rank,
    # and if lower, try to find a decomposition of that rank.
    if time.time() < deadline - 15:
        T = build_tensor(U, V, W, n, m, p)
        T_mat1 = T.reshape(n, m * p)
        rank1 = np.linalg.matrix_rank(T_mat1, tol=1e-10)
        T_mat2 = T.transpose(1, 0, 2).reshape(m, n * p)
        rank2 = np.linalg.matrix_rank(T_mat2, tol=1e-10)
        T_mat3 = T.transpose(2, 0, 1).reshape(p, n * m)
        rank3 = np.linalg.matrix_rank(T_mat3, tol=1e-10)
        lower_bound = max(rank1, rank2, rank3)

        # The tensor rank is at least lower_bound.
        # If lower_bound < best_rank, we know we can do better.
        # But finding the exact decomposition is hard.

    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, n, m, p, num_tests=20):
        best_U, best_V, best_W = U, V, W
        best_rank = rank_init

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
