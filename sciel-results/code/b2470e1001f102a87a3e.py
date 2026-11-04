import numpy as np
import time
from itertools import product
from fractions import Fraction
from math import isclose
import sys

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Build the matrix multiplication tensor T[u,v,w]
    # u = i*m + j (input: row i, col j of A)
    # v = j*p + k (input: row j, col k of B)
    # w = k*n + i (output: row i, col k)  <-- official trace convention W[k*n+i, r]
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    schoolbook_rank = n * m * p  # 100

    # Schoolbook fallback (rank 100)
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=float)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=float)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1.0
                V_sb[v, col] = 1.0
                W_sb[w, col] = 1.0
                col += 1

    def check_solution(U, V, W, tol=1e-9):
        # Exact-ish check using float; if entries are integers/half-integers this is reliable
        T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
        err = np.linalg.norm(T_calc - T)
        return err < tol

    def snap_to_half_int(U, V, W):
        U_int = np.round(U * 2) / 2.0
        V_int = np.round(V * 2) / 2.0
        W_int = np.round(W * 2) / 2.0
        return U_int, V_int, W_int

    def snap_to_int(U, V, W):
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        return U_int, V_int, W_int

    # ------------------------------------------------------------------
    # NEW STRATEGY:
    # Combinatorial search over sparse integer patterns for each rank-1 term.
    # For each term t, we choose:
    #   - support sets SU_t ⊆ [n*m], SV_t ⊆ [m*p], SW_t ⊆ [p*n]
    #   - signs in {+1, -1} for each supported entry
    # Then we solve for scale factors alpha_t, beta_t, gamma_t (scalars per term)
    # such that the sum of rank-1 tensors equals T.
    #
    # But solving for 3r unknowns is hard. Instead, we fix a "canonical" scaling:
    # For each term, we require that the product of the max absolute values is 1,
    # and we search over sign patterns only. Then we verify if the sum equals T.
    #
    # Even better: We can use the fact that T has entries in {0,1}.
    # We'll search for decompositions where each term is a "signed permutation-like"
    # pattern, and use a greedy/beam search to build the decomposition term by term.
    # ------------------------------------------------------------------

    # Precompute the list of non-zero entries of T
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                nonzero_entries.append((u, v, w, 1.0))

    num_nonzero = len(nonzero_entries)  # 100

    # Strategy: Try to find a rank-r decomposition by searching for r rank-1 terms
    # each of which is a "simple" pattern (e.g., a single non-zero entry, or a small
    # structured pattern). We use a beam search over the space of partial decompositions.

    # For efficiency, we'll represent a rank-1 term as a tuple of three sparse vectors.
    # But storing full vectors is expensive. Instead, we'll use a dictionary representation.

    # Let's try a different approach: use the known structure of matrix multiplication.
    # The tensor T is the "matrix multiplication tensor" of format (4,5,5).
    # We know that rank(4x5x5) <= 100 (schoolbook).
    # We'll try to find a rank < 100 decomposition by using a "Strassen-like"
    # combination of smaller blocks.

    # For 4x5x5, we can view it as:
    #   A (4x5) * B (5x5) = C (4x5)
    # We can partition A into 2x2 blocks of size 2x2 and 2x3, etc.
    # But let's try a more direct approach: use the fact that the tensor is
    # a sum of 100 rank-1 terms (schoolbook), and try to merge some of them.

    # Actually, let's try a completely different approach:
    # Use the "Laderman" algorithm for 3x3x3 (rank 23) and combine with
    # additional terms for the remaining dimensions.

    # For 4x5x5, we can write:
    #   4x5x5 = (3+1)x(3+2)x(3+2)
    # We can use a block decomposition:
    #   A = [[A11, A12], [A21, A22]] where A11 is 3x3, A12 is 3x2, A21 is 1x3, A22 is 1x2
    #   B = [[B11, B12], [B21, B22]] where B11 is 3x3, B12 is 3x2, B21 is 2x3, B22 is 2x2
    #   C = A*B = [[C11, C12], [C21, C22]] where C11 is 3x3, C12 is 3x2, C21 is 1x3, C22 is 1x2

    # We can use Laderman for the 3x3x3 block (rank 23), and schoolbook for the rest.
    # But this requires careful bookkeeping.

    # Let's try a simpler approach: use a "tensor train" or "Tucker" decomposition
    # with integer/half-integer entries.

    # Actually, the most promising approach is to use the "Strassen" algorithm
    # for 2x2x2 (rank 7) and combine it with additional terms.

    # For 4x5x5, we can partition into 2x2 blocks:
    #   A = [[A11, A12, A13], [A21, A22, A23]] where each block is 2x2 or 2x1
    #   B = [[B11, B12, B13], [B21, B22, B23]] where each block is 2x2 or 1x2
    #   C = A*B = [[C11, C12, C13], [C21, C22, C23]] where each block is 2x2 or 2x1

    # This is getting complicated. Let's try a more direct approach:
    # Use a "randomized" search over integer/half-integer patterns, but with
    # a much more efficient verification.

    # Key insight: We can use the fact that T has a very sparse structure.
    # Each entry T[u,v,w] is 1 if and only if u = i*m+j, v = j*p+k, w = k*n+i
    # for some i,j,k. So the tensor is "diagonal" in a certain sense.

    # Let's try a "greedy" approach: at each step, find the rank-1 term that
    # best matches the remaining tensor, subtract it, and repeat.

    def greedy_decompose(max_rank, max_iters_per_term=10000):
        """
        Greedy decomposition: at each step, find the rank-1 term that best
        matches the remaining tensor, subtract it, and repeat.
        """
        T_rem = T.copy()
        U_terms = []
        V_terms = []
        W_terms = []

        for term_idx in range(max_rank):
            if time.time() > deadline - 5:
                break

            # Find the best rank-1 term that matches T_rem
            # We'll search over a limited set of candidate patterns.

            best_err = float('inf')
            best_U = None
            best_V = None
            best_W = None

            # Candidate patterns: sparse patterns with at most 3 non-zero entries in each vector
            # We'll use a randomized search over a subset of patterns.

            num_candidates = min(2000, max(100, (int(deadline - time.time()) * 100) // 10))

            for cand_idx in range(num_candidates):
                if time.time() > deadline - 5:
                    break

                # Generate a random sparse pattern
                # U: at most 3 non-zero entries
                num_u = np.random.randint(1, 4)
                u_indices = np.random.choice(n * m, num_u, replace=False)
                u_signs = np.random.choice([-1.0, 1.0], num_u)

                # V: at most 3 non-zero entries
                num_v = np.random.randint(1, 4)
                v_indices = np.random.choice(m * p, num_v, replace=False)
                v_signs = np.random.choice([-1.0, 1.0], num_v)

                # W: at most 3 non-zero entries
                num_w = np.random.randint(1, 4)
                w_indices = np.random.choice(p * n, num_w, replace=False)
                w_signs = np.random.choice([-1.0, 1.0], num_w)

                U_cand = np.zeros(n * m)
                V_cand = np.zeros(m * p)
                W_cand = np.zeros(p * n)

                U_cand[u_indices] = u_signs
                V_cand[v_indices] = v_signs
                W_cand[w_indices] = w_signs

                # Compute the rank-1 tensor
                T_cand = np.einsum('u,v,w->uvw', U_cand, V_cand, W_cand, optimize=True)

                # Compute the error
                err = np.linalg.norm(T_rem - T_cand)

                if err < best_err:
                    best_err = err
                    best_U = U_cand
                    best_V = V_cand
                    best_W = W_cand

            if best_U is None:
                break

            # Subtract the best term from the remaining tensor
            T_rem -= np.einsum('u,v,w->uvw', best_U, best_V, best_W, optimize=True)

            # Store the term
            U_terms.append(best_U)
            V_terms.append(best_V)
            W_terms.append(best_W)

            # Check if we've reached zero
            if np.linalg.norm(T_rem) < 1e-9:
                break

        if not U_terms:
            return None

        # Assemble the full matrices
        U = np.column_stack(U_terms)
        V = np.column_stack(V_terms)
        W = np.column_stack(W_terms)

        # Verify the solution
        if check_solution(U, V, W):
            return U, V, W

        return None

    # Try greedy decomposition for various ranks
    best_result = None
    best_rank = schoolbook_rank

    for r in range(80, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break

        result = greedy_decompose(r)
        if result:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = r
            break

    if best_result is not None:
        U, V, W = best_result
        # Try to snap to integers/half-integers
        for snap_func in [snap_to_int, snap_to_half_int]:
            U_s, V_s, W_s = snap_func(U, V, W)
            if check_solution(U_s, V_s, W_s):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
