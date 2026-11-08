import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = 3, 3, 4

    # Build the target tensor T[i,j,k] where i in [0,9), j in [0,12), k in [0,12)
    # T[i,j,k] = 1 if i = i_row*3+i_col, j = i_col*4+k_row, k = k_row*3+i_row
    T_expected = np.zeros((9, 12, 12), dtype=int)
    for i_row in range(3):
        for i_col in range(3):
            for k_row in range(4):
                i = i_row * 3 + i_col
                j = i_col * 4 + k_row
                k = k_row * 3 + i_row
                T_expected[i, j, k] = 1

    def verify(U, V, W):
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        return np.array_equal(T_recon, T_expected)

    # Known rank-34 Strassen-based construction
    def build_rank34():
        rank = 34
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0

        # Strassen Left (A11 * B1L) - 7 terms
        # P1
        U[0, col] = 1; U[4, col] = 1
        V[0, col] = 1; V[5, col] = 1
        W[0, col] = 1; W[4, col] = 1
        col += 1
        # P2
        U[3, col] = 1; U[4, col] = 1
        V[0, col] = 1
        W[1, col] = 1; W[4, col] = -1
        col += 1
        # P3
        U[0, col] = 1
        V[1, col] = 1; V[5, col] = -1
        W[3, col] = 1; W[4, col] = 1
        col += 1
        # P4
        U[4, col] = 1
        V[4, col] = 1; V[0, col] = -1
        W[0, col] = 1; W[1, col] = 1
        col += 1
        # P5
        U[0, col] = 1; U[1, col] = 1
        V[5, col] = 1
        W[0, col] = -1; W[3, col] = 1
        col += 1
        # P6
        U[3, col] = 1; U[0, col] = -1
        V[0, col] = 1; V[1, col] = 1
        W[4, col] = 1
        col += 1
        # P7
        U[1, col] = 1; U[4, col] = -1
        V[4, col] = 1; V[5, col] = 1
        W[0, col] = 1
        col += 1

        # A12 * B2L - 4 terms
        U[2, col] = 1; V[8, col] = 1; W[0, col] = 1; col += 1
        U[2, col] = 1; V[9, col] = 1; W[3, col] = 1; col += 1
        U[5, col] = 1; V[8, col] = 1; W[1, col] = 1; col += 1
        U[5, col] = 1; V[9, col] = 1; W[4, col] = 1; col += 1

        # A21 * B1L - 4 terms
        U[6, col] = 1; V[0, col] = 1; W[2, col] = 1; col += 1
        U[6, col] = 1; V[1, col] = 1; W[5, col] = 1; col += 1
        U[7, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1
        U[7, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1

        # A22 * B2L - 2 terms
        U[8, col] = 1; V[8, col] = 1; W[2, col] = 1; col += 1
        U[8, col] = 1; V[9, col] = 1; W[5, col] = 1; col += 1

        # Strassen Right (A11 * B1R) - 7 terms
        # P1
        U[0, col] = 1; U[4, col] = 1
        V[2, col] = 1; V[7, col] = 1
        W[6, col] = 1; W[10, col] = 1
        col += 1
        # P2
        U[3, col] = 1; U[4, col] = 1
        V[2, col] = 1
        W[7, col] = 1; W[10, col] = -1
        col += 1
        # P3
        U[0, col] = 1
        V[3, col] = 1; V[7, col] = -1
        W[9, col] = 1; W[10, col] = 1
        col += 1
        # P4
        U[4, col] = 1
        V[6, col] = 1; V[2, col] = -1
        W[6, col] = 1; W[7, col] = 1
        col += 1
        # P5
        U[0, col] = 1; U[1, col] = 1
        V[7, col] = 1
        W[6, col] = -1; W[9, col] = 1
        col += 1
        # P6
        U[3, col] = 1; U[0, col] = -1
        V[2, col] = 1; V[3, col] = 1
        W[10, col] = 1
        col += 1
        # P7
        U[1, col] = 1; U[4, col] = -1
        V[6, col] = 1; V[7, col] = 1
        W[6, col] = 1
        col += 1

        # A12 * B2R - 4 terms
        U[2, col] = 1; V[10, col] = 1; W[6, col] = 1; col += 1
        U[2, col] = 1; V[11, col] = 1; W[9, col] = 1; col += 1
        U[5, col] = 1; V[10, col] = 1; W[7, col] = 1; col += 1
        U[5, col] = 1; V[11, col] = 1; W[10, col] = 1; col += 1

        # A21 * B1R - 4 terms
        U[6, col] = 1; V[2, col] = 1; W[8, col] = 1; col += 1
        U[6, col] = 1; V[3, col] = 1; W[11, col] = 1; col += 1
        U[7, col] = 1; V[6, col] = 1; W[8, col] = 1; col += 1
        U[7, col] = 1; V[7, col] = 1; W[11, col] = 1; col += 1

        # A22 * B2R - 2 terms
        U[8, col] = 1; V[10, col] = 1; W[8, col] = 1; col += 1
        U[8, col] = 1; V[11, col] = 1; W[11, col] = 1; col += 1

        return U, V, W, rank

    U, V, W, rank = build_rank34()

    if not verify(U, V, W):
        # Fallback to trivial rank-36
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(9):
            for j in range(12):
                for k in range(12):
                    if T_expected[i, j, k] != 0:
                        U[i, col] = 1
                        V[j, col] = 1
                        W[k, col] = 1
                        col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Strategy 1: Try to find rank-33 by checking if any column can be eliminated
    # and the remaining 33 columns can be adjusted to still produce T_expected
    # This is equivalent to checking if the tensor has rank <= 33

    # For a given set of 33 columns, we need to find U', V', W' of width 33
    # such that sum_r U'_r V'_r W'_r^T = T_expected
    # This is a system of 9*12*12 = 1296 equations with 3*(9+12+12)*33 = 2574 unknowns
    # But it's nonlinear. We can use the fact that the original 34 columns work,
    # and check if one column is in the "span" of the others in a tensor sense.

    # More practically: try removing each column r, and see if we can find
    # small perturbations to the remaining columns to compensate.
    # This is a local search problem.

    # Alternative: Use the structure. The Strassen construction has symmetry.
    # The left and right halves are similar. Check if we can merge terms across
    # the left/right boundary.

    # Let's try a different approach: iterative column elimination with
    # least-squares recovery.

    def try_eliminate_column(U, V, W, r_to_eliminate, max_iter=50):
        """Try to eliminate column r by adjusting remaining columns."""
        n_cols = U.shape[1]
        if n_cols <= 1:
            return None, None, None

        # Create new matrices without column r
        mask = [i for i in range(n_cols) if i != r_to_eliminate]
        U_new = U[:, mask].copy().astype(float)
        V_new = V[:, mask].copy().astype(float)
        W_new = W[:, mask].copy().astype(float)

        # The target is T_expected
        # Current reconstruction from U_new, V_new, W_new:
        # T_current = einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
        # We want T_current + delta = T_expected
        # where delta comes from adjusting the columns

        # This is a nonlinear problem. Use a simple approach:
        # For each remaining column c, try to add a small perturbation
        # that helps close the gap.

        # Actually, let's use a different strategy:
        # The removed column contributed T_r = U_r V_r W_r^T
        # We need to absorb this into the other columns.
        # T_r is a rank-1 tensor. Can it be written as a sum of rank-1 tensors
        # that are "aligned" with the existing columns?

        # This is generally not possible unless T_r is in the span of the
        # other rank-1 tensors in the tensor product sense.

        # Let's try a randomized approach: perturb the remaining columns
        # and check if the reconstruction improves.

        T_target = T_expected.astype(float)
        T_current = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new, optimize=True)
        error = T_target - T_current

        # If the error is zero, we're done
        if np.allclose(error, 0, atol=1e-9):
            return U_new, V_new, W_new

        # Try to fix by adjusting columns. Use a simple gradient-free approach.
        # For each column c, try scaling it slightly.
        for c in range(n_cols - 1):
            if time.time() > deadline - 30:
                break
            # Try scaling column c by different factors
            for scale in [0.5, 0.75, 1.25, 1.5, 2.0, -0.5, -1.0]:
                U_test = U_new.copy()
                V_test = V_new.copy()
                W_test = W_new.copy()
                U_test[:, c] *= scale

                T_test = np.einsum('ir,jr,kr->ijk', U_test, V_test, W_test, optimize=True)
                err_test = T_target - T_test
                if np.allclose(err_test, 0, atol=1e-9):
                    return U_test, V_test, W_test

        return None, None, None

    # Try eliminating each column
    for r in range(34):
        if time.time() > deadline - 40:
            break
        U_new, V_new, W_new = try_eliminate_column(U, V, W, r)
        if U_new is not None:
            if verify(U_new.astype(int), V_new.astype(int), W_new.astype(int)):
                best_rank = 33
                best_U = U_new.astype(int)
                best_V = V_new.astype(int)
                best_W = W_new.astype(int)
                break

    # Strategy 2: More sophisticated search for rank-33
    # Use the fact that we can combine columns with integer coefficients
    # If we have two columns r1, r2 with U_r1 = a*U_r2, V_r1 = b*V_r2, W_r1 = c*W_r2
    # where a*b*c = 1 (for the tensor term to be preserved), we might merge them.

    # Actually, a better approach: try all pairs and see if one can be expressed
    # as a linear combination of the other plus adjustments to other columns.

    # Let's try a randomized local search on the factor matrices
    if best_rank > 33 and time.time() < deadline - 50:
        # Start from rank-34 and try to find a rank-33 solution
        # by randomly removing a column and then optimizing the remaining

        np.random.seed(42)
        found = False

        for trial in range(100):
            if time.time() > deadline - 30:
                break

            # Randomly choose a column to eliminate
            r_to_elim = np.random.randint(0, 34)
            mask = [i for i in range(34) if i != r_to_elim]
            U_init = U[:, mask].copy().astype(float)
            V_init = V[:, mask].copy().astype(float)
            W_init = W[:, mask].copy().astype(float)

            # Add small random perturbations
            for _ in range(20):
                U_pert = U_init + np.random.randn(*U_init.shape) * 0.1
                V_pert = V_init + np.random.randn(*V_init.shape) * 0.1
                W_pert = W_init + np.random.randn(*W_init.shape) * 0.1

                T_pert = np.einsum('ir,jr,kr->ijk', U_pert, V_pert, W_pert, optimize=True)
                if np.allclose(T_pert, T_expected, atol=1e-6):
                    # Round to nearest half-integers or integers
                    U_round = np.round(U_pert * 2) / 2
                    V_round = np.round(V_pert * 2) / 2
                    W_round = np.round(W_pert * 2) / 2

                    T_round = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round, optimize=True)
                    if np.allclose(T_round, T_expected, atol=1e-9):
                        # Check if entries are integer or half-integer
                        if np.allclose(U_round * 2, np.round(U_round * 2)) and \
                           np.allclose(V_round * 2, np.round(V_round * 2)) and \
                           np.allclose(W_round * 2, np.round(W_round * 2)):
                            best_rank = 33
                            best_U = U_round.astype(int) if np.allclose(U_round, np.round(U_round)) else U_round
                            best_V = V_round.astype(int) if np.allclose(V_round, np.round(V_round)) else V_round
                            best_W = W_round.astype(int) if np.allclose(W_round, np.round(W_round)) else W_round
                            found = True
                            break

            if found:
                break

    # Final verification
    if best_rank < 34:
        U, V, W = best_U, best_V, best_W
        rank = best_rank
    else:
        rank = 34

    # Ensure entries are integer or half-integer
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2

    if not verify(U, V, W):
        # Fallback to trivial rank-36
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(9):
            for j in range(12):
                for k in range(12):
                    if T_expected[i, j, k] != 0:
                        U[i, col] = 1
                        V[j, col] = 1
                        W[k, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
