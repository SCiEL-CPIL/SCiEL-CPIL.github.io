"""Combined program: rank-33 Strassen baseline + systematic rank-32 search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Baseline: rank 33 via 3 blocks of (Strassen 2x2x2 rank 7 + Standard 2x1x2 rank 4).
    Search: attempt to find rank 32 by merging terms across block boundaries.
    """
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p = 36."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def strassen_decomposition():
        """
        Rank-33 decomposition using Strassen for 2x2x2 blocks.
        Split A (2x3) into A1 (2x2, cols 0-1) and A2 (2x1, col 2).
        Split B (3x6) into B1 (2x6, rows 0-1) and B2 (1x6, row 2).
        C = A1*B1 + A2*B2.
        A1*B1: 2x2x6 = 3 blocks of 2x2x2, each rank 7 (Strassen), total 21.
        A2*B2: 2x1x6 = 6 blocks of 2x1x1, each rank 2, total 12.
        Total: 33.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)

        # Strassen 2x2x2 factors
        S_U = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 0], [1, 1]]),
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 1], [0, 0]]),
            np.array([[-1, 0], [1, 0]]),
            np.array([[0, 1], [0, -1]])
        ]
        S_V = [
            np.array([[1, 0], [0, 1]]),
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 1], [0, -1]]),
            np.array([[-1, 0], [1, 0]]),
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 1], [0, 0]]),
            np.array([[0, 0], [1, 1]])
        ]
        S_W = [
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1], [0, -1]]),
            np.array([[0, 0], [1, 1]]),
            np.array([[1, 1], [0, 0]]),
            np.array([[-1, 0], [1, 0]]),
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [0, 0]])
        ]

        # Standard 2x1x2 factors
        Std_U = [
            np.array([[1], [0]]),
            np.array([[1], [0]]),
            np.array([[0], [1]]),
            np.array([[0], [1]])
        ]
        Std_V = [
            np.array([[1, 0]]),
            np.array([[0, 1]]),
            np.array([[1, 0]]),
            np.array([[0, 1]])
        ]
        Std_W = [
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [1, 0]]),
            np.array([[0, 1], [0, 0]]),
            np.array([[0, 0], [0, 1]])
        ]

        col = 0
        for b in range(3):
            # Strassen part: 2x2x2 for A[0:2, 0:2] * B[0:2, 2b:2b+2]
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i, j]
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j, kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp, i]
                col += 1

            # Standard part: 2x1x2 for A[0:2, 2] * B[2, 2b:2b+2]
            for r in range(4):
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i, 0]
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0, kp]
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp, i]
                col += 1

        return U, V, W

    # Start with the known rank-33 decomposition
    best_U, best_V, best_W = strassen_decomposition()
    best_rank = 33

    # Verify the baseline
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if Strassen construction is wrong
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    # Try to find rank 32 via structured search
    # Strategy: try to merge one Strassen term with one standard term
    # by adjusting coefficients slightly (this is a heuristic search)
    if best_rank > 32 and time.time() < deadline - 15:
        base_U = best_U.copy().astype(np.float64)
        base_V = best_V.copy().astype(np.float64)
        base_W = best_W.copy().astype(np.float64)

        # Try removing each column and see if we can fix the residual
        # by adjusting other columns (small perturbations)
        for drop_col in range(33):
            if time.time() > deadline - 15:
                break

            U_test = np.delete(base_U, drop_col, axis=1)
            V_test = np.delete(base_V, drop_col, axis=1)
            W_test = np.delete(base_W, drop_col, axis=1)

            # Compute residual
            residual = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
            for c in range(32):
                residual += np.einsum('i,j,k->ijk', U_test[:, c], V_test[:, c], W_test[:, c])
            residual = T - residual

            # Check if residual can be expressed as a linear combination
            # of the existing 32 rank-1 terms (i.e., if it lies in their span)
            # This is a complex check; for now, just verify if dropping works
            if np.allclose(residual, 0, atol=1e-10):
                best_U = U_test.astype(np.int64)
                best_V = V_test.astype(np.int64)
                best_W = W_test.astype(np.int64)
                best_rank = 32
                break

    # Additional search: try small integer perturbations of the rank-33 solution
    # to see if we can find an equivalent rank-32 solution
    if best_rank > 32 and time.time() < deadline - 10:
        # Try various column elimination strategies
        for drop_col in range(33):
            if time.time() > deadline - 10:
                break

            # Try to express the dropped term as a combination of others
            # by solving a linear system (simplified heuristic)
            U_test = best_U[:, :drop_col].copy().astype(np.float64)
            V_test = best_V[:, :drop_col].copy().astype(np.float64)
            W_test = best_W[:, :drop_col].copy().astype(np.float64)

            U_test = np.hstack([U_test, best_U[:, drop_col+1:]].astype(np.float64))
            V_test = np.hstack([V_test, best_V[:, drop_col+1:]].astype(np.float64))
            W_test = np.hstack([W_test, best_W[:, drop_col+1:]].astype(np.float64))

            if verify_decomposition(U_test.astype(np.int64), V_test.astype(np.int64), W_test.astype(np.int64)):
                best_U = U_test.astype(np.int64)
                best_V = V_test.astype(np.int64)
                best_W = W_test.astype(np.int64)
                best_rank = 32
                break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
