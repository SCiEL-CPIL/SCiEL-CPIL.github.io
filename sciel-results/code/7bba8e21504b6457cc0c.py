import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill

    def verify(U, V, W):
        """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        R = len(U[0]) if isinstance(U, list) else U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        u = U[i*m+j][r] if isinstance(U, list) else U[i*m+j, r]
                        v = V[j*p+k][r] if isinstance(V, list) else V[j*p+k, r]
                        w = W[k*n+i][r] if isinstance(W, list) else W[k*n+i, r]
                        s += u * v * w
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def to_list(mat):
        """Convert numpy array to list of lists"""
        if isinstance(mat, list):
            return mat
        return [[float(x) for x in row] for row in mat]

    # Build the known rank-28 block decomposition
    def build_rank_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_strassen_block(rank_offset, j_offset, k_offset):
            a_idx = 0 * m + j_offset
            b_idx = 0 * m + (j_offset + 1)
            c_idx = 1 * m + j_offset
            d_idx = 1 * m + (j_offset + 1)

            e_idx = j_offset * p + k_offset
            f_idx = j_offset * p + (k_offset + 1)
            g_idx = (j_offset + 1) * p + k_offset
            h_idx = (j_offset + 1) * p + (k_offset + 1)

            r0 = rank_offset
            r1 = rank_offset + 1
            r2 = rank_offset + 2
            r3 = rank_offset + 3
            r4 = rank_offset + 4
            r5 = rank_offset + 5
            r6 = rank_offset + 6

            U[a_idx][r0] += 1; U[d_idx][r0] += 1
            U[c_idx][r1] += 1; U[d_idx][r1] += 1
            U[a_idx][r2] += 1
            U[d_idx][r3] += 1
            U[a_idx][r4] += 1; U[b_idx][r4] += 1
            U[c_idx][r5] += 1; U[a_idx][r5] -= 1
            U[b_idx][r6] += 1; U[d_idx][r6] -= 1

            V[e_idx][r0] += 1; V[h_idx][r0] += 1
            V[e_idx][r1] += 1
            V[f_idx][r2] += 1; V[h_idx][r2] -= 1
            V[g_idx][r3] += 1; V[e_idx][r3] -= 1
            V[h_idx][r4] += 1
            V[e_idx][r5] += 1; V[f_idx][r5] += 1
            V[g_idx][r6] += 1; V[h_idx][r6] += 1

            W[k_offset * n + 0][r0] += 1
            W[k_offset * n + 0][r3] += 1
            W[k_offset * n + 0][r4] -= 1
            W[k_offset * n + 0][r6] += 1

            W[(k_offset + 1) * n + 0][r2] += 1
            W[(k_offset + 1) * n + 0][r4] += 1

            W[k_offset * n + 1][r1] += 1
            W[k_offset * n + 1][r3] += 1

            W[(k_offset + 1) * n + 1][r0] += 1
            W[(k_offset + 1) * n + 1][r2] += 1
            W[(k_offset + 1) * n + 1][r5] += 1
            W[(k_offset + 1) * n + 1][r6] += 1

        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        add_strassen_block(0, 0, 0)

        # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        add_strassen_block(7, 0, 2)

        # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[:, 4:5] (schoolbook 2x2x1)
        # C04: A00*B04 + A01*B14
        U[0][14] += 1; V[0*5+4][14] += 1; W[4*2+0][14] += 1
        U[1][15] += 1; V[1*5+4][15] += 1; W[4*2+0][15] += 1
        # C14: A10*B04 + A11*B14
        U[3][16] += 1; V[0*5+4][16] += 1; W[4*2+1][16] += 1
        U[4][17] += 1; V[1*5+4][17] += 1; W[4*2+1][17] += 1

        # Block 4: A[0:2, 2:3] * B[2:3, 0:5] -> C[:, 0:5] (schoolbook 2x1x5)
        for k in range(5):
            U[2][18 + k] += 1
            V[2*5 + k][18 + k] += 1
            W[k*2 + 0][18 + k] += 1
            U[5][23 + k] += 1
            V[2*5 + k][23 + k] += 1
            W[k*2 + 1][23 + k] += 1

        return {"U": U, "V": V, "W": W}

    # Try to find lower rank solutions using iterative method
    def try_lower_rank(R, max_restarts=10, max_iters=500):
        if time.time() > deadline - 10:
            return None

        best_sol = None

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5

            best_error = float('inf')
            best_U, best_V, best_W = None, None, None

            for iteration in range(max_iters):
                if time.time() > deadline - 2:
                    break

                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, R))
                        for k in range(p):
                            for r in range(R):
                                A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        if np.linalg.matrix_rank(A) > 0:
                            try:
                                U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                            except:
                                pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, R))
                        for i in range(n):
                            for r in range(R):
                                A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        if np.linalg.matrix_rank(A) > 0:
                            try:
                                V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                            except:
                                pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, R))
                        for j in range(m):
                            for r in range(R):
                                A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        if np.linalg.matrix_rank(A) > 0:
                            try:
                                W[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                            except:
                                pass

                # Check error
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            error += abs(val - 1)

                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if error < 1e-10:
                    # Try to round to half-integers
                    for scale in [2, 4, 8]:
                        U_r = np.round(U * scale) / scale
                        V_r = np.round(V * scale) / scale
                        W_r = np.round(W * scale) / scale

                        err_r = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                                    err_r += abs(val - 1)

                        if err_r < 1e-9:
                            result = {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                            if verify(result["U"], result["V"], result["W"]):
                                return result

        return None

    # First, get the guaranteed rank-28 solution
    rank28_sol = build_rank_28()
    if not verify(rank28_sol["U"], rank28_sol["V"], rank28_sol["W"]):
        # Fallback to pure schoolbook if something went wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # --- Rank-27 exact feasibility test via structured ansatz ---
    # Fix supports of U and V to a structured pattern derived from the
    # rank-28 solution with one column removed, then solve for W over
    # rationals. If a solution exists, we have a rank-27 decomposition.
    def try_rank27_exact():
        if time.time() > deadline - 20:
            return None

        R = 27
        # Start from the rank-28 solution and try removing each column r
        # from the "free" (non-Strassen-core) region. The Strassen blocks
        # use columns 0-6 and 7-13. Columns 14-27 are the schoolbook
        # tail. Removing a schoolbook column and re-solving for W may work
        # if the remaining columns span the needed space.
        base = build_rank_28()
        U28 = base["U"]
        V28 = base["V"]
        W28 = base["W"]

        # Try removing each column from 14 to 27 (the schoolbook tail)
        # and also try removing columns from the Strassen blocks
        for removed in range(28):
            if time.time() > deadline - 5:
                break

            # Build U, V with column `removed` dropped
            U = [[0] * R for _ in range(n * m)]
            V = [[0] * R for _ in range(m * p)]

            for i in range(n * m):
                for r in range(R):
                    orig_r = r if r < removed else r + 1
                    U[i][r] = U28[i][orig_r]

            for j in range(m * p):
                for r in range(R):
                    orig_r = r if r < removed else r + 1
                    V[j][r] = V28[j][orig_r]

            # Check: for each (i,j,k), the vector a_{i,j,k} = (U[i,j,r]*V[j,k,r])
            # must have the property that there exists W[k,i,r] such that
            # sum_r W[k,i,r] * U[i,j,r] * V[j,k,r] = 1 for all (i,j,k).
            #
            # For each (k,i) pair, W[k,i,:] must satisfy:
            #   sum_r W[k,i,r] * U[i,j,r] * V[j,k,r] = 1  for all j
            #
            # This is a system of m equations in R unknowns for each (k,i).
            # But the same W[k,i,:] must work for all j simultaneously.
            # So for each (k,i): solve  A_{k,i} * W[k,i,:]^T = 1_m
            # where A_{k,i}[j, r] = U[i*m+j, r] * V[j*p+k, r]
            #
            # If the system is consistent (rank <= m and no contradictions),
            # we get a valid W.

            W = [[0] * R for _ in range(p * n)]
            valid = True

            for k in range(p):
                for i in range(n):
                    # Build A: m x R matrix where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = [[0] * R for _ in range(m)]
                    for j in range(m):
                        for r in range(R):
                            A[j][r] = U[i*m+j][r] * V[j*p+k][r]

                    # Solve A * w = 1 (vector of m ones) using exact rationals
                    # Convert to Fraction
                    A_frac = [[Fraction(A[j][r]) for r in range(R)] for j in range(m)]
                    b_frac = [Fraction(1) for _ in range(m)]

                    # Gaussian elimination over Q
                    # We need to find w such that A*w = b
                    # Augmented matrix: m x (R+1)
                    aug = []
                    for j in range(m):
                        row = [A_frac[j][r] for r in range(R)] + [b_frac[j]]
                        aug.append(row)

                    # Row reduce
                    rank = 0
                    for col in range(R):
                        # Find pivot
                        pivot = -1
                        for row in range(rank, m):
                            if aug[row][col] != 0:
                                pivot = row
                                break
                        if pivot == -1:
                            continue
                        # Swap
                        aug[rank], aug[pivot] = aug[pivot], aug[rank]
                        # Normalize pivot row
                        pv = aug[rank][col]
                        for c in range(col, R + 1):
                            aug[rank][c] /= pv
                        # Eliminate
                        for row in range(m):
                            if row == rank:
                                continue
                            factor = aug[row][col]
                            if factor != 0:
                                for c in range(col, R + 1):
                                    aug[row][c] -= factor * aug[rank][c]
                        rank += 1

                    # Check consistency: any row with all zeros in A but nonzero in b
                    for row in range(m):
                        all_zero = all(aug[row][c] == 0 for c in range(R))
                        if all_zero and aug[row][R] != 0:
                            valid = False
                            break

                    if not valid:
                        break

                    # Extract solution: w[c] = aug[row_with_pivot_at_c][R]
                    w = [Fraction(0) for _ in range(R)]
                    for row in range(rank):
                        # Find which column is the pivot
                        for c in range(R):
                            if aug[row][c] == 1:
                                w[c] = aug[row][R]
                                break

                    if valid:
                        for r in range(R):
                            W[k*n+i][r] = float(w[r])

            if not valid:
                continue

            # Verify the solution
            result = {"U": to_list(np.array(U, dtype=float)),
                      "V": to_list(np.array(V, dtype=float)),
                      "W": to_list(np.array(W, dtype=float))}

            if verify(result["U"], result["V"], result["W"]):
                return result

        return None

    # Try exact rank-27 first
    sol27 = try_rank27_exact()
    if sol27 is not None:
        return sol27

    # Try to find lower rank solutions, focusing on 27
    sol = try_lower_rank(27, max_restarts=20, max_iters=300)
    if sol is not None:
        return sol

    # Return the guaranteed rank-28 solution
    return rank28_sol
