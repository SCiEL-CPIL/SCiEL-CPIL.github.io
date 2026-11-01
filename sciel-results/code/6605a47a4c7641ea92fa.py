"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin under the 150s wall-clock limit

    n, m, p = DIMENSIONS
    assert (n, m, p) == (2, 3, 6), "This program is specialized to 2x3x6."

    # ------------------------------------------------------------------
    # Build a valid decomposition and attempt to reduce its rank.
    # Start with the schoolbook rank-36 decomposition, then use a greedy
    # rank-reduction heuristic to find a lower-rank equivalent.
    # ------------------------------------------------------------------
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    # ------------------------------------------------------------------
    # Greedy rank reduction: try to eliminate columns that are linear
    # combinations of others, or that can be absorbed into existing terms.
    # ------------------------------------------------------------------
    def check_decomposition(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def try_remove_column(U, V, W, col_idx, n, m, p):
        """Try to remove column col_idx by expressing it as a combination of others."""
        r = len(U[0])
        if r <= 1:
            return None, None, None

        # Try to find coefficients c_s such that for all (i,j,k):
        # U[i*m+j, col_idx] * V[j*p+k, col_idx] * W[k*n+i, col_idx]
        #   == sum_s c_s * U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        # This is a system of equations. Use a simple approach: try each
        # existing column as a candidate to absorb.

        # For simplicity, try setting the column to zero and checking if
        # we can adjust other columns. This is complex, so use a heuristic:
        # if the column has small entries, try removing it.

        # A better heuristic: if U[i*m+j, col] == 0 for all i,j or
        # V[j*p+k, col] == 0 for all j,k or W[k*n+i, col] == 0 for all k,i,
        # then the column contributes nothing and can be removed.
        u_vec = [U[i * m + j][col_idx] for i in range(n) for j in range(m)]
        v_vec = [V[j * p + k][col_idx] for j in range(m) for k in range(p)]
        w_vec = [W[k * n + i][col_idx] for k in range(p) for i in range(n)]

        if all(abs(x) < 1e-10 for x in u_vec) or \
           all(abs(x) < 1e-10 for x in v_vec) or \
           all(abs(x) < 1e-10 for x in w_vec):
            # Column is already zero in one factor; remove it
            U_new = [row[:col_idx] + row[col_idx+1:] for row in U]
            V_new = [row[:col_idx] + row[col_idx+1:] for row in V]
            W_new = [row[:col_idx] + row[col_idx+1:] for row in W]
            return U_new, V_new, W_new

        return None, None, None

    # Attempt rank reduction
    improved = True
    while improved and time.time() < deadline:
        improved = False
        r = len(U[0])
        for col in range(r):
            if time.time() > deadline:
                break
            U_new, V_new, W_new = try_remove_column(U, V, W, col, n, m, p)
            if U_new is not None:
                if check_decomposition(U_new, V_new, W_new, n, m, p):
                    U, V, W = U_new, V_new, W_new
                    improved = True
                    break

    # ------------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------------
    if check_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: guaranteed schoolbook decomposition of rank n*m*p = 36.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
