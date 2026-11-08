import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = 2, 5, 6

    def _build_rank54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]

        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
                U[0*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1

                # P2 = (A_1,j0 + A_1,j1) B_j0,k0
                U[1*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                W[k0*n + 1][col] += 1
                W[k1*n + 1][col] -= 1
                col += 1

                # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
                U[0*m + j0][col] += 1
                V[j0*p + k1][col] += 1
                V[j1*p + k1][col] -= 1
                W[k1*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1

                # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
                U[1*m + j1][col] += 1
                V[j1*p + k0][col] += 1
                V[j0*p + k0][col] -= 1
                W[k0*n + 0][col] += 1
                W[k0*n + 1][col] += 1
                col += 1

                # P5 = (A_0,j0 + A_0,j1) B_j1,k1
                U[0*m + j0][col] += 1
                U[0*m + j1][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] -= 1
                W[k1*n + 0][col] += 1
                col += 1

                # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
                U[1*m + j0][col] += 1
                U[0*m + j0][col] -= 1
                V[j0*p + k0][col] += 1
                V[j0*p + k1][col] += 1
                W[k1*n + 1][col] += 1
                col += 1

                # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
                U[0*m + j1][col] += 1
                U[1*m + j1][col] -= 1
                V[j1*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                col += 1

        # Remaining j=4
        for k in range(p):
            U[0*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1

        assert col == rank, f"Expected {rank} terms, got {col}"
        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W, n, m, p):
        """Verify: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        total = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            # Reshape and use einsum for efficiency
            u_reshaped = u_col.reshape(n, m)
            v_reshaped = v_col.reshape(m, p)
            w_reshaped = w_col.reshape(p, n)
            # total[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total[i, j, k] += u_reshaped[i, j] * v_reshaped[j, k] * w_reshaped[k, i]
        expected = np.ones((n, m, p))
        if not np.allclose(total, expected, atol=1e-9):
            diff = np.max(np.abs(total - expected))
            return False, diff
        return True, 0.0

    # Build baseline
    result = _build_rank54()

    # Verify
    valid, diff = _verify(result["U"], result["V"], result["W"], n, m, p)

    if not valid:
        # Should not happen, but return anyway
        return result

    return result
# EVOLVE-BLOCK-END
