import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 5, 6

    def _build_rank54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        # Pairs of j indices: (0,1) and (2,3)
        j_blocks = [(0, 1), (2, 3)]
        # Pairs of k indices: (0,1), (2,3), (4,5)
        k_blocks = [(0, 1), (2, 3), (4, 5)]

        for j0, j1 in j_blocks:
            for k0, k1 in k_blocks:
                # Term 1: (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
                U[0*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1

                # Term 2: (A_1,j0 + A_1,j1) B_j0,k0
                U[1*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                W[k0*n + 1][col] += 1
                W[k1*n + 1][col] -= 1
                col += 1

                # Term 3: A_0,j0 (B_j0,k1 - B_j1,k1)
                U[0*m + j0][col] += 1
                V[j0*p + k1][col] += 1
                V[j1*p + k1][col] -= 1
                W[k1*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1

                # Term 4: A_1,j1 (B_j1,k0 - B_j0,k0)
                U[1*m + j1][col] += 1
                V[j1*p + k0][col] += 1
                V[j0*p + k0][col] -= 1
                W[k0*n + 0][col] += 1
                W[k0*n + 1][col] += 1
                col += 1

                # Term 5: (A_0,j0 + A_0,j1) B_j1,k1
                U[0*m + j0][col] += 1
                U[0*m + j1][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] -= 1
                W[k1*n + 0][col] += 1
                col += 1

                # Term 6: (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
                U[1*m + j0][col] += 1
                U[0*m + j0][col] -= 1
                V[j0*p + k0][col] += 1
                V[j0*p + k1][col] += 1
                W[k1*n + 1][col] += 1
                col += 1

                # Term 7: (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
                U[0*m + j1][col] += 1
                U[1*m + j1][col] -= 1
                V[j1*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                col += 1

        # Handle remaining j=4 trivially
        for k in range(p):
            U[0*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1

        assert col == rank, f"Rank mismatch: {col} vs {rank}"
        return {"U": U, "V": V, "W": W}

    def _verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
        total = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U_arr[:, r].reshape(n, m)
            v_r = V_arr[:, r].reshape(m, p)
            w_r = W_arr[:, r].reshape(p, n)
            total += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
        
        expected = np.ones((n, m, p))
        if not np.allclose(total, expected, atol=1e-9):
            return False
        return True

    result = _build_rank54()
    
    # Optional verification in debug mode
    # if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
    #     print("Verification failed")
    
    return result
# EVOLVE-BLOCK-END
