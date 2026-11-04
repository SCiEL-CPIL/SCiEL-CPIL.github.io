"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen for 2x2x2 blocks
    rank = 42

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(i_start, j_start, k_start, offset):
        # Adds a 2x2x2 Strassen block
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1

        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1

        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1

        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    # Part 1: A[0:2, 0:2] * B[0:2, 0:6]
    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)

    # Part 2: A[0:2, 2:4] * B[2:4, 0:6]
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        rank = len(U_mat[0])
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_np[i * m + j]
                    v_val = V_np[j * p + k]
                    w_val = W_np[k * n + i]
                    total = np.dot(u_val * v_val, w_val)
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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

    # Try to reduce rank by checking if any factor is redundant
    # A factor r is redundant if removing it still gives a valid decomposition
    # (i.e., the tensor entries that relied on factor r are zero or compensated)
    
    def try_remove_factor(U_list, V_list, W_list, r_to_remove, n, m, p):
        rank = len(U_list[0])
        new_rank = rank - 1
        if new_rank <= 0:
            return None
        U_new = [row[:r_to_remove] + row[r_to_remove+1:] for row in U_list]
        V_new = [row[:r_to_remove] + row[r_to_remove+1:] for row in V_list]
        W_new = [row[:r_to_remove] + row[r_to_remove+1:] for row in W_list]
        
        U_np = np.array(U_new, dtype=float)
        V_np = np.array(V_new, dtype=float)
        W_np = np.array(W_new, dtype=float)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_np[i * m + j]
                    v_val = V_np[j * p + k]
                    w_val = W_np[k * n + i]
                    total = np.dot(u_val * v_val, w_val)
                    if abs(total - 1.0) > 1e-9:
                        return None
        return (U_new, V_new, W_new)

    # Try removing each factor one at a time
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    for r in range(rank):
        if time.time() > deadline - 10:
            break
        result = try_remove_factor(U, V, W, r, n, m, p)
        if result is not None:
            U, V, W = result
            rank = rank - 1
            # After removing one, try again from the beginning
            r = -1  # Reset loop
            if time.time() > deadline - 10:
                break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
