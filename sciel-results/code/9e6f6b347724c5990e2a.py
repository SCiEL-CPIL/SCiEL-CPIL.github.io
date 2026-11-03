"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline with margin

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    best_rank = schoolbook_rank
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find a lower rank decomposition using random initialization and gradient-free search
    # For 2x3x5, known bounds: rank is between 22 and 30
    # Try ranks from 22 to 27

    for target_rank in range(22, 28):
        if time.time() > deadline - 30:
            break

        # Random initialization
        best_loss = float('inf')
        best_U_try = None
        best_V_try = None
        best_W_try = None

        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 30:
                break

            # Initialize with small random values
            U_r = np.random.randn(n * m, target_rank) * 0.1
            V_r = np.random.randn(m * p, target_rank) * 0.1
            W_r = np.random.randn(p * n, target_rank) * 0.1

            # Use simple iterative refinement
            # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # We want T[i,j,k] = 1 for all i,j,k

            # Compute current tensor and error
            def compute_tensor(U_mat, V_mat, W_mat):
                T = np.zeros((n, m, p))
                for c in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i, j, k] += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                return T

            def compute_loss(U_mat, V_mat, W_mat):
                T = compute_tensor(U_mat, V_mat, W_mat)
                return np.sum((T - 1.0) ** 2)

            current_loss = compute_loss(U_r, V_r, W_r)

            # Simple coordinate descent / random perturbation
            num_iterations = 1000
            step_size = 0.05
            for it in range(num_iterations):
                if time.time() > deadline - 30:
                    break

                # Randomly pick a parameter to update
                choice = random.random()
                if choice < 1/3:
                    idx = (random.randint(0, n * m - 1), random.randint(0, target_rank - 1))
                    old_val = U_r[idx]
                    U_r[idx] += random.gauss(0, step_size)
                elif choice < 2/3:
                    idx = (random.randint(0, m * p - 1), random.randint(0, target_rank - 1))
                    old_val = V_r[idx]
                    V_r[idx] += random.gauss(0, step_size)
                else:
                    idx = (random.randint(0, p * n - 1), random.randint(0, target_rank - 1))
                    old_val = W_r[idx]
                    W_r[idx] += random.gauss(0, step_size)

                new_loss = compute_loss(U_r, V_r, W_r)
                if new_loss > current_loss:
                    # Revert
                    if choice < 1/3:
                        U_r[idx] = old_val
                    elif choice < 2/3:
                        V_r[idx] = old_val
                    else:
                        W_r[idx] = old_val
                else:
                    current_loss = new_loss

                # Decrease step size
                step_size *= 0.999

            if current_loss < best_loss:
                best_loss = current_loss
                best_U_try = U_r.copy()
                best_V_try = V_r.copy()
                best_W_try = W_r.copy()

        # Check if we found a valid decomposition
        if best_U_try is not None and best_loss < 1e-6:
            # Verify
            T = compute_tensor(best_U_try, best_V_try, best_W_try)
            if np.max(np.abs(T - 1.0)) < 1e-4:
                # Convert to lists
                U_list = [[float(best_U_try[i, c]) for c in range(target_rank)] for i in range(n * m)]
                V_list = [[float(best_V_try[i, c]) for c in range(target_rank)] for i in range(m * p)]
                W_list = [[float(best_W_try[i, c]) for c in range(target_rank)] for i in range(p * n)]
                return {"U": U_list, "V": V_list, "W": W_list}

    # Return schoolbook if no better found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
