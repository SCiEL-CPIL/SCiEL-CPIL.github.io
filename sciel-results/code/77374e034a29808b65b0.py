"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48 schoolbook
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Schoolbook decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Try to reduce rank by finding dependent columns
    # For rank r < 48, we need to find U,V,W of width r such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)

    # Strategy: try ranks from 47 down, using a structured search
    # For each candidate rank, try to find a decomposition over small fields
    # then lift to integers

    for target_rank in range(47, 30, -1):
        if time.time() > deadline:
            break

        found = False
        # Try over small prime fields
        for field in [2, 3, 5, 7]:
            if time.time() > deadline:
                break

            # Use numpy for speed
            Uf = np.zeros((n * m, target_rank), dtype=int)
            Vf = np.zeros((m * p, target_rank), dtype=int)
            Wf = np.zeros((p * n, target_rank), dtype=int)

            # Initialize with schoolbook pattern (first target_rank columns)
            for c in range(min(target_rank, n * m * p)):
                i = c // (m * p)
                j = (c // p) % m
                k = c % p
                Uf[i * m + j, c] = 1
                Vf[j * p + k, c] = 1
                Wf[k * n + i, c] = 1

            # Try random restarts with structured modifications
            for restart in range(50):
                if time.time() > deadline:
                    break

                # Reset to schoolbook
                Uf[:] = 0
                Vf[:] = 0
                Wf[:] = 0
                for c in range(min(target_rank, n * m * p)):
                    i = c // (m * p)
                    j = (c // p) % m
                    k = c % p
                    Uf[i * m + j, c] = 1
                    Vf[j * p + k, c] = 1
                    Wf[k * n + i, c] = 1

                # Try to find a valid decomposition by solving constraints
                # The constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # This is a system of 48 equations in the entries of U,V,W

                # Random perturbation approach: modify a few entries at a time
                # and check if the system remains satisfied
                for step in range(2000):
                    if time.time() > deadline:
                        break

                    # Pick a random entry to modify
                    mat = random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        row = random.randint(0, n * m - 1)
                        col = random.randint(0, target_rank - 1)
                        old_val = Uf[row, col]
                        Uf[row, col] = random.randint(0, field - 1)
                    elif mat == 'V':
                        row = random.randint(0, m * p - 1)
                        col = random.randint(0, target_rank - 1)
                        old_val = Vf[row, col]
                        Vf[row, col] = random.randint(0, field - 1)
                    else:
                        row = random.randint(0, p * n - 1)
                        col = random.randint(0, target_rank - 1)
                        old_val = Wf[row, col]
                        Wf[row, col] = random.randint(0, field - 1)

                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(target_rank):
                                    s = (s + int(Uf[i * m + j, c]) * int(Vf[j * p + k, c]) * int(Wf[k * n + i, c])) % field
                                if s != 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break

                    if valid:
                        # Found a valid decomposition over F_field
                        # Convert to integer lists
                        U_list = Uf.astype(int).tolist()
                        V_list = Vf.astype(int).tolist()
                        W_list = Wf.astype(int).tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}

                    # Revert if not valid
                    if mat == 'U':
                        Uf[row, col] = old_val
                    elif mat == 'V':
                        Vf[row, col] = old_val
                    else:
                        Wf[row, col] = old_val

        # If no lower rank found, return schoolbook
        break

    # Return schoolbook
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list[i * m + j][col] = 1
                V_list[j * p + k][col] = 1
                W_list[k * n + i][col] = 1
                col += 1
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
