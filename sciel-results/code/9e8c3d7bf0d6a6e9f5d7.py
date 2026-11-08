import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 4, 5, 5

    def build_rank_85():
        rank = 85
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        # 4x4x4 Strassen part: 49 tensors
        # Build 2x2x2 Strassen factors
        s_u = np.zeros((4, 7), dtype=float)
        s_v = np.zeros((4, 7), dtype=float)
        s_w = np.zeros((4, 7), dtype=float)

        # Term 0: (a00+a11)(b00+b11)
        s_u[0, 0] = 1; s_u[3, 0] = 1
        s_v[0, 0] = 1; s_v[3, 0] = 1
        s_w[0, 0] = 1; s_w[3, 0] = 1

        # Term 1: (a10+a11)b00
        s_u[2, 1] = 1; s_u[3, 1] = 1
        s_v[0, 1] = 1
        s_w[2, 1] = 1; s_w[3, 1] = -1

        # Term 2: a00(b01-b11)
        s_u[0, 2] = 1
        s_v[1, 2] = 1; s_v[3, 2] = -1
        s_w[1, 2] = 1; s_w[3, 2] = 1

        # Term 3: a11(b10-b00)
        s_u[3, 3] = 1
        s_v[2, 3] = 1; s_v[0, 3] = -1
        s_w[0, 3] = 1; s_w[2, 3] = 1

        # Term 4: (a00+a01)b11
        s_u[0, 4] = 1; s_u[1, 4] = 1
        s_v[3, 4] = 1
        s_w[0, 4] = -1; s_w[1, 4] = 1

        # Term 5: (a10-a00)(b00+b01)
        s_u[0, 5] = -1; s_u[2, 5] = 1
        s_v[0, 5] = 1; s_v[1, 5] = 1
        s_w[3, 5] = 1

        # Term 6: (a01-a11)(b10+b11)
        s_u[1, 6] = 1; s_u[3, 6] = -1
        s_v[2, 6] = 1; s_v[3, 6] = 1
        s_w[0, 6] = 1

        # Terms mapping for 4x4x4
        terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]

        for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
            for inner_k in range(7):
                col_idx = t_idx * 7 + inner_k
                u_vec = s_u[:, inner_k]
                v_vec = s_v[:, inner_k]
                w_vec = s_w[:, inner_k]

                for (a_br, a_bc), a_c in a_coeff.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            u_val = u_vec[local_idx]
                            if u_val != 0:
                                gi = 2 * a_br + lr
                                gj = 2 * a_bc + lc
                                U[gi * m + gj, col_idx] += a_c * u_val

                for (b_br, b_bc), b_c in b_coeff.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            v_val = v_vec[local_idx]
                            if v_val != 0:
                                gj = 2 * b_br + lr
                                gk = 2 * b_bc + lc
                                V[gj * p + gk, col_idx] += b_c * v_val

                for (c_br, c_bc), c_c in c_coeff.items():
                    for lr in range(2):
                        for lc in range(2):
                            local_idx = lr * 2 + lc
                            w_val = w_vec[local_idx]
                            if w_val != 0:
                                gk = 2 * c_bc + lc
                                gi = 2 * c_br + lr
                                W[gk * n + gi, col_idx] += c_c * w_val

        col = 49

        # 4x4x1 part: 16
        for i in range(4):
            for j in range(4):
                U[i * m + j, col] = 1
                V[j * p + 4, col] = 1
                W[4 * n + i, col] = 1
                col += 1

        # 4x1x4 part: 16
        for i in range(4):
            for k in range(4):
                U[i * m + 4, col] = 1
                V[4 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        # 4x1x1 part: 4
        for i in range(4):
            U[i * m + 4, col] = 1
            V[4 * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if j == k else 0.0
                    u_vals = U[i * m + j, :]
                    v_vals = V[j * p + k, :]
                    w_vals = W[k * n + i, :]
                    prod_sum = np.dot(u_vals, v_vals * w_vals)
                    if abs(prod_sum - expected) > 1e-9:
                        return False
        return True

    best_solution = build_rank_85()
    best_rank = 85

    if verify(best_solution['U'], best_solution['V'], best_solution['W']):
        # Search for rank-84 by trying to merge columns
        # Heuristic: check if any column can be zeroed out while still valid
        # or if any pair of columns can be combined
        U_arr = np.array(best_solution['U'], dtype=float)
        V_arr = np.array(best_solution['V'], dtype=float)
        W_arr = np.array(best_solution['W'], dtype=float)

        # Try removing each column and see if we can fix with a single perturbation
        # This is a very limited search
        for trial in range(min(50, int((deadline - time.time()) / 0.5))):
            if time.time() > deadline:
                break

            # Try removing a random column
            r = random.randint(0, 84)
            U_try = U_arr.copy()
            V_try = V_arr.copy()
            W_try = W_arr.copy()

            # Zero out column r
            U_try[:, r] = 0
            V_try[:, r] = 0
            W_try[:, r] = 0

            # Check if still valid (unlikely, but check)
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0 if j == k else 0.0
                        prod_sum = np.dot(U_try[i * m + j, :], V_try[j * p + k, :] * W_try[k * n + i, :])
                        if abs(prod_sum - expected) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                # Found rank-84!
                best_rank = 84
                best_solution = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
                break

        # Try merging two columns into one (rank reduction by 1)
        for trial in range(min(20, int((deadline - time.time()) / 0.5))):
            if time.time() > deadline:
                break

            r1 = random.randint(0, 83)
            r2 = random.randint(0, 84)
            if r1 == r2:
                continue

            U_try = U_arr.copy()
            V_try = V_arr.copy()
            W_try = W_arr.copy()

            # Combine r2 into r1, then remove r2
            U_try[:, r1] += U_try[:, r2]
            V_try[:, r1] += V_try[:, r2]
            W_try[:, r1] += W_try[:, r2]

            # Remove column r2 (shift columns after r2 left)
            if r1 < r2:
                U_try[:, r2-1] = 0
                V_try[:, r2-1] = 0
                W_try[:, r2-1] = 0
            else:
                U_try[:, r1-1] = 0
                V_try[:, r1-1] = 0
                W_try[:, r1-1] = 0

            # This is a crude merge; unlikely to work but try
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0 if j == k else 0.0
                        prod_sum = np.dot(U_try[i * m + j, :], V_try[j * p + k, :] * W_try[k * n + i, :])
                        if abs(prod_sum - expected) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                best_rank = 84
                best_solution = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
                break

    return best_solution
