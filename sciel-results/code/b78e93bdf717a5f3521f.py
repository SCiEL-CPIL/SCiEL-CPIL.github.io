import numpy as np
import time

start_time = time.time()
TIME_LIMIT = 110.0  # Safety margin before 120s limit


def construct():
    n, m, p = 2, 3, 9
    R = 50
    U = [[0] * R for _ in range(n * m)]  # 6 rows
    V = [[0] * R for _ in range(m * p)]  # 27 rows
    W = [[0] * R for _ in range(p * n)]  # 18 rows

    col = 0
    # ---- Part 1: 2x2 block (A cols 0,1) times B rows 0,1 ----
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        ve, vf = k0, k1
        vg, vh = p + k0, p + k1

        u_a, u_b, u_c, u_d = 0, 1, 3, 4

        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[ve][col] += 1
        V[vh][col] += 1
        W[9 * k0 + 0][col] += 1
        W[9 * k0 + 1][col] += 1
        W[9 * k1 + 1][col] += 1

        # M2 = (c+d)e
        U[u_c][col + 1] += 1
        U[u_d][col + 1] += 1
        V[ve][col + 1] += 1
        W[9 * k0 + 1][col + 1] += 1
        W[9 * k1 + 1][col + 1] -= 1

        # M3 = a(f-h)
        U[u_a][col + 2] += 1
        V[vf][col + 2] += 1
        V[vh][col + 2] -= 1
        W[9 * k1 + 0][col + 2] += 1
        W[9 * k1 + 1][col + 2] += 1

        # M4 = d(g-e)
        U[u_d][col + 3] += 1
        V[vg][col + 3] += 1
        V[ve][col + 3] -= 1
        W[9 * k0 + 0][col + 3] += 1
        W[9 * k0 + 1][col + 3] += 1
        W[9 * k1 + 1][col + 3] += 1

        # M5 = (a+b)h
        U[u_a][col + 4] += 1
        U[u_b][col + 4] += 1
        V[vh][col + 4] += 1
        W[9 * k0 + 0][col + 4] -= 1
        W[9 * k1 + 0][col + 4] += 1
        W[9 * k1 + 1][col + 4] += 1

        # M6 = (c-a)(e+f)
        U[u_c][col + 5] += 1
        U[u_a][col + 5] -= 1
        V[ve][col + 5] += 1
        V[vf][col + 5] += 1
        W[9 * k1 + 1][col + 5] += 1

        # M7 = (b-d)(g+h)
        U[u_b][col + 6] += 1
        U[u_d][col + 6] -= 1
        V[vg][col + 6] += 1
        V[vh][col + 6] += 1
        W[9 * k0 + 0][col + 6] += 1

        col += 7

    # ---- Handle remaining column k=8 (only row 0,1 of B) ----
    k8 = 8
    U[0][col] += 1
    V[k8][col] += 1
    W[9 * k8 + 0][col] += 1

    U[1][col + 1] += 1
    V[p + k8][col + 1] += 1
    W[9 * k8 + 0][col + 1] += 1

    U[3][col + 2] += 1
    V[k8][col + 2] += 1
    W[9 * k8 + 1][col + 2] += 1

    U[4][col + 3] += 1
    V[p + k8][col + 3] += 1
    W[9 * k8 + 1][col + 3] += 1
    col += 4

    # ---- Part 2: A col 2 (scalar) times all 9 columns of B row 2 ----
    for k in range(9):
        U[2][col] += 1
        V[2 * p + k][col] += 1
        W[9 * k + 0][col] += 1

        U[5][col + 1] += 1
        V[2 * p + k][col + 1] += 1
        W[9 * k + 1][col + 1] += 1

        col += 2

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    n, m, p = 2, 3, 9
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B

    A_flat = A.flatten()
    B_flat = B.flatten()
    C_flat = np.zeros(n * p)
    U_arr = np.array(res["U"])
    V_arr = np.array(res["V"])
    W_arr = np.array(res["W"])

    for s in range(U_arr.shape[1]):
        val = np.dot(U_arr[:, s], A_flat) * np.dot(V_arr[:, s], B_flat)
        C_flat += val * W_arr[:, s]

    C_actual = C_flat.reshape((p, n)).T
    if np.allclose(C_actual, C_expected, atol=1e-6):
        print("Validation Passed")
    else:
        print("Mismatch detected")
