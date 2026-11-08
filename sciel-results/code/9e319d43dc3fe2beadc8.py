import numpy as np
import time

start_time = time.time()
TIME_LIMIT = 118.0


def construct():
    n, m, p = 2, 3, 9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        ve = k0
        vf = k1
        vg = p + k0
        vh = p + k1
        wa, wb, wc, wd = 0, 1, 3, 4
        w00 = k0 * n
        w01 = k1 * n
        w10 = k0 * n + 1
        w11 = k1 * n + 1

        # M1 = (a+d)(e+h)
        U[wa][col] += 1
        U[wd][col] += 1
        V[ve][col] += 1
        V[vh][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1

        # M2 = (c+d)e
        U[wc][col+1] += 1
        U[wd][col+1] += 1
        V[ve][col+1] += 1
        W[w10][col+1] += 1

        # M3 = a(f-h)
        U[wa][col+2] += 1
        V[vf][col+2] += 1
        V[vh][col+2] -= 1
        W[w01][col+2] += 1
        W[w11][col+2] += 1

        # M4 = d(g-e)
        U[wd][col+3] += 1
        V[vg][col+3] += 1
        V[ve][col+3] -= 1
        W[w00][col+3] += 1
        W[w10][col+3] += 1

        # M5 = (a+b)h
        U[wa][col+4] += 1
        U[wb][col+4] += 1
        V[vh][col+4] += 1
        W[w00][col+4] -= 1
        W[w01][col+4] += 1

        # M6 = (c-a)(e+f)
        U[wc][col+5] += 1
        U[wa][col+5] -= 1
        V[ve][col+5] += 1
        V[vf][col+5] += 1
        W[w11][col+5] += 1

        # M7 = (b-d)(g+h)
        U[wb][col+6] += 1
        U[wd][col+6] -= 1
        V[vg][col+6] += 1
        V[vh][col+6] += 1
        W[w00][col+6] += 1

        col += 7

    k8 = 8
    ve8 = k8
    vg8 = p + k8
    wa, wb, wc, wd = 0, 1, 3, 4
    w08 = k8 * n
    w18 = k8 * n + 1

    U[wa][col] += 1
    V[ve8][col] += 1
    W[w08][col] += 1

    U[wb][col+1] += 1
    V[vg8][col+1] += 1
    W[w08][col+1] += 1

    U[wc][col+2] += 1
    V[ve8][col+2] += 1
    W[w18][col+2] += 1

    U[wd][col+3] += 1
    V[vg8][col+3] += 1
    W[w18][col+3] += 1

    col += 4

    for k in range(9):
        vk = 2 * p + k
        wk = k * n
        U[2][col] += 1
        V[vk][col] += 1
        W[wk][col] += 1
        U[5][col+1] += 1
        V[vk][col+1] += 1
        W[wk+1][col+1] += 1
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
    for s in range(len(U_arr[0])):
        val = np.dot(U_arr[:, s], A_flat) * np.dot(V_arr[:, s], B_flat)
        C_flat += val * W_arr[:, s]
    C_actual = C_flat.reshape((p, n)).T
    if not np.allclose(C_actual, C_expected, atol=1e-6):
        print("Mismatch detected")
    else:
        print("Validation Passed")
