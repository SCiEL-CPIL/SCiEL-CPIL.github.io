import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 120.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    rank = 55
    U = np.zeros((n * m, rank))  # 6 x 55
    V = np.zeros((m * p, rank))  # 30 x 55
    W = np.zeros((p * n, rank))  # 20 x 55

    idx = 0

    def deadline_exceeded():
        return (time.time() - start_time) > DEADLINE - 5.0

    # Strassen for 2x2 x 2x2 blocks
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices for block b: B00=2b, B01=2b+1, B10=10+2b, B11=10+2b+1
    # C indices for block b: C00=4b, C01=4b+2, C10=4b+1, C11=4b+3

    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    # M2 = (A10+A11)B00 -> C10 - C11
    # M3 = A00(B01-B11) -> C01 + C11
    # M4 = A11(B10-B00) -> C00 + C10
    # M5 = (A00+A01)B11 -> -C00 + C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00

    strassen_U = {
        0: [(0, 1), (4, 1)],
        1: [(3, 1), (4, 1)],
        2: [(0, 1)],
        3: [(4, 1)],
        4: [(0, 1), (1, 1)],
        5: [(3, 1), (0, -1)],
        6: [(1, 1), (4, -1)],
    }

    strassen_V = {
        0: [(0, 1), (3, 1)],
        1: [(0, 1)],
        2: [(1, 1), (3, -1)],
        3: [(2, 1), (0, -1)],
        4: [(3, 1)],
        5: [(0, 1), (1, 1)],
        6: [(2, 1), (3, 1)],
    }

    strassen_W = {
        0: [(0, 1), (3, 1)],
        1: [(1, 1), (3, -1)],
        2: [(2, 1), (3, 1)],
        3: [(0, 1), (1, 1)],
        4: [(0, -1), (2, 1)],
        5: [(3, 1)],
        6: [(0, 1)],
    }

    for b in range(5):
        if deadline_exceeded():
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        col0 = 2 * b
        col1 = 2 * b + 1
        v_offsets = {0: col0, 1: col1, 2: 10 + col0, 3: 10 + col1}
        w_offsets = {0: 4 * b, 1: 4 * b + 1, 2: 4 * b + 2, 3: 4 * b + 3}

        for s in range(7):
            for (u_idx, coeff) in strassen_U[s]:
                U[u_idx, idx] += coeff
            for (v_off, coeff) in strassen_V[s]:
                V[v_offsets[v_off], idx] += coeff
            for (w_off, coeff) in strassen_W[s]:
                W[w_offsets[w_off], idx] += coeff
            idx += 1

    # Part 2: Schoolbook for A[*,2] * B[2,*]
    for i in range(n):
        if deadline_exceeded():
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        for k in range(p):
            U[i * m + 2, idx] = 1
            V[2 * p + k, idx] = 1
            W[k * n + i, idx] = 1
            idx += 1

    assert idx == rank, f"Expected rank {rank}, got {idx}"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
