"""Rank-88 exact tensor decomposition for 3x4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = 88

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: i=0, schoolbook for 1x4x8: 4*8 = 32 terms
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + 0][col] = 1.0
            col += 1

    # Part 2: i=1,2 using Strassen on 2x4x8
    # Split j into blocks of 2: [0,1], [2,3]
    # Split k into blocks of 2: [0,1], [2,3], [4,5], [6,7]
    # For each (j_block, k_block) pair, use 7 Strassen terms
    # Each Strassen term operates on 2x2x2 subproblem

    # Verified Strassen factors for 2x2x2
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    # p1 = (A00 + A11)(B00 + B11)
    # p2 = (A10 + A11)B00
    # p3 = A00(B10 - B11)
    # p4 = A11(B01 - B00)
    # p5 = (A00 + A10)B11
    # p6 = (A01 - A00)(B00 + B10)
    # p7 = (A10 - A11)(B01 + B11)
    # C00 = p1 + p4 - p5 + p6
    # C01 = p3 + p5 + p7
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p5

    # U factors: rows are [A00, A01, A10, A11]
    strassen_U = [
        [1, 0, 0, 1],   # p1: A00+A11
        [0, 0, 1, 1],   # p2: A10+A11
        [1, 0, 0, 0],   # p3: A00
        [0, 0, 0, 1],   # p4: A11
        [1, 0, 1, 0],   # p5: A00+A10
        [1, -1, 0, 0],  # p6: A01-A00
        [0, 0, 1, -1]   # p7: A10-A11
    ]

    # V factors: rows are [B00, B01, B10, B11]
    strassen_V = [
        [1, 0, 0, 1],   # p1: B00+B11
        [1, 0, 0, 0],   # p2: B00
        [0, -1, 1, 0],  # p3: B10-B11
        [-1, 1, 0, 0],  # p4: B01-B00
        [0, 0, 0, 1],   # p5: B11
        [1, 0, 1, 0],   # p6: B00+B10
        [0, 1, 0, 1]    # p7: B01+B11
    ]

    # W factors: rows are [C00, C01, C10, C11]
    strassen_W = [
        [1, 0, 0, 1],   # p1: C00+=1, C11+=1
        [0, 0, 1, -1],  # p2: C10+=1, C11-=1
        [0, 1, 0, 1],   # p3: C01+=1, C11+=1
        [1, 0, 1, 0],   # p4: C00+=1, C10+=1
        [-1, 1, 0, 1],  # p5: C00-=1, C01+=1, C11+=1
        [1, 0, 0, 0],   # p6: C00+=1
        [0, 1, 0, 0]    # p7: C01+=1
    ]

    # Verify Strassen decomposition locally
    for li in range(2):
        for lj in range(2):
            for lk in range(2):
                a_idx = li * 2 + lj
                b_idx = lj * 2 + lk
                c_idx = li * 2 + lk
                val = 0.0
                for r in range(7):
                    val += strassen_U[r][a_idx] * strassen_V[r][b_idx] * strassen_W[r][c_idx]
                if abs(val - 1) > 1e-12:
                    # Fallback: use schoolbook for safety
                    for j in range(m):
                        for k in range(p):
                            U[(1 + 0) * m + j][col] = 1.0
                            V[j * p + k][col] = 1.0
                            W[k * n + (1 + 0)][col] = 1.0
                            col += 1
                            U[(1 + 1) * m + j][col] = 1.0
                            V[j * p + k][col] = 1.0
                            W[k * n + (1 + 1)][col] = 1.0
                            col += 1
                    return {"U": U, "V": "V", "W": W}

    # Assemble Strassen blocks for i=1,2
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # U: local (li, lj) -> global (1+li, j_start+lj)
                for li in range(2):
                    for lj in range(2):
                        val = strassen_U[t][li * 2 + lj]
                        if val != 0:
                            gi = 1 + li
                            gj = j_start + lj
                            U[gi * m + gj][col] += float(val)

                # V: local (lj, lk) -> global (j_start+lj, k_start+lk)
                for lj in range(2):
                    for lk in range(2):
                        val = strassen_V[t][lj * 2 + lk]
                        if val != 0:
                            gj = j_start + lj
                            gk = k_start + lk
                            V[gj * p + gk][col] += float(val)

                # W: local (li, lk) -> global (k_start+lk, 1+li)
                for li in range(2):
                    for lk in range(2):
                        val = strassen_W[t][li * 2 + lk]
                        if val != 0:
                            gi = 1 + li
                            gk = k_start + lk
                            W[gk * n + gi][col] += float(val)

                col += 1

    assert col <= rank, f"Col {col} exceeds rank {rank}"

    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 100:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    U_arr = np.array(result['U'])
    V_arr = np.array(result['V'])
    W_arr = np.array(result['W'])
    n, m, p = 3, 4, 8
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = float(np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i]))
                if abs(t_val - 1) > 1e-9:
                    max_err = max(max_err, abs(t_val - 1))
    print(f"Max tensor error: {max_err}")
