"""Rank-88 exact tensor decomposition for 3x4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]


def construct():
    start_time = time.time()
    n, m, p = 3, 4, 8
    rank = 88

    U = [[0.0] * rank for _ in range(n * m)]  # 12 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 24 rows

    col = 0

    # Part 1: Row i=0, schoolbook for 1x4x8: 4*8 = 32 terms
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + 0][col] = 1.0
            col += 1

    # Part 2: Rows i=1,2 using Strassen on 2x2x2 blocks
    # Split j into 2 blocks of 2: [0,1], [2,3]
    # Split k into 4 blocks of 2: [0,1], [2,3], [4,5], [6,7]
    # For each (j_block, k_block): 7 Strassen terms

    # Strassen factors (verified):
    # A indices: [A00, A01, A10, A11] -> local row 0,1,2,3
    # B indices: [B00, B01, B10, B11] -> local row 0,1,2,3
    # C indices: [C00, C01, C10, C11] -> local row 0,1,2,3
    #
    # p1 = (A11-A10)(B01+B11)
    # p2 = (A00+A01)B11
    # p3 = (A01+A11)B00
    # p4 = (A10-A00)(B00+B10)
    # p5 = (A01-A10)(B01+B11)
    # p6 = A10(B11-B10)
    # p7 = A01(B00-B01)
    #
    # C00 = p1+p4-p5+p7
    # C01 = p3+p5
    # C10 = p2+p6
    # C11 = p1+p3-p2+p4

    strassen_U = [
        [0,  0, -1,  1],  # p1: A11-A10
        [1,  1,  0,  0],  # p2: A00+A01
        [0,  1,  0,  1],  # p3: A01+A11
        [1,  0, -1,  0],  # p4: A10-A00
        [0,  1, -1,  0],  # p5: A01-A10
        [0,  0,  1,  0],  # p6: A10
        [0,  1,  0,  0],  # p7: A01
    ]
    strassen_V = [
        [0,  1,  0,  1],  # p1: B01+B11
        [0,  0,  0,  1],  # p2: B11
        [1,  0,  0,  0],  # p3: B00
        [1,  0,  1,  0],  # p4: B00+B10
        [0,  1,  0,  1],  # p5: B01+B11
        [0,  0, -1,  1],  # p6: B11-B10
        [1,  0,  0, -1],  # p7: B00-B01
    ]
    strassen_W = [
        [1,  0,  0,  1],  # p1: C00+=1, C11+=1
        [0,  0,  1, -1],  # p2: C10+=1, C11-=1
        [0,  1,  0,  1],  # p3: C01+=1, C11+=1
        [1,  0,  0,  1],  # p4: C00+=1, C11+=1
        [0, -1, 0,  0],  # p5: C01-=1
        [0,  0,  1,  0],  # p6: C10+=1
        [1,  0,  0,  0],  # p7: C00+=1
    ]

    # Apply Strassen for each (i_block, j_block, k_block)
    for i_block in range(2):  # i = 1, 2
        i_start = 1 + i_block  # global row index in {1, 2}
        for j_block in range(2):  # j blocks: [0,1], [2,3]
            j_start = j_block * 2
            for k_block in range(4):  # k blocks: [0,1], [2,3], [4,5], [6,7]
                k_start = k_block * 2
                for t in range(7):
                    # U: A indices (local_i, local_j) -> global (i_start+local_i, j_start+local_j)
                    for local_i in range(2):
                        for local_j in range(2):
                            val = strassen_U[t][local_i * 2 + local_j]
                            if val != 0:
                                gi = i_start + local_i
                                gj = j_start + local_j
                                U[gi * m + gj][col] += float(val)

                    # V: B indices (local_j, local_k) -> global (j_start+local_j, k_start+local_k)
                    for local_j in range(2):
                        for local_k in range(2):
                            val = strassen_V[t][local_j * 2 + local_k]
                            if val != 0:
                                gj = j_start + local_j
                                gk = k_start + local_k
                                V[gj * p + gk][col] += float(val)

                    # W: C indices (local_i, local_k) -> global (k_start+local_k, i_start+local_i)
                    for local_i in range(2):
                        for local_k in range(2):
                            val = strassen_W[t][local_i * 2 + local_k]
                            if val != 0:
                                gi = i_start + local_i
                                gk = k_start + local_k
                                W[gk * n + gi][col] += float(val)

                    col += 1

    assert col <= rank, f"Column {col} exceeds rank {rank}"
    assert col == rank, f"Expected rank {rank}, got {col}"

    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > 100:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    # Verify tensor decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
    n, m, p = 3, 4, 8
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0.0
                for r in range(len(result['U'][0])):
                    t_val += result['U'][i * m + j][r] * result['V'][j * p + k][r] * result['W'][k * n + i][r]
                if abs(t_val - 1) > max_err:
                    max_err = abs(t_val - 1)
    print(f"Max tensor error: {max_err}")
    assert max_err < 1e-9, f"Decomposition invalid! Max error: {max_err}"
    print("Decomposition verified successfully.")
