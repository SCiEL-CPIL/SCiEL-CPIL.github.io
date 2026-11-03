"""Rank-88 exact tensor decomposition for 3x4x8 matrix multiplication."""
import numpy as np
DIMENSIONS = [3, 4, 8]
import time

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
    # Total: 2 * 4 * 7 = 56 terms

    # Strassen factors for 2x2x2:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Local indices: A[i,j] -> idx i*2+j, B[i,j] -> idx i*2+j, C[i,j] -> idx i*2+j
    #
    # Using the standard Strassen formulation:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A01 - A00)(B00 + B10)
    # M7 = (A10 - A11)(B01 + B11)
    #
    # C00 = M1 + M4 - M5 + M6
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 + M3 - M2 + M5
    #
    # Let me verify C00: M1 + M4 - M5 + M6
    # = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A10)B11 + (A01-A00)(B00+B10)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B00 - A00*B11 - A10*B11 + A01*B00 + A01*B10 - A00*B00 - A00*B10
    # = A00*B00 + A11*B11 + A11*B01 - A10*B11 + A01*B00 + A01*B10 - A00*B10
    # Hmm, this doesn't simplify to A00*B00 + A01*B10. Let me recheck.
    #
    # Actually, let me just use the well-known correct Strassen decomposition and verify numerically.
    # I'll use the factors from the inspiration that scored 1.0909 and verify them.

    sU = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1]   # M7: A01 - A11
    ]
    sV = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B10 - B11
        [-1, 0, 1, 0],  # M4: B01 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1]    # M7: B10 + B11
    ]
    sW = [
        [1, 0, 0, 1],   # M1: C00 += 1, C11 += 1
        [0, 0, 1, -1],  # M2: C10 += 1, C11 -= 1
        [0, 1, 0, 1],   # M3: C01 += 1, C11 += 1
        [1, 0, 1, 0],   # M4: C00 += 1, C10 += 1
        [-1, 1, 0, 0],  # M5: C00 -= 1, C01 += 1
        [0, 0, 0, 1],   # M6: C11 += 1
        [1, 0, 0, 0]    # M7: C00 += 1
    ]

    # Verify the Strassen factors produce T[li,lj,lk] = 1 for all li,lj,lk in {0,1}^3
    # T[li,lj,lk] = sum_t sU[t][li*2+lj] * sV[t][lj*2+lk] * sW[t][lk*2+li]
    for li in range(2):
        for lj in range(2):
            for lk in range(2):
                val = 0.0
                for t in range(7):
                    val += sU[t][li * 2 + lj] * sV[t][lj * 2 + lk] * sW[t][lk * 2 + li]
                if val != 1.0:
                    raise ValueError(f"Strassen factor verification failed at ({li},{lj},{lk}): {val}")

    # Now apply the Strassen factors to the 2x4x8 subproblem
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # U factor: for each (li, lj)
                for li in range(2):
                    for lj in range(2):
                        uval = sU[t][li * 2 + lj]
                        if uval != 0:
                            global_i = 1 + li
                            global_j = j_start + lj
                            U[global_i * m + global_j][col] += float(uval)
                # V factor: for each (lj, lk)
                for lj in range(2):
                    for lk in range(2):
                        vval = sV[t][lj * 2 + lk]
                        if vval != 0:
                            global_j = j_start + lj
                            global_k = k_start + lk
                            V[global_j * p + global_k][col] += float(vval)
                # W factor: for each (lk, li)
                for lk in range(2):
                    for li in range(2):
                        wval = sW[t][lk * 2 + li]
                        if wval != 0:
                            global_k = k_start + lk
                            global_i = 1 + li
                            W[global_k * n + global_i][col] += float(wval)
                col += 1

    assert col <= rank, f"Col {col} exceeds rank {rank}"

    # Safety: ensure we haven't exceeded deadline
    elapsed = time.time() - start_time
    if elapsed > 100:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    # Quick verification
    U_arr = np.array(result['U'])
    V_arr = np.array(result['V'])
    W_arr = np.array(result['W'])
    n, m, p = 3, 4, 8
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = np.dot(U_arr[i*m+j], V_arr[j*p+k])
                # This isn't the right check. We need:
                # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
    # Proper check:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0
                for rr in range(len(result['U'][0])):
                    t_val += result['U'][i*m+j][rr] * result['V'][j*p+k][rr] * result['W'][k*n+i][rr]
                if abs(t_val - 1) > 1e-9:
                    max_err = max(max_err, abs(t_val - 1))
    print(f"Max tensor error: {max_err}")
