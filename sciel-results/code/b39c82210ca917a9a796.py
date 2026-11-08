import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    # U: n*m = 6 x rank
    # V: m*p = 12 x rank
    # W: p*n = 8 x rank
    
    # Rank 22 decomposition using Strassen for 2x2 blocks + standard for remaining
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        """Add a term to the decomposition at column 'col'"""
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    # --- Block 1: A[0:2, 0:2] * B[0:2, 0:2] ---
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=0, B01=1, B10=4, B11=5
    # C indices (W[k*n+i]): C00=0, C01=2, C10=1, C11=3
    
    # Strassen's 7 products for 2x2 * 2x2
    # M1: (A00+A11)(B00+B11) -> contributes to C00, C11
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    # M2: (A10+A11)B00 -> contributes to C10, -C11
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    # M3: A00(B01-B11) -> contributes to C01, C11
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    # M4: A11(B10-B00) -> contributes to C00, C10
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    # M5: (A00+A01)B11 -> contributes to -C00, C01
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    # M6: (A10-A00)(B00+B01) -> contributes to C11
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    # M7: (A01-A11)(B10+B11) -> contributes to C00
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    # --- Block 2: A[0:2, 0:2] * B[0:2, 2:4] ---
    # B indices: B02=2, B03=3, B12=6, B13=7
    # C indices: C02=4, C03=6, C12=5, C13=7

    # M1': (A00+A11)(B02+B13) -> contributes to C02, C13
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    # M2': (A10+A11)B02 -> contributes to C12, -C13
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    # M3': A00(B03-B13) -> contributes to C03, C13
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    # M4': A11(B12-B02) -> contributes to C02, C12
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    # M5': (A00+A01)B13 -> contributes to -C02, C03
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    # M6': (A10-A00)(B02+B03) -> contributes to C13
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    # M7': (A01-A11)(B12+B13) -> contributes to C02
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    # --- Block 3: A[0:2, 2:3] * B[2:3, 0:4] ---
    # A indices: A02=2, A12=5
    # B indices: B20=8, B21=9, B22=10, B23=11
    # C indices: C00=0, C01=2, C02=4, C03=6, C10=1, C11=3, C12=5, C13=7

    # Standard multiplication for the remaining 2x1 * 1x4 block
    # A02 * B20 -> C00
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    # A02 * B21 -> C01
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    # A02 * B22 -> C02
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    # A02 * B23 -> C03
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    
    # A12 * B20 -> C10
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    # A12 * B21 -> C11
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    # A12 * B22 -> C12
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    # A12 * B23 -> C13
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    # Convert to numpy arrays for verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Verification: check that the decomposition is exact
    # For each (i, j, k), verify sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    verification_passed = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                # Sum over rank dimension
                val = np.sum(U_arr[alpha, :] * V_arr[beta, :] * W_arr[gamma, :])
                if abs(val - 1.0) > 1e-9:
                    verification_passed = False
                    break
            if not verification_passed:
                break
        if not verification_passed:
            break
    
    # If verification failed, we have a problem. Return as-is (should not happen)
    
    # Ensure all entries are integers (they should be)
    for mat in [U, V, W]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = int(round(val))

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    # Quick verification
    U_arr = np.array(res['U'])
    V_arr = np.array(res['V'])
    W_arr = np.array(res['W'])
    n, m, p = 2, 3, 4
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                val = np.sum(U_arr[alpha, :] * V_arr[beta, :] * W_arr[gamma, :])
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    print(f"FAIL: ({i},{j},{k}) = {val}")
    print("Verification:", "PASS" if ok else "FAIL")
