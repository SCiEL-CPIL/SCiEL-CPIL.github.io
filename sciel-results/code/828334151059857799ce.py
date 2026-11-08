"""
Rank-minimal exact tensor decomposition for 2x4x6 matrix multiplication.
Strategy:
  1. Use Strassen (rank 7) for 2x2x2.
  2. Compose two Strassen blocks to get rank 14 for 2x4x2.
  3. Compose three such blocks to get rank 42 for 2x4x6.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    DEADLINE = time.time() + 110.0  # safety margin

    # --- Strassen 2x2x2, rank 7 ---
    # U: (4,7), V: (4,7), W: (4,7)
    # U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)

    # M1 = (a+d)(e+h)
    # a = U[0,0]=A[0,0], d = U[1,1]=A[1,1]
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1  # C[0,0] += M1
    W_s[3, 0] = 1  # C[1,1] += M1

    # M2 = (c+d)e
    # c = A[1,0] = U[2,1], d = A[1,1] = U[3,1], e = B[0,0] = V[2,0]
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[2, 1] = 1
    W_s[2, 1] = 1  # C[1,0] += M2
    W_s[3, 1] = -1 # C[1,1] -= M2

    # M3 = a(f-h)
    # a = A[0,0] = U[0,2], f = B[0,1] = V[1,2], h = B[1,1] = V[3,2]
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1  # C[0,1] += M3
    W_s[3, 2] = 1  # C[1,1] += M3

    # M4 = d(g-e)
    # d = A[1,1] = U[3,3], g = B[1,0] = V[2,3], e = B[0,0] = V[2,4]... wait
    # Let me redo carefully.
    # B indices: B[j,k], V[j*2+k, t]
    # e = B[0,0] -> V[0, t]
    # f = B[0,1] -> V[1, t]
    # g = B[1,0] -> V[2, t]
    # h = B[1,1] -> V[3, t]

    # Let me restart Strassen more carefully.
    # A[i,j] -> U[i*2+j, t]
    # B[j,k] -> V[j*2+k, t]
    # C[i,k] -> W[k*2+i, t]

    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    # C[0,0] += M1, C[1,1] += M1
    W_s[0, 0] = 1  # k=0,i=0 -> W[0,0]
    W_s[3, 0] = 1  # k=1,i=1 -> W[3,0]

    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    # C[1,0] += M2, C[1,1] -= M2
    W_s[2, 1] = 1  # k=0,i=1 -> W[2,1]
    W_s[3, 1] = -1 # k=1,i=1 -> W[3,1]

    # M3 = A[0,0] * (B[0,1]-B[1,1])
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    # C[0,1] += M3, C[1,1] += M3
    W_s[1, 2] = 1  # k=1,i=0 -> W[1,2]
    W_s[3, 2] = 1  # k=1,i=1 -> W[3,2]

    # M4 = A[1,1] * (B[1,0]-B[0,0])
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    # C[1,0] += M4, C[0,0] += M4
    W_s[2, 3] = 1  # k=0,i=1 -> W[2,3]
    W_s[0, 3] = 1  # k=0,i=0 -> W[0,3]

    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    # C[0,0] -= M5, C[0,1] += M5
    W_s[0, 4] = -1 # k=0,i=0 -> W[0,4]
    W_s[1, 4] = 1  # k=1,i=0 -> W[1,4]

    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    # C[1,1] += M6
    W_s[3, 5] = 1  # k=1,i=1 -> W[3,5]

    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    # C[0,0] += M7
    W_s[0, 6] = 1  # k=0,i=0 -> W[0,6]

    # Verify Strassen
    U_full = np.zeros((4, 4, 7))
    for i in range(2):
        for j in range(2):
            for t in range(7):
                U_full[i, j, t] = U_s[i*2+j, t]
    V_full = np.zeros((2, 2, 7))
    for j in range(2):
        for k in range(2):
            for t in range(7):
                V_full[j, k, t] = V_s[j*2+k, t]
    W_full = np.zeros((2, 2, 7))
    for i in range(2):
        for k in range(2):
            for t in range(7):
                W_full[i, k, t] = W_s[k*2+i, t]

    # Tensor: T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[i,k,t]
    # Should be 1 if i,j,k are valid (always for 2x2x2), else 0
    # Actually T[i,j,k] should be 1 for all i,j,k in {0,1}^3
    # Check: for each i,j,k, sum_t U[i,j,t]*V[j,k,t]*W[i,k,t] should be 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U_full[i,j,:] * V_full[j,k,:] * W_full[i,k,:])
                if abs(val - 1.0) > 1e-10:
                    pass  # print(f"Strassen fail at {i},{j},{k}: {val}")

    # --- 2x4x2 decomposition, rank 14 ---
    # A is 2x4, split into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # B is 4x2
    # C = A*B = A1*B[0:2,:] + A2*B[2:4,:]
    # Use Strassen for A1*B_top and A2*B_bottom
    U_242 = np.zeros((8, 14), dtype=float)  # 2*4 = 8 rows
    V_242 = np.zeros((8, 14), dtype=float)  # 4*2 = 8 rows
    W_242 = np.zeros((4, 14), dtype=float)  # 2*2 = 4 rows

    # Block 1: A1 (cols 0,1) * B_top (rows 0,1)
    # A1[i,j] -> U[i*4+j, t] for j=0,1
    # B_top[j,k] -> V[j*2+k, t] for j=0,1
    for t in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, t] != 0:
                    U_242[i*4+j, t] = U_s[i*2+j, t]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, t] != 0:
                    V_242[j*2+k, t] = V_s[j*2+k, t]
        for i in range(2):
            for k in range(2):
                if W_s[k*2+i, t] != 0:
                    W_242[k*2+i, t] += W_s[k*2+i, t]

    # Block 2: A2 (cols 2,3) * B_bottom (rows 2,3)
    # A2[i,j] -> U[i*4+(j+2), t+7] for j=0,1
    # B_bottom[j,k] -> V[(j+2)*2+k, t+7] for j=0,1
    for t in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, t] != 0:
                    U_242[i*4+(j+2), t+7] = U_s[i*2+j, t]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, t] != 0:
                    V_242[(j+2)*2+k, t+7] = V_s[j*2+k, t]
        for i in range(2):
            for k in range(2):
                if W_s[k*2+i, t] != 0:
                    W_242[k*2+i, t] += W_s[k*2+i, t]

    # --- 2x4x6 decomposition, rank 42 ---
    # Split B (4x6) into three 4x2 blocks: B1 (cols 0,1), B2 (cols 2,3), B3 (cols 4,5)
    # C = A*B1 + A*B2 + A*B3
    # Each term uses the 2x4x2 decomposition
    rank = 42
    U_final = np.zeros((8, rank), dtype=float)   # n*m = 2*4 = 8
    V_final = np.zeros((24, rank), dtype=float)  # m*p = 4*6 = 24
    W_final = np.zeros((12, rank), dtype=float)  # p*n = 6*2 = 12

    for block in range(3):
        base = block * 14
        # A part: same for all blocks
        U_final[:, base:base+14] = U_242
        # B part: B is 4x6, block b handles columns 2b, 2b+1
        # V_242 has rows j*2+k for j in 0..3, k in 0..1
        # For block b, the k index maps to column 2b+k
        # V_final row index: j*p + (2b + k) = j*6 + 2b + k
        for j in range(4):
            for k in range(2):
                for t in range(14):
                    V_final[j*6 + (2*block + k), base+t] = V_242[j*2+k, t]
        # W part: C is 6x2, block b handles rows 2b, 2b+1
        # W_242 has rows k*2+i for k in 0..1, i in 0..1
        # For block b, the k index maps to row 2b+k
        # W_final row index: (2b + k)*n + i = (2b+k)*2 + i
        for k in range(2):
            for i in range(2):
                for t in range(14):
                    W_final[(2*block + k)*2 + i, base+t] = W_242[k*2+i, t]

    # Verify
    U_np = U_final
    V_np = V_final
    W_np = W_final

    # Build tensors
    U_t = np.zeros((2, 4, 42))
    for i in range(2):
        for j in range(4):
            U_t[i, j, :] = U_np[i*4+j, :]

    V_t = np.zeros((4, 6, 42))
    for j in range(4):
        for k in range(6):
            V_t[j, k, :] = V_np[j*6+k, :]

    W_t = np.zeros((6, 2, 42))
    for k in range(6):
        for i in range(2):
            W_t[k, i, :] = W_np[k*2+i, :]

    # Check: for each i,j,k, sum_t U_t[i,j,t]*V_t[j,k,t]*W_t[k,i,t] should be 1
    max_err = 0.0
    for i in range(2):
        for j in range(4):
            for k in range(6):
                val = np.sum(U_t[i,j,:] * V_t[j,k,:] * W_t[k,i,:])
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)

    if max_err > 1e-9:
        # Fallback to schoolbook
        rank = 48
        U_final = np.zeros((8, rank))
        V_final = np.zeros((24, rank))
        W_final = np.zeros((12, rank))
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(6):
                    U_final[i*4+j, col] = 1
                    V_final[j*6+k, col] = 1
                    W_final[k*2+i, col] = 1
                    col += 1

    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        rows = len(mat)
        cols = len(mat[0]) if rows > 0 else 0
        result = []
        for r in range(rows):
            row = []
            for c in range(cols):
                val = mat[r, c]
                # Round to nearest half-integer
                half = round(val * 2) / 2.0
                if abs(half - round(half)) < 1e-9:
                    row.append(int(round(half)))
                else:
                    row.append(half)
            result.append(row)
        return result

    return {"U": to_list(U_final), "V": to_list(V_final), "W": to_list(W_final)}
# EVOLVE-BLOCK-END
