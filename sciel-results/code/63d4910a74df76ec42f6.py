import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0

    def time_left():
        return deadline - time.time()

    # Schoolbook fallback
    rank_school = n * m * p
    U0 = np.zeros((n*m, rank_school))
    V0 = np.zeros((m*p, rank_school))
    W0 = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1
                V0[j*p + k, col] = 1
                W0[k*n + i, col] = 1
                col += 1
    best = (U0, V0, W0, rank_school)

    # Rank 34 Decomposition
    rank_target = 34
    U = np.zeros((n*m, rank_target))
    V = np.zeros((m*p, rank_target))
    W = np.zeros((p*n, rank_target))
    
    c_idx = 0 # current column index in U, V, W
    
    def add_strassen(U_idx, V_idx, W_idx, col_start):
        # U_idx: [u00, u01, u10, u11]
        # V_idx: [v00, v01, v10, v11]
        # W_idx: [w00, w01, w10, w11]
        # 7 terms
        # M1 = (u00+u11)(v00+v11)
        # M2 = (u10+u11)v00
        # M3 = u00(v10-v00)
        # M4 = u11(v01-v11)
        # M5 = (u00+u01)v11
        # M6 = (u10-u00)(v00+v01)
        # M7 = (u01-u11)(v10+v11)
        
        # W reconstruction:
        # w00 = M1 + M4 - M5 + M7
        # w01 = M3 + M5
        # w10 = M2 + M4
        # w11 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        U[U_idx[0], col_start] += 1
        U[U_idx[3], col_start] += 1
        V[V_idx[0], col_start] += 1
        V[V_idx[3], col_start] += 1
        W[W_idx[0], col_start] += 1
        W[W_idx[3], col_start] += -1
        W[W_idx[1], col_start] += 1
        W[W_idx[2], col_start] += 1
        col_start += 1
        
        # Term 2: M2
        U[U_idx[1], col_start] += 1
        U[U_idx[3], col_start] += 1
        V[V_idx[0], col_start] += 1
        W[W_idx[3], col_start] += 1
        W[W_idx[2], col_start] += 1
        col_start += 1
        
        # Term 3: M3
        U[U_idx[0], col_start] += 1
        V[V_idx[1], col_start] += 1
        V[V_idx[0], col_start] += -1
        W[W_idx[1], col_start] += 1
        W[W_idx[2], col_start] += 1
        col_start += 1
        
        # Term 4: M4
        U[U_idx[3], col_start] += 1
        V[V_idx[0], col_start] += 1
        V[V_idx[3], col_start] += -1
        W[W_idx[0], col_start] += 1
        W[W_idx[2], col_start] += 1
        col_start += 1
        
        # Term 5: M5
        U[U_idx[0], col_start] += 1
        U[U_idx[1], col_start] += 1
        V[V_idx[3], col_start] += 1
        W[W_idx[0], col_start] += -1
        W[W_idx[1], col_start] += 1
        col_start += 1
        
        # Term 6: M6
        U[U_idx[1], col_start] += 1
        U[U_idx[0], col_start] += -1
        V[V_idx[0], col_start] += 1
        V[V_idx[1], col_start] += 1
        W[W_idx[3], col_start] += 1
        col_start += 1
        
        # Term 7: M7
        U[U_idx[1], col_start] += 1
        U[U_idx[3], col_start] += -1
        V[V_idx[1], col_start] += 1
        V[V_idx[3], col_start] += 1
        W[W_idx[0], col_start] += 1
        W[W_idx[2], col_start] += -1
        col_start += 1
        
        return col_start

    def add_simple(u_idx, v_idx, w_idx, col_start):
        U[u_idx, col_start] = 1
        V[v_idx, col_start] = 1
        W[w_idx, col_start] = 1
        return col_start + 1

    # Indices
    # A: i*m+j
    A00, A01, A02 = 0, 1, 2
    A10, A11, A12 = 3, 4, 5
    A20, A21, A22 = 6, 7, 8
    
    # B: j*p+k
    B00, B01, B02, B03 = 0, 1, 2, 3
    B10, B11, B12, B13 = 4, 5, 6, 7
    B20, B21, B22, B23 = 8, 9, 10, 11
    
    # C: k*n+i
    # k=0: C00(0), C10(1), C20(2)
    # k=1: C01(3), C11(4), C21(5)
    # k=2: C02(6), C12(7), C22(8)
    # k=3: C03(9), C13(10), C23(11)
    
    # Block 1: A11 B11 -> C00, C01, C10, C11
    # A11: A00, A01, A10, A11
    # B11: B00, B01, B10, B11
    # C: W[0], W[3], W[1], W[4]
    c_idx = add_strassen([A00, A01, A10, A11], [B00, B01, B10, B11], [0, 3, 1, 4], c_idx)
    
    # Block 2: A11 B12 -> C02, C03, C12, C13
    # A11: A00, A01, A10, A11
    # B12: B02, B03, B12, B13
    # C: W[6], W[9], W[7], W[10]
    c_idx = add_strassen([A00, A01, A10, A11], [B02, B03, B12, B13], [6, 9, 7, 10], c_idx)
    
    # Block 3: A21 B11 -> C20, C21
    # A21: A20, A21
    # B11: B00, B01, B10, B11
    # C20 = A20 B00 + A21 B10
    # C21 = A20 B01 + A21 B11
    c_idx = add_simple(A20, B00, 2, c_idx)
    c_idx = add_simple(A21, B10, 2, c_idx)
    c_idx = add_simple(A20, B01, 5, c_idx)
    c_idx = add_simple(A21, B11, 5, c_idx)
    
    # Block 4: A21 B12 -> C22, C23
    # C22 = A20 B02 + A21 B12
    # C23 = A20 B03 + A21 B13
    c_idx = add_simple(A20, B02, 8, c_idx)
    c_idx = add_simple(A21, B12, 8, c_idx)
    c_idx = add_simple(A20, B03, 11, c_idx)
    c_idx = add_simple(A21, B13, 11, c_idx)
    
    # Block 5: A12 B21 -> C00, C01, C10, C11
    # A12: A02, A12
    # B21: B20, B21
    # C00 += A02 B20
    # C01 += A02 B21
    # C10 += A12 B20
    # C11 += A12 B21
    c_idx = add_simple(A02, B20, 0, c_idx)
    c_idx = add_simple(A02, B21, 3, c_idx)
    c_idx = add_simple(A12, B20, 1, c_idx)
    c_idx = add_simple(A12, B21, 4, c_idx)
    
    # Block 6: A12 B22 -> C02, C03, C12, C13
    # A12: A02, A12
    # B22: B22, B23
    # C02 += A02 B22
    # C03 += A02 B23
    # C12 += A12 B22
    # C13 += A12 B23
    c_idx = add_simple(A02, B22, 6, c_idx)
    c_idx = add_simple(A02, B23, 9, c_idx)
    c_idx = add_simple(A12, B22, 7, c_idx)
    c_idx = add_simple(A12, B23, 10, c_idx)
    
    # Block 7: A22 B21 -> C20, C21
    # A22: A22
    # B21: B20, B21
    # C20 += A22 B20
    # C21 += A22 B21
    c_idx = add_simple(A22, B20, 2, c_idx)
    c_idx = add_simple(A22, B21, 5, c_idx)
    
    # Block 8: A22 B22 -> C22, C23
    # A22: A22
    # B22: B22, B23
    # C22 += A22 B22
    # C23 += A22 B23
    c_idx = add_simple(A22, B22, 8, c_idx)
    c_idx = add_simple(A22, B23, 11, c_idx)
    
    if c_idx == rank_target:
        # Verify
        U_check = U
        V_check = V
        W_check = W
        # Simple check: compute tensor and compare to all-ones?
        # No, compare to schoolbook tensor.
        T = np.zeros((n, m, p))
        for c in range(rank_target):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U_check[i*m+j, c] * V_check[j*p+k, c] * W_check[k*n+i, c]
        
        Tref = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Tref[i, j, k] = 1.0
        
        if np.allclose(T, Tref, atol=1e-9):
            best = (U, V, W, rank_target)
        else:
            # Fallback to schoolbook if verification fails
            pass
    
    Ub, Vb, Wb, rb = best
    U_final = [[int(Ub[i, c]) if abs(Ub[i, c] - round(Ub[i, c])) < 1e-9 else (Ub[i, c] * 2) for c in range(rb)] for i in range(Ub.shape[0])]
    V_final = [[int(Vb[i, c]) if abs(Vb[i, c] - round(Vb[i, c])) < 1e-9 else (Vb[i, c] * 2) for c in range(rb)] for i in range(Vb.shape[0])]
    W_final = [[int(Wb[i, c]) if abs(Wb[i, c] - round(Wb[i, c])) < 1e-9 else (Wb[i, c] * 2) for c in range(rb)] for i in range(Wb.shape[0])]
    
    def clean(x):
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return float(x)
    
    U_final = [[clean(Ub[i, c]) for c in range(rb)] for i in range(Ub.shape[0])]
    V_final = [[clean(Vb[i, c]) for c in range(rb)] for i in range(Vb.shape[0])]
    W_final = [[clean(Wb[i, c]) for c in range(rb)] for i in range(Wb.shape[0])]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
