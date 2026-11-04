import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 140.0
    
    n, m, p = DIMENSIONS
    
    def get_strassen_2x2x2():
        # Returns U (4x7), V (4x7), W (4x7) for 2x2x2 Strassen
        # U indices: x00, x01, x10, x11
        # V indices: y00, y01, y10, y11
        # W indices: z00, z01, z10, z11 (k*n+i, n=2)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (x00 + x11)(y00 + y11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # z00 += M1, z11 += M1
        
        # M2 = (x10 + x11)y00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # z10 += M2, z11 -= M2
        
        # M3 = x00(y10 + y11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = 1
        W[1, 2] = 1; W[3, 2] = 1 # z01 += M3, z11 += M3
        
        # M4 = x11y11
        U[3, 3] = 1
        V[3, 3] = 1
        W[0, 3] = 1; W[2, 3] = 1 # z00 += M4, z10 += M4
        
        # M5 = (x00 + x01)y11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # z00 -= M5, z01 += M5
        
        # M6 = (x10 - x00)(y00 + y01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # z11 += M6
        
        # M7 = (x01 - x11)(y10 + y11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # z00 += M7
        
        return U, V, W

    def get_schoolbook_2x2x1():
        # Returns U (4x4), V (2x4), W (2x4) for 2x2x1 Schoolbook
        # U indices: x00, x01, x10, x11
        # V indices: y00, y10 (y is 2x1)
        # W indices: z00, z10 (z is 2x1, k*n+i, n=2, k=0) -> 0, 1
        
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # z00 = x00y00 + x01y10
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        # z10 = x10y00 + x11y10
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    def construct_2x2x5():
        # Constructs U (4x18), V (10x18), W (10x18) for 2x2x5
        # U: x (2x2)
        # V: y (2x5)
        # W: z (5x2)
        
        U_total = np.zeros((4, 18))
        V_total = np.zeros((10, 18))
        W_total = np.zeros((10, 18))
        
        # Part 1: y cols 0,1 (2x2) -> Strassen
        U1, V1, W1 = get_strassen_2x2x2()
        U_total[:, 0:7] = U1
        
        # Map V1 (4x7) to V_total (10x7)
        # V1 rows: y00, y01, y10, y11
        # V_total rows: y00(0), y01(1), y10(5), y11(6), y02(2)...
        # Indices in V_total for y00, y01, y10, y11:
        # y00: 0*5+0=0
        # y01: 0*5+1=1
        # y10: 1*5+0=5
        # y11: 1*5+1=6
        V_total[0, 0:7] = V1[0]
        V_total[1, 0:7] = V1[1]
        V_total[5, 0:7] = V1[2]
        V_total[6, 0:7] = V1[3]
        
        # Map W1 (4x7) to W_total (10x7)
        # W1 rows: z00, z01, z10, z11
        # W_total rows: z00(0), z01(1), z10(2), z11(3), z20(4)...
        # Indices in W_total for z00, z01, z10, z11:
        # z00: 0*2+0=0
        # z01: 0*2+1=1
        # z10: 1*2+0=2
        # z11: 1*2+1=3
        W_total[0, 0:7] = W1[0]
        W_total[1, 0:7] = W1[1]
        W_total[2, 0:7] = W1[2]
        W_total[3, 0:7] = W1[3]
        
        # Part 2: y cols 2,3 (2x2) -> Strassen
        U2, V2, W2 = get_strassen_2x2x2()
        U_total[:, 7:14] = U2
        
        # Map V2 to V_total
        # y02: 0*5+2=2
        # y03: 0*5+3=3
        # y12: 1*5+2=7
        # y13: 1*5+3=8
        V_total[2, 7:14] = V2[0]
        V_total[3, 7:14] = V2[1]
        V_total[7, 7:14] = V2[2]
        V_total[8, 7:14] = V2[3]
        
        # Map W2 to W_total
        # z20: 2*2+0=4
        # z21: 2*2+1=5
        # z30: 3*2+0=6
        # z31: 3*2+1=7
        W_total[4, 7:14] = W2[0]
        W_total[5, 7:14] = W2[1]
        W_total[6, 7:14] = W2[2]
        W_total[7, 7:14] = W2[3]
        
        # Part 3: y col 4 (2x1) -> Schoolbook
        U3, V3, W3 = get_schoolbook_2x2x1()
        U_total[:, 14:18] = U3
        
        # Map V3 to V_total
        # y04: 0*5+4=4
        # y14: 1*5+4=9
        V_total[4, 14:18] = V3[0]
        V_total[9, 14:18] = V3[1]
        
        # Map W3 to W_total
        # z40: 4*2+0=8
        # z41: 4*2+1=9
        W_total[8, 14:18] = W3[0]
        W_total[9, 14:18] = W3[1]
        
        return U_total, V_total, W_total

    # Construct full 2x4x5
    # U (8x36), V (20x36), W (10x36)
    
    U_full = np.zeros((8, 36))
    V_full = np.zeros((20, 36))
    W_full = np.zeros((10, 36))
    
    # Block 1: A_L (2x2) * B_L (2x5) -> C_L (5x2)
    U1, V1, W1 = construct_2x2x5()
    
    # Map U1 (4x18) to U_full (8x18)
    # U1 rows: x00, x01, x10, x11
    # U_full rows for A_L (cols 0,1):
    # x00: 0*4+0=0
    # x01: 0*4+1=1
    # x10: 1*4+0=4
    # x11: 1*4+1=5
    U_full[0, 0:18] = U1[0]
    U_full[1, 0:18] = U1[1]
    U_full[4, 0:18] = U1[2]
    U_full[5, 0:18] = U1[3]
    
    # Map V1 (10x18) to V_full (20x18)
    # V1 rows: y00..y14 (indices 0..9)
    # V_full rows for B_L (rows 0,1):
    # y00: 0*5+0=0
    # ...
    # y14: 1*5+4=9
    # So V_full[0:10, 0:18] = V1
    V_full[0:10, 0:18] = V1
    
    # Map W1 (10x18) to W_full (10x18)
    # W1 rows: z00..z41 (indices 0..9)
    # W_full rows for C_L (rows 0,1):
    # z00: 0*2+0=0
    # ...
    # z41: 4*2+1=9
    # So W_full[0:10, 0:18] = W1
    W_full[0:10, 0:18] = W1
    
    # Block 2: A_R (2x2) * B_R (2x5) -> C_R (5x2)
    U2, V2, W2 = construct_2x2x5()
    
    # Map U2 to U_full
    # U2 rows: x00, x01, x10, x11
    # U_full rows for A_R (cols 2,3):
    # x00: 0*4+2=2
    # x01: 0*4+3=3
    # x10: 1*4+2=6
    # x11: 1*4+3=7
    U_full[2, 18:36] = U2[0]
    U_full[3, 18:36] = U2[1]
    U_full[6, 18:36] = U2[2]
    U_full[7, 18:36] = U2[3]
    
    # Map V2 to V_full
    # V2 rows: y00..y14 (indices 0..9)
    # V_full rows for B_R (rows 2,3):
    # y00: 2*5+0=10
    # ...
    # y14: 3*5+4=19
    # So V_full[10:20, 18:36] = V2
    V_full[10:20, 18:36] = V2
    
    # Map W2 to W_full
    # W2 rows: z00..z41 (indices 0..9)
    # W_full rows for C_R (rows 0,1) - same as C_L
    # So W_full[0:10, 18:36] = W2
    W_full[0:10, 18:36] = W2
    
    # Verify
    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)
        
        # Check off-diagonals
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for i_prime in range(n):
                            for k_prime in range(p):
                                if i == i_prime and k == k_prime:
                                    continue
                                s = 0.0
                                for r in range(rank):
                                    s += U_3d[i, j, r] * V_3d[j_prime, k, r] * W_3d[k_prime, i_prime, r]
                                if abs(s) > max_err:
                                    max_err = abs(s)
        
        return max_err < tol

    if verify(U_full.tolist(), V_full.tolist(), W_full.tolist()):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
    # Fallback to schoolbook if verification fails (should not happen)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
