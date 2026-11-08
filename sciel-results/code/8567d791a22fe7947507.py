import numpy as np
import time

start_time = time.time()
TIME_LIMIT = 115.0

def construct():
    n, m, p = 2, 3, 9
    
    # We'll build U, V, W as lists of lists, then determine rank
    # Part A: columns 0,1 -> 2x2x2 Strassen (rank 7)
    # Part B: columns 2,3,4 -> 2x3x3 naive (rank 18)
    # Part C: columns 5,6,7 -> 2x3x3 naive (rank 18)
    # Part D: column 8 -> 2x3x1 naive (rank 6)
    # Total rank: 7 + 18 + 18 + 6 = 49
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 27 rows
    W = [[0] * rank for _ in range(p * n)]  # 18 rows
    
    col_idx = 0
    
    # Part A: columns 0,1 - Strassen for 2x2x2
    k0, k1 = 0, 1
    u_a, u_b, u_c, u_d = 0, 1, 3, 4  # A[0,0], A[0,1], A[1,0], A[1,1]
    v_e, v_f = 0, 1  # B[0,0], B[0,1]
    v_g, v_h = 3, 4  # B[1,0], B[1,1]
    
    # M1 = (a+d)(e+h)
    U[u_a][col_idx] += 1; U[u_d][col_idx] += 1
    V[v_e][col_idx] += 1; V[v_h][col_idx] += 1
    W[k0*n+0][col_idx] += 1
    W[k1*n+1][col_idx] += 1
    col_idx += 1
    
    # M2 = (c+d)e
    U[u_c][col_idx] += 1; U[u_d][col_idx] += 1
    V[v_e][col_idx] += 1
    W[k0*n+1][col_idx] += 1
    col_idx += 1
    
    # M3 = a(f-h)
    U[u_a][col_idx] += 1
    V[v_f][col_idx] += 1; V[v_h][col_idx] -= 1
    W[k1*n+0][col_idx] += 1
    col_idx += 1
    
    # M4 = d(g-e)
    U[u_d][col_idx] += 1
    V[v_g][col_idx] += 1; V[v_e][col_idx] -= 1
    W[k0*n+1][col_idx] += 1
    col_idx += 1
    
    # M5 = (a+b)h
    U[u_a][col_idx] += 1; U[u_b][col_idx] += 1
    V[v_h][col_idx] += 1
    W[k1*n+0][col_idx] += 1
    col_idx += 1
    
    # M6 = (c-a)(e+f)
    U[u_c][col_idx] += 1; U[u_a][col_idx] -= 1
    V[v_e][col_idx] += 1; V[v_f][col_idx] += 1
    W[k1*n+1][col_idx] += 1
    col_idx += 1
    
    # M7 = (b-d)(g+h)
    U[u_b][col_idx] += 1; U[u_d][col_idx] -= 1
    V[v_g][col_idx] += 1; V[v_h][col_idx] += 1
    W[k1*n+1][col_idx] += 1
    col_idx += 1
    
    # Part B: columns 2,3,4 - naive 2x3x3
    for k in range(2, 5):
        for i in range(2):  # row of A / row of C
            for j in range(3):  # col of A / row of B
                U[i*3+j][col_idx] += 1
                V[j*9+k][col_idx] += 1
                W[k*2+i][col_idx] += 1
                col_idx += 1
    
    # Part C: columns 5,6,7 - naive 2x3x3
    for k in range(5, 8):
        for i in range(2):
            for j in range(3):
                U[i*3+j][col_idx] += 1
                V[j*9+k][col_idx] += 1
                W[k*2+i][col_idx] += 1
                col_idx += 1
    
    # Part D: column 8 - naive 2x3x1
    for i in range(2):
        for j in range(3):
            U[i*3+j][col_idx] += 1
            V[j*9+8][col_idx] += 1
            W[8*2+i][col_idx] += 1
            col_idx += 1
    
    assert col_idx == rank, f"col_idx={col_idx}, rank={rank}"
    
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

