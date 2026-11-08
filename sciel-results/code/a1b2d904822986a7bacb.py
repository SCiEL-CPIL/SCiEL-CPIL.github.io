import numpy as np
import time

start_time = time.time()
TIME_LIMIT = 118.0  # Safety margin before 120s limit


def construct():
    n, m, p = 2, 3, 9

    # Rank 50 decomposition
    # This is the sum of:
    # 1. Strassen-based decomposition for A[:, 0:2] * B[0:2, :] (2x2 * 2x9)
    #    Split into 4 pairs of columns (rank 7 each) + 1 single column (rank 4)
    #    Total rank = 4*7 + 4 = 32
    # 2. Direct multiplication for A[:, 2:3] * B[2, :] (2x1 * 1x9)
    #    Total rank = 2*9 = 18
    # Total rank = 32 + 18 = 50

    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 27 rows
    W = [[0] * rank for _ in range(p * n)]  # 18 rows

    col_idx = 0
    
    # Part 1: A[:, 0:2] * B[0:2, :]
    # Process columns of B in pairs for Strassen
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Indices in V (B matrix)
        # B is 3x9. We are using rows 0 and 1.
        # V index = row * p + col
        v_e = 0 * p + k0  # B[0, k0]
        v_f = 0 * p + k1  # B[0, k1]
        v_g = 1 * p + k0  # B[1, k0]
        v_h = 1 * p + k1  # B[1, k1]

        # Indices in U (A matrix)
        # A is 2x3. We are using cols 0 and 1.
        # U index = row * m + col
        u_a = 0 * m + 0  # A[0, 0]
        u_b = 0 * m + 1  # A[0, 1]
        u_c = 1 * m + 0  # A[1, 0]
        u_d = 1 * m + 1  # A[1, 1]

        # Strassen's 7 multiplications
        # M1 = (a+d)(e+h)
        U[u_a][col_idx] += 1
        U[u_d][col_idx] += 1
        V[v_e][col_idx] += 1
        V[v_h][col_idx] += 1

        # M2 = (c+d)e
        U[u_c][col_idx+1] += 1
        U[u_d][col_idx+1] += 1
        V[v_e][col_idx+1] += 1

        # M3 = a(f-h)
        U[u_a][col_idx+2] += 1
        V[v_f][col_idx+2] += 1
        V[v_h][col_idx+2] -= 1

        # M4 = d(g-e)
        U[u_d][col_idx+3] += 1
        V[v_g][col_idx+3] += 1
        V[v_e][col_idx+3] -= 1

        # M5 = (a+b)h
        U[u_a][col_idx+4] += 1
        U[u_b][col_idx+4] += 1
        V[v_h][col_idx+4] += 1

        # M6 = (c-a)(e+f)
        U[u_c][col_idx+5] += 1
        U[u_a][col_idx+5] -= 1
        V[v_e][col_idx+5] += 1
        V[v_f][col_idx+5] += 1

        # M7 = (b-d)(g+h)
        U[u_b][col_idx+6] += 1
        U[u_d][col_idx+6] -= 1
        V[v_g][col_idx+6] += 1
        V[v_h][col_idx+6] += 1

        # Indices in W (Output C)
        # C is 2x9. W index = col * n + row (based on task description: W[k*n+i, r])
        # Wait, task says "output factor follows the official trace convention W[k*n+i,r]"
        # This usually means W is indexed by [output_element_index, rank_index]
        # Output element (i, j) corresponds to index j*n + i if C is stored column-major or similar?
        # Let's check the seed program's W indexing.
        # Seed: W[k*n + 0] and W[k*n + 1].
        # Here k is the column index of B (which is the column index of C).
        # So W index = col * n + row.
        
        w_00 = k0 * n + 0  # C[0, k0]
        w_01 = k1 * n + 0  # C[0, k1]
        w_10 = k0 * n + 1  # C[1, k0]
        w_11 = k1 * n + 1  # C[1, k1]

        # Reconstruct C from Strassen multiplies
        # C00 = M1 + M4 - M5 + M7
        W[w_00][col_idx] += 1
        W[w_00][col_idx+3] += 1
        W[w_00][col_idx+4] -= 1
        W[w_00][col_idx+6] += 1

        # C01 = M3 + M5
        W[w_01][col_idx+2] += 1
        W[w_01][col_idx+4] += 1

        # C10 = M2 + M4
        W[w_10][col_idx+1] += 1
        W[w_10][col_idx+3] += 1

        # C11 = M1 - M2 + M3 + M6
        W[w_11][col_idx] += 1
        W[w_11][col_idx+1] -= 1
        W[w_11][col_idx+2] += 1
        W[w_11][col_idx+5] += 1

        col_idx += 7

    # Single column k=8 for A[:, 0:2] * B[0:2, 8]
    k8 = 8
    v_e8 = 0 * p + k8  # B[0, 8]
    v_g8 = 1 * p + k8  # B[1, 8]

    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1

    # Direct multiplication for 2x2 * 2x1
    # C0 = A00*B00 + A01*B10
    # C1 = A10*B00 + A11*B10
    
    # Mult 1: A00 * B00
    U[u_a][col_idx] += 1
    V[v_e8][col_idx] += 1
    W[k8 * n + 0][col_idx] += 1

    # Mult 2: A01 * B10
    U[u_b][col_idx+1] += 1
    V[v_g8][col_idx+1] += 1
    W[k8 * n + 0][col_idx+1] += 1

    # Mult 3: A10 * B00
    U[u_c][col_idx+2] += 1
    V[v_e8][col_idx+2] += 1
    W[k8 * n + 1][col_idx+2] += 1

    # Mult 4: A11 * B10
    U[u_d][col_idx+3] += 1
    V[v_g8][col_idx+3] += 1
    W[k8 * n + 1][col_idx+3] += 1

    col_idx += 4

    # Part 2: A[:, 2:3] * B[2, :]
    # A col 2, B row 2.
    # Outer product.
    for k in range(9):
        v_b = 2 * p + k  # B[2, k]
        
        u_a2 = 0 * m + 2  # A[0, 2]
        u_b2 = 1 * m + 2  # A[1, 2]
        
        # C0k = A02 * B2k
        U[u_a2][col_idx] += 1
        V[v_b][col_idx] += 1
        W[k * n + 0][col_idx] += 1
        
        # C1k = A12 * B2k
        U[u_b2][col_idx+1] += 1
        V[v_b][col_idx+1] += 1
        W[k * n + 1][col_idx+1] += 1
        
        col_idx += 2

    if time.time() - start_time > TIME_LIMIT:
        return {"U": U, "V": V, "W": W}

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

    # W is indexed as [col * n + row]
    # C is [n, p]. C[row, col] is at index col * n + row in C_flat?
    # Yes, if C_flat is stored column-major or we reshape appropriately.
    # C_actual = C_flat.reshape((p, n)).T
    C_actual = C_flat.reshape((p, n)).T

    if not np.allclose(C_actual, C_expected, atol=1e-6):
        print("Mismatch detected")
    else:
        print("Validation Passed")
