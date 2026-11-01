import numpy as np
import time

def construct():
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit
    
    rank = 25
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    def add_strassen_block(col_start, k_start):
        # Standard Strassen 2x2x2 block adapted for column pair (k_start, k_start+1)
        # Term 0: (a00+a11)(b0k0 + b1k1) -> contributes to c0k0 and c1k1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # Term 1: (a10+a11)(b0k0) -> c1k0, c1k1
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # Term 2: a00*(b0k1 - b1k1) -> c0k1, c1k1
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # Term 3: a11*(b1k0 - b0k0) -> c0k0, c1k0
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # Term 4: (a00+a01)*(b1k1) -> -c0k0, c0k1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # Term 5: (a10-a00)*(b0k0+b0k1) -> c1k1
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # Term 6: (a01-a11)*(b1k0+b1k1) -> c0k0
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1

    # Apply Strassen blocks for column pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)

    # Handle column 6 with 4 separate terms (trivial rank-4 for single column)
    col_start = 21
    k = 6
    U[0, col_start+0] += 1
    V[7*0 + k, col_start+0] += 1
    W[2*k + 0, col_start+0] += 1

    U[1, col_start+1] += 1
    V[7*1 + k, col_start+1] += 1
    W[2*k + 0, col_start+1] += 1

    U[2, col_start+2] += 1
    V[7*0 + k, col_start+2] += 1
    W[2*k + 1, col_start+2] += 1

    U[3, col_start+3] += 1
    V[7*1 + k, col_start+3] += 1
    W[2*k + 1, col_start+3] += 1

    # Verify the decomposition is correct
    if time.time() < deadline - 5:
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        # Check all entries of the tensor
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U_arr[a_idx, :], V_arr[b_idx, :] * W_arr[c_idx, :])
                    if abs(val - 1.0) > max_err:
                        max_err = abs(val - 1.0)
        # Also check that non-matching entries are 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for j2 in range(m):
                        if j2 != j:
                            a_idx = i * m + j
                            b_idx = j2 * p + k
                            c_idx = k * n + i
                            val = np.dot(U_arr[a_idx, :], V_arr[b_idx, :] * W_arr[c_idx, :])
                            if abs(val) > max_err:
                                max_err = abs(val)
        if max_err > 1e-10:
            # Fallback: return trivial rank-28 decomposition
            rank = 28
            U = np.zeros((n*m, rank), dtype=float)
            V = np.zeros((m*p, rank), dtype=float)
            W = np.zeros((p*n, rank), dtype=float)
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, r] = 1
                        V[j*p+k, r] = 1
                        W[k*n+i, r] = 1
                        r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
