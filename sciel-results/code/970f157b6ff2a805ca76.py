"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_with_numpy(n, m, p, U, V, W):
    """Faster verification using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j]
                v_val = V_arr[j*p + k]
                w_val = W_arr[k*n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Strategy: Use Strassen's algorithm on 2x2x2 sub-blocks for the j={0,1} part,
    # and schoolbook for the j={2} part.
    # Part 1: 2x2x10 -> split k into 5 pairs of 2, each using Strassen (rank 7)
    #         Total: 5 * 7 = 35
    # Part 2: 2x1x10 -> schoolbook: 2*1*10 = 20
    # Total rank: 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 3*10 = 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 10*2 = 20 rows
    
    # Strassen decomposition for 2x2x2 matrix multiplication
    # Indices: A[i,j] with i,j in {0,1}; B[j,k] with j,k in {0,1}; C[i,k] with i,k in {0,1}
    # W is indexed by (k,i) as W[k*n+i], so for n=2: W[0,0]=C[0,0], W[0,1]=C[1,0], W[1,0]=C[0,1], W[1,1]=C[1,1]
    
    # The 7 Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # U factors (coefficients on A[i,j], indexed as [A00, A01, A10, A11]):
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    
    # V factors (coefficients on B[j,k], indexed as [B00, B01, B10, B11]):
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ]
    
    # W factors (coefficients on C, indexed as [C00, C10, C01, C11] since W[k*n+i]):
    # W[0,0]=C[0,0], W[0,1]=C[1,0], W[1,0]=C[0,1], W[1,1]=C[1,1]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 1, 0, 0],   # M2: C10
        [0, 0, 1, -1],  # M3: C01 - C11
        [-1, 1, 0, 0],  # M4: -C00 + C10
        [0, 0, 1, 0],   # M5: C01
        [1, 0, 0, 0],   # M6: C00
        [0, 0, 0, 1],   # M7: C11
    ]
    
    col = 0
    
    # Part 1: 2x2x10 using Strassen on pairs of k
    for pair_idx in range(5):
        k0 = 2 * pair_idx
        k1 = 2 * pair_idx + 1
        for c in range(7):
            # U: for i in {0,1}, j in {0,1}: U[i*m+j][col] = strassen_U[c][i*2+j]
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = strassen_U[c][i * 2 + j]
            
            # V: for j in {0,1}, k in {k0,k1}: V[j*p+k][col] = strassen_V[c][j*2+(k-k0)]
            for j in range(2):
                for k in (k0, k1):
                    V[j * p + k][col] = strassen_V[c][j * 2 + (k - k0)]
            
            # W: for k in {k0,k1}, i in {0,1}: W[k*n+i][col] = strassen_W[c][(k-k0)*2+i]
            for k in (k0, k1):
                for i in range(2):
                    W[k * n + i][col] = strassen_W[c][(k - k0) * 2 + i]
            
            col += 1
    
    # Part 2: 2x1x10 schoolbook for j=2
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    
    # Verify
    if not _verify_with_numpy(n, m, p, U, V, W):
        # Fallback to naive rank-60 decomposition
        rank_naive = n * m * p
        U = [[0] * rank_naive for _ in range(n * m)]
        V = [[0] * rank_naive for _ in range(m * p)]
        W = [[0] * rank_naive for _ in range(p * n)]
        col2 = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii*m+jj][col2] = 1
                    V[jj*p+kk][col2] = 1
                    W[kk*n+ii][col2] = 1
                    col2 += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
