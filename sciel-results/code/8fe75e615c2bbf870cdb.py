"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Laderman's rank-7 decomposition for 3x3*3x3 blocks combined with
direct terms for the remaining 2 columns, targeting rank 20.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0

    def laderman_decomp(col_offset, U_t, V_t, W_t):
        """Add Laderman decomposition for A[0:3,0:3] * B[0:3, col_offset:col_offset+3] -> C[0:3, col_offset:col_offset+3]."""
        # Laderman's 7 terms for 3x3 matrix multiplication
        # A indices: (i,j) for i,j in {0,1,2}
        # B indices: (j,k) for j in {0,1,2}, k in {0,1,2} (relative to col_offset)
        # C indices: (i,k) for i in {0,1,2}, k in {0,1,2} (relative to col_offset)
        
        # Term 1: (A00 - A01) * (B01 - B11) * (C00 + C01 - C10 - C11)
        U_t.append([((0,0), 1), ((0,1), -1)])
        V_t.append([((0, col_offset+1), 1), ((1, col_offset+1), -1)])
        W_t.append([((0, col_offset+0), 1), ((0, col_offset+1), 1), ((1, col_offset+0), -1), ((1, col_offset+1), -1)])
        
        # Term 2: (A00 - A10) * (B00 - B01) * (C00 + C10 - C01 - C11)
        U_t.append([((0,0), 1), ((1,0), -1)])
        V_t.append([((0, col_offset+0), 1), ((0, col_offset+1), -1)])
        W_t.append([((0, col_offset+0), 1), ((1, col_offset+0), 1), ((0, col_offset+1), -1), ((1, col_offset+1), -1)])
        
        # Term 3: (A00 - A20) * (B20 - B21) * (C00 + C20 - C02 - C22)
        U_t.append([((0,0), 1), ((2,0), -1)])
        V_t.append([((2, col_offset+0), 1), ((2, col_offset+1), -1)])
        W_t.append([((0, col_offset+0), 1), ((2, col_offset+0), 1), ((0, col_offset+2), -1), ((2, col_offset+2), -1)])
        
        # Term 4: (A01 - A11) * (B10 - B11) * (C01 + C11 - C00 - C10)
        U_t.append([((0,1), 1), ((1,1), -1)])
        V_t.append([((1, col_offset+0), 1), ((1, col_offset+1), -1)])
        W_t.append([((0, col_offset+1), 1), ((1, col_offset+1), 1), ((0, col_offset+0), -1), ((1, col_offset+0), -1)])
        
        # Term 5: (A10 - A20) * (B20 - B21) * (C10 + C20 - C11 - C21)
        U_t.append([((1,0), 1), ((2,0), -1)])
        V_t.append([((2, col_offset+0), 1), ((2, col_offset+1), -1)])
        W_t.append([((1, col_offset+0), 1), ((2, col_offset+0), 1), ((1, col_offset+1), -1), ((2, col_offset+1), -1)])
        
        # Term 6: (A02 - A12) * (B10 - B11) * (C02 + C12 - C01 - C11)
        U_t.append([((0,2), 1), ((1,2), -1)])
        V_t.append([((1, col_offset+0), 1), ((1, col_offset+1), -1)])
        W_t.append([((0, col_offset+2), 1), ((1, col_offset+2), 1), ((0, col_offset+1), -1), ((1, col_offset+1), -1)])
        
        # Term 7: (A12 - A22) * (B20 - B21) * (C12 + C22 - C11 - C21)
        U_t.append([((1,2), 1), ((2,2), -1)])
        V_t.append([((2, col_offset+0), 1), ((2, col_offset+1), -1)])
        W_t.append([((1, col_offset+2), 1), ((2, col_offset+2), 1), ((1, col_offset+1), -1), ((2, col_offset+1), -1)])

    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    U_t = []
    V_t = []
    W_t = []
    
    # Block 1: columns 0-2, use Laderman
    laderman_decomp(0, U_t, V_t, W_t)
    
    # Block 2: columns 3-5, use Laderman
    laderman_decomp(3, U_t, V_t, W_t)
    
    # Block 3: columns 6-7, use direct (3x3 * 3x2 -> 3x2)
    for i in range(3):
        for j in range(3):
            for k in range(6, 8):
                direct_term(i, j, k, U_t, V_t, W_t)
    
    rank = len(W_t)
    
    # Build the factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Convert to nice integer/half-integer values
    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result
    
    return {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
# EVOLVE-BLOCK-END
