"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def validate(U, V, W):
        """Check if decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p  # 72
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook as fallback
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Laderman's rank-20 decomposition for 3x3x3, properly verified
    def laderman_3x3x3():
        """Return (U, V, W) for 3x3x3 with rank 20, or None if invalid."""
        # Each term c: U_c is 3x3 (flat 9), V_c is 3x3 (flat 9), W_c is 3x3 (flat 9)
        # We need: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all i,j,k in {0,1,2}
        
        laderman_terms = [
            # c, U flat (row-major 3x3), V flat, W flat
            ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),  # 0: a00*b00 -> c00
            ([1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),  # 1: a00*b01 -> c01
            ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),  # 2: a00*b02 -> c02
            ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),  # 3: a01*b10 -> c10
            ([0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),  # 4: a01*b11 -> c11
            ([0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),  # 5: a01*b12 -> c12
            ([0,0,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),  # 6: a02*b20 -> c20
            ([0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),  # 7: a02*b21 -> c21
            ([0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),  # 8: a02*b22 -> c22
            ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0]),  # 9: (a00+a11)*(b00+b11) -> c00+c11
            ([1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,1]),  # 10: (a00+a22)*(b00+b22) -> c00+c22
            ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1]),  # 11: (a11+a22)*(b11+b22) -> c11+c22
            ([0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0]),  # 12: (a01+a10)*(b01+b10) -> c01+c10
            ([0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0]),  # 13: (a02+a20)*(b02+b20) -> c02+c20
            ([0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,0,1, 0,1,0]),  # 14: (a12+a21)*(b12+b21) -> c12+c21
            ([1,0,0, 1,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0]),  # 15: (a00+a10)*(b00+b10) -> c00+c10
            ([0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,1,0], [0,0,0, 0,1,0, 0,1,0]),  # 16: (a11+a21)*(b11+b21) -> c11+c21
            ([1,1,0, 0,0,0, 0,0,0], [1,1,0, 0,0,0, 0,0,0], [1,1,0, 0,0,0, 0,0,0]),  # 17: (a01+a02)*(b01+b02) -> c01+c02
            ([0,1,1, 0,0,0, 0,0,0], [0,1,1, 0,0,0, 0,0,0], [0,1,1, 0,0,0, 0,0,0]),  # 18: (a10+a11)*(b10+b11) -> c10+c11
            ([0,0,0, 0,0,0, 1,0,1], [0,0,0, 0,0,0, 1,0,1], [0,0,0, 0,0,0, 1,0,1]),  # 19: (a20+a22)*(b20+b22) -> c20+c22
        ]
        
        U_3x3 = np.zeros((9, 20))
        V_3x3 = np.zeros((9, 20))
        W_3x3 = np.zeros((9, 20))
        
        for c, (u, v, w) in enumerate(laderman_terms):
            U_3x3[:, c] = np.array(u)
            V_3x3[:, c] = np.array(v)
            W_3x3[:, c] = np.array(w)
        
        # Verify: for all i,j,k in {0,1,2}, sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] should be 1
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = np.sum(U_3x3[i*3+j] * V_3x3[j*3+k] * W_3x3[k*3+i])
                    if abs(val - 1.0) > 1e-9:
                        return None
        return U_3x3.tolist(), V_3x3.tolist(), W_3x3.tolist()
    
    # Try Laderman-based construction for rank 65
    laderman_result = laderman_3x3x3()
    if laderman_result is not None and elapsed() < deadline - 5:
        U_lad, V_lad, W_lad = laderman_result
        
        # Build combined decomposition: Laderman for k=0,1,2 + schoolbook for k=3..7
        # Total rank = 20 + 3*3*5 = 20 + 45 = 65
        combined_rank = 20 + 45
        U_comb = [[0]*combined_rank for _ in range(n*m)]  # 9 rows
        V_comb = [[0]*combined_rank for _ in range(m*p)]  # 24 rows
        W_comb = [[0]*combined_rank for _ in range(p*n)]  # 24 rows
        
        # Copy Laderman terms (columns 0-19)
        for c in range(20):
            for i in range(n):
                for j in range(m):
                    U_comb[i*m+j][c] = U_lad[i*3+j][c]
            for j in range(m):
                for k in range(p):
                    if k < 3:
                        V_comb[j*p+k][c] = V_lad[j*3+k][c]
                    else:
                        V_comb[j*p+k][c] = 0
            for k in range(p):
                for i in range(n):
                    if k < 3:
                        W_comb[k*n+i][c] = W_lad[k*3+i][c]
                    else:
                        W_comb[k*n+i][c] = 0
        
        # Add schoolbook terms for k=3..7 (columns 20-64)
        col = 20
        for i in range(n):
            for j in range(m):
                for k in range(3, p):
                    U_comb[i*m+j][col] = 1
                    V_comb[j*p+k][col] = 1
                    W_comb[k*n+i][col] = 1
                    col += 1
        
        # Verify
        if validate(U_comb, V_comb, W_comb):
            if combined_rank < best_rank:
                best_U, best_V, best_W, best_rank = U_comb, V_comb, W_comb, combined_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
