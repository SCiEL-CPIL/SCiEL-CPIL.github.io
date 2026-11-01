"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def check_decomposition(U, V, W):
        """Check if U,V,W form a valid decomposition."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Strategy: Rank 36 decomposition
    # Split B into 2 halves of 4 columns each.
    # For each half (3x4), split into:
    #   - Top 2 rows of B (2x4) multiplied by top 2 cols of A (2x2) -> use 2 Strassen blocks of rank 7 each = 14
    #   - Bottom 1 row of B (1x4) multiplied by bottom 1 col of A (2x1) -> rank 4
    # Total per half: 18. Total: 36.
    
    rank = 36
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(16)]
    
    # Indexing:
    # U[i*m+j, c] = U[i*3+j, c] for A[i,j]
    # V[j*p+k, c] = V[j*8+k, c] for B[j,k]
    # W[k*n+i, c] = W[k*2+i, c] for C[i,k]
    
    def add_strassen_2x2x2(base, a_r0, a_r1, b_c0, b_c1, b_k0):
        """
        Add Strassen 2x2*2x2 decomposition at column offset 'base'.
        A submatrix: rows a_r0, a_r1; cols b_c0, b_c1 (these are indices into A)
        B submatrix: rows b_c0, b_c1; cols b_k0, b_k0+1 (these are indices into B)
        C submatrix: rows a_r0, a_r1; cols b_k0, b_k0+1
        
        Strassen formulas:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B10-B00)
        M4 = A11*(B01+B11)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B01-B11)
        M7 = (A01-A11)*(B00+B10)
        
        C00 = M1 - M3 + M4 - M7
        C01 = M4 + M5
        C10 = M2 + M3
        C11 = M1 - M2 + M6 - M7
        
        Where:
        A00 = A[a_r0, b_c0], A01 = A[a_r0, b_c1]
        A10 = A[a_r1, b_c0], A11 = A[a_r1, b_c1]
        B00 = B[b_c0, b_k0],  B01 = B[b_c0, b_k0+1]
        B10 = B[b_c1, b_k0],  B11 = B[b_c1, b_k0+1]
        C00 = C[a_r0, b_k0],  C01 = C[a_r0, b_k0+1]
        C10 = C[a_r1, b_k0],  C11 = C[a_r1, b_k0+1]
        """
        # M1 = (A00+A11)*(B00+B11)
        U[a_r0*3+b_c0][base] += 1
        U[a_r1*3+b_c1][base] += 1
        V[b_c0*8+b_k0][base] += 1
        V[b_c1*8+b_k0+1][base] += 1
        W[b_k0*2+a_r0][base] += 1    # C00
        W[(b_k0+1)*2+a_r0][base] += 1 # C01
        W[b_k0*2+a_r1][base] += 1    # C10
        W[(b_k0+1)*2+a_r1][base] += 1 # C11
        
        # M2 = (A10+A11)*B00
        U[a_r1*3+b_c0][base+1] += 1
        U[a_r1*3+b_c1][base+1] += 1
        V[b_c0*8+b_k0][base+1] += 1
        W[b_k0*2+a_r1][base+1] += 1    # C10
        W[(b_k0+1)*2+a_r1][base+1] += -1 # C11
        
        # M3 = A00*(B10-B00)
        U[a_r0*3+b_c0][base+2] += 1
        V[b_c1*8+b_k0][base+2] += 1
        V[b_c0*8+b_k0][base+2] += -1
        W[b_k0*2+a_r1][base+2] += 1    # C10
        W[(b_k0+1)*2+a_r0][base+2] += -1 # C01
        
        # M4 = A11*(B01+B11)
        U[a_r1*3+b_c1][base+3] += 1
        V[b_c0*8+b_k0+1][base+3] += 1
        V[b_c1*8+b_k0+1][base+3] += 1
        W[b_k0*2+a_r0][base+3] += 1    # C00
        W[(b_k0+1)*2+a_r0][base+3] += 1 # C01
        
        # M5 = (A00+A01)*B11
        U[a_r0*3+b_c0][base+4] += 1
        U[a_r0*3+b_c1][base+4] += 1
        V[b_c1*8+b_k0+1][base+4] += 1
        W[b_k0*2+a_r0][base+4] += 1    # C00
        W[(b_k0+1)*2+a_r0][base+4] += -1 # C01
        
        # M6 = (A10-A00)*(B01-B11)
        U[a_r1*3+b_c0][base+5] += 1
        U[a_r0*3+b_c0][base+5] += -1
        V[b_c0*8+b_k0+1][base+5] += 1
        V[b_c1*8+b_k0+1][base+5] += -1
        W[(b_k0+1)*2+a_r1][base+5] += 1 # C11
        
        # M7 = (A01-A11)*(B00+B10)
        U[a_r0*3+b_c1][base+6] += 1
        U[a_r1*3+b_c1][base+6] += -1
        V[b_c0*8+b_k0][base+6] += 1
        V[b_c1*8+b_k0][base+6] += 1
        W[b_k0*2+a_r0][base+6] += -1   # C00
        W[b_k0*2+a_r1][base+6] += -1   # C10
    
    # Build the rank-36 decomposition
    for half in range(2):
        bk = half * 4  # base column index in B
        base = half * 18  # base column index in rank
        
        # 2x2*2x4 part: A rows 0,1; A cols 0,1; B rows 0,1; B cols bk..bk+3
        # Process 2 cols at a time with Strassen
        add_strassen_2x2x2(base, 0, 1, 0, 1, bk)       # B cols bk, bk+1
        add_strassen_2x2x2(base+7, 0, 1, 0, 1, bk+2)   # B cols bk+2, bk+3
        
        # 2x1*1x4 part: A col 2; B row 2; B cols bk..bk+3
        # C[i, bk+dk] += A[i,2] * B[2, bk+dk]
        for dk in range(4):
            c = base + 14 + dk
            U[0*3+2][c] += 1  # A[0,2]
            U[1*3+2][c] += 1  # A[1,2]
            V[2*8+bk+dk][c] += 1  # B[2, bk+dk]
            W[(bk+dk)*2+0][c] += 1  # C[0, bk+dk]
            W[(bk+dk)*2+1][c] += 1  # C[1, bk+dk]
    
    # Verify the decomposition
    if check_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If verification failed, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
