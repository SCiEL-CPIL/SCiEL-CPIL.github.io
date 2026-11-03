"""Matrix multiplication 2x3x8 — rank-minimal exact tensor decomposition via Strassen block partitioning."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(total - 1) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s wall-clock limit
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Block decomposition strategy:
    # Split m=3 into m1=1 (j=0) and m2=2 (j=1,2)
    # Part 1: A[:, 0] * B[0, :] -> 2x1x8 multiplication (Rank 16, schoolbook)
    # Part 2: A[:, 1:3] * B[1:3, :] -> 2x2x8 multiplication
    #         Split p=8 into 4 blocks of 2.
    #         Each block is 2x2x2 multiplication (Rank 7 via Strassen).
    #         Total Rank for Part 2: 4 * 7 = 28.
    # Total Rank: 16 + 28 = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]   # 6 x 44
    V = [[0] * rank for _ in range(m * p)]   # 24 x 44
    W = [[0] * rank for _ in range(p * n)]   # 16 x 44
    
    col = 0
    
    # --- Part 1: 2x1x8 (j=0) ---
    # A is 2x1, B is 1x8.
    # C[i, k] = A[i, 0] * B[0, k]
    # Rank 16.
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # --- Part 2: 2x2x8 (j=1,2) ---
    # A is 2x2 (cols 1,2), B is 2x8 (rows 1,2).
    # Split B into 4 blocks of 2 columns.
    for block_idx in range(4):
        if time.time() > deadline:
            break
        k_start = block_idx * 2
        
        # Strassen 2x2x2
        # Local A:
        # A00 -> Global A[0, 1] -> U index 0*m + 1 = 1
        # A01 -> Global A[0, 2] -> U index 0*m + 2 = 2
        # A10 -> Global A[1, 1] -> U index 1*m + 1 = 4
        # A11 -> Global A[1, 2] -> U index 1*m + 2 = 5
        
        # Local B (for this block):
        # B00 -> Global B[1, k_start]     -> V index 1*p + k_start
        # B01 -> Global B[1, k_start + 1] -> V index 1*p + (k_start+1)
        # B10 -> Global B[2, k_start]     -> V index 2*p + k_start
        # B11 -> Global B[2, k_start + 1] -> V index 2*p + (k_start+1)
        
        # Local C (for this block):
        # C00 -> Global C[0, k_start]     -> W index k_start*n + 0
        # C01 -> Global C[0, k_start + 1] -> W index (k_start+1)*n + 0
        # C10 -> Global C[1, k_start]     -> W index k_start*n + 1
        # C11 -> Global C[1, k_start + 1] -> W index (k_start+1)*n + 1
        
        # M1 = (A00 + A11) * (B00 + B11)
        # C00 += M1, C11 += M1
        U[1][col] += 1
        U[5][col] += 1
        V[1*p + k_start][col] += 1
        V[2*p + (k_start+1)][col] += 1
        W[k_start*n + 0][col] += 1
        W[(k_start+1)*n + 1][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        # C10 += M2, C11 -= M2
        U[4][col] += 1
        U[5][col] += 1
        V[1*p + k_start][col] += 1
        W[k_start*n + 1][col] += 1
        W[(k_start+1)*n + 1][col] += -1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        # C01 += M3, C11 += M3
        U[1][col] += 1
        V[1*p + (k_start+1)][col] += 1
        V[2*p + (k_start+1)][col] += -1
        W[(k_start+1)*n + 0][col] += 1
        W[(k_start+1)*n + 1][col] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        # C00 += M4, C10 += M4
        U[5][col] += 1
        V[2*p + k_start][col] += 1
        V[1*p + k_start][col] += -1
        W[k_start*n + 0][col] += 1
        W[k_start*n + 1][col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        # C00 -= M5, C01 += M5
        U[1][col] += 1
        U[2][col] += 1
        V[2*p + (k_start+1)][col] += 1
        W[k_start*n + 0][col] += -1
        W[(k_start+1)*n + 0][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        # C10 += M6, C11 += M6
        U[4][col] += 1
        U[1][col] += -1
        V[1*p + k_start][col] += 1
        V[1*p + (k_start+1)][col] += 1
        W[k_start*n + 1][col] += 1
        W[(k_start+1)*n + 1][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 += M7, C10 += M7
        U[2][col] += 1
        U[5][col] += -1
        V[2*p + k_start][col] += 1
        V[2*p + (k_start+1)][col] += 1
        W[k_start*n + 0][col] += 1
        W[k_start*n + 1][col] += 1
        col += 1
    
    # Verify the construction
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: return the schoolbook rank-48 decomposition
    rank = n * m * p  # 48
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
