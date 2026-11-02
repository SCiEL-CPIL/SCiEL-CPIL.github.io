"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # safety margin

    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 48

    # Try to use Strassen for the 2x2 part, sharing across the 12 columns
    # This gives rank 7 * 6 = 42 (Strassen rank 7 for 2x2, times 6 pairs of columns)
    # Actually, let's think more carefully.
    
    # For multiplying (2x2) by (2x12):
    # The result is 2x12.
    # We can view this as 6 independent 2x2 multiplications (columns 0-1, 2-3, ..., 10-11)
    # But Strassen gives rank 7 for 2x2 multiplication.
    # If we share the Strassen factors across all 6 blocks, we get rank 7 * 6 = 42.
    
    # Actually, we can do better. Let's think of it as:
    # A is 2x2, B is 2x12.
    # C = A @ B, so C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor is of size 2*2*12 = 48.
    # Schoolbook: rank 48.
    
    # Using Strassen-like decomposition:
    # For each pair of output columns (2k, 2k+1), we can use Strassen on the 2x2 subproblem.
    # But we can share the U and V factors if we structure it right.
    
    # Let me construct a rank-42 decomposition using Strassen.
    # Strassen for 2x2: 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For our problem, A is 2x2 and B is 2x12.
    # Let's partition B into 6 blocks of 2x2: B[:, 0:2], B[:, 2:4], ..., B[:, 10:12]
    
    # For block t (t=0..5), B_t is 2x2 with entries B_t[0,0]=B[0,2t], B_t[0,1]=B[0,2t+1], etc.
    
    # Strassen factors for block t:
    # U factors (for A): 
    #   u1 = A00+A11, u2 = A10+A11, u3 = A00, u4 = A11, u5 = A00+A10, u6 = A11-A10, u7 = A01-A11
    # These are the same for all blocks! So we only need 7 U entries.
    
    # V factors (for B): for each block t, we need:
    #   v1_t = B_t[0,0]+B_t[1,1]
    #   v2_t = B_t[0,0]
    #   v3_t = B_t[1,0]-B_t[1,1]
    #   v4_t = B_t[0,1]-B_t[0,0]
    #   v5_t = B_t[1,1]
    #   v6_t = B_t[0,0]+B_t[1,0]
    #   v7_t = B_t[0,1]+B_t[1,1]
    
    # So V has 7*6 = 42 entries.
    
    # W factors (for C): for each block t and each output position:
    # C[0,2t] = M1 - M4 + M6 - M7
    # C[0,2t+1] = M3 + M4 + M7
    # C[1,2t] = M2 + M3 - M5 + M7
    # C[1,2t+1] = M1 - M2 + M5 + M6
    
    # Wait, let me re-derive Strassen properly.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A@B = [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M4 + M7
    # C10 = M2 + M3 - M5 + M7
    # C11 = M1 - M2 + M5 + M6
    
    # Let me verify: 
    # C00 = A00*B00 + A01*B10
    # M1 - M4 + M6 - M7 = (A00+A11)(B00+B11) - A11(B01-B00) + (A11-A10)(B00+B10) - (A01-A11)(B01+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A11*B01 + A11*B00 + A11*B00 + A11*B10 - A10*B00 - A10*B10 - A01*B01 - A01*B11 + A11*B01 + A11*B11
    # Let me collect terms:
    # A00*B00: +1
    # A00*B11: +1 - 1 = 0 (from -A01*B11? No wait, -A01*B11 is different)
    # Let me redo this more carefully.
    
    # Actually, let me just use the known Strassen decomposition and verify numerically.
    
    # For now, let's construct the rank-42 decomposition and verify it.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # U factors (shared across all 6 blocks):
    # Row 0 = A[0,0], Row 1 = A[0,1], Row 2 = A[1,0], Row 3 = A[1,1]
    # u1 = A00+A11: rows 0 and 3 get +1
    # u2 = A10+A11: rows 2 and 3 get +1
    # u3 = A00: row 0 gets +1
    # u4 = A11: row 3 gets +1
    # u5 = A00+A10: rows 0 and 2 get +1
    # u6 = A11-A10: row 3 gets +1, row 2 gets -1
    # u7 = A01-A11: row 1 gets +1, row 3 gets -1
    
    # For each block t (0..5), columns 7*t + s (s=0..6) correspond to M_{s+1}
    # U[0][7*t+0] = 1  (u1: A00)
    # U[3][7*t+0] = 1  (u1: A11)
    # U[2][7*t+1] = 1  (u2: A10)
    # U[3][7*t+1] = 1  (u2: A11)
    # U[0][7*t+2] = 1  (u3: A00)
    # U[3][7*t+3] = 1  (u4: A11)
    # U[0][7*t+4] = 1  (u5: A00)
    # U[2][7*t+4] = 1  (u5: A10)
    # U[3][7*t+5] = 1  (u6: A11)
    # U[2][7*t+5] = -1 (u6: -A10)
    # U[1][7*t+6] = 1  (u7: A01)
    # U[3][7*t+6] = -1 (u7: -A11)
    
    for t in range(6):
        base = 7 * t
        U[0][base + 0] = 1
        U[3][base + 0] = 1
        U[2][base + 1] = 1
        U[3][base + 1] = 1
        U[0][base + 2] = 1
        U[3][base + 3] = 1
        U[0][base + 4] = 1
        U[2][base + 4] = 1
        U[3][base + 5] = 1
        U[2][base + 5] = -1
        U[1][base + 6] = 1
        U[3][base + 6] = -1
    
    # V factors: for block t, B_t is 2x2 with:
    # B_t[0,0] = B[0, 2t], B_t[0,1] = B[0, 2t+1]
    # B_t[1,0] = B[1, 2t], B_t[1,1] = B[1, 2t+1]
    # V rows: 0..11 for B[0,*], 12..23 for B[1,*]
    # v1_t = B_t[0,0]+B_t[1,1]: V[0,2t] += 1, V[12+2t+1] += 1
    # v2_t = B_t[0,0]: V[0,2t] += 1
    # v3_t = B_t[1,0]-B_t[1,1]: V[12+2t] += 1, V[12+2t+1] += -1
    # v4_t = B_t[0,1]-B_t[0,0]: V[0,2t+1] += 1, V[0,2t] += -1
    # v5_t = B_t[1,1]: V[12+2t+1] += 1
    # v6_t = B_t[0,0]+B_t[1,0]: V[0,2t] += 1, V[12+2t] += 1
    # v7_t = B_t[0,1]+B_t[1,1]: V[0,2t+1] += 1, V[12+2t+1] += 1
    
    for t in range(6):
        base = 7 * t
        b00_idx = 0 * p + 2 * t       # B[0, 2t]
        b01_idx = 0 * p + 2 * t + 1   # B[0, 2t+1]
        b10_idx = 1 * p + 2 * t       # B[1, 2t]
        b11_idx = 1 * p + 2 * t + 1   # B[1, 2t+1]
        
        V[b00_idx][base + 0] += 1
        V[b11_idx][base + 0] += 1
        
        V[b00_idx][base + 1] += 1
        
        V[b10_idx][base + 2] += 1
        V[b11_idx][base + 2] += -1
        
        V[b01_idx][base + 3] += 1
        V[b00_idx][base + 3] += -1
        
        V[b11_idx][base + 4] += 1
        
        V[b00_idx][base + 5] += 1
        V[b10_idx][base + 5] += 1
        
        V[b01_idx][base + 6] += 1
        V[b11_idx][base + 6] += 1
    
    # W factors: C is 2x12, stored as W[k*n+i, c] = W[12*i + k, c]
    # C[0,2t] = M1 - M4 + M6 - M7
    # C[0,2t+1] = M3 + M4 + M7
    # C[1,2t] = M2 + M3 - M5 + M7
    # C[1,2t+1] = M1 - M2 + M5 + M6
    
    for t in range(6):
        base = 7 * t
        c00_idx = 0 * n + 2 * t       # C[0, 2t]
        c01_idx = 0 * n + 2 * t + 1   # C[0, 2t+1]
        c10_idx = 1 * n + 2 * t       # C[1, 2t]
        c11_idx = 1 * n + 2 * t + 1   # C[1, 2t+1]
        
        # C[0,2t] = M1 - M4 + M6 - M7
        W[c00_idx][base + 0] += 1
        W[c00_idx][base + 3] += -1
        W[c00_idx][base + 5] += 1
        W[c00_idx][base + 6] += -1
        
        # C[0,2t+1] = M3 + M4 + M7
        W[c01_idx][base + 2] += 1
        W[c01_idx][base + 3] += 1
        W[c01_idx][base + 6] += 1
        
        # C[1,2t] = M2 + M3 - M5 + M7
        W[c10_idx][base + 1] += 1
        W[c10_idx][base + 2] += 1
        W[c10_idx][base + 4] += -1
        W[c10_idx][base + 6] += 1
        
        # C[1,2t+1] = M1 - M2 + M5 + M6
        W[c11_idx][base + 0] += 1
        W[c11_idx][base + 1] += -1
        W[c11_idx][base + 4] += 1
        W[c11_idx][base + 5] += 1
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = delta (i.e., T[i,j,k] = 1 if this is the matrix mult tensor)
    # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] where the decomposition is sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Wait, the convention is W[k*n+i, r], so W index is k*n+i.
    
    # Let's verify by checking that for random A, B:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = sum_j A[i,j]*B[j,k]
    
    # Actually, the tensor decomposition means:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal the tensor entry.
    # The matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]B[j,k].
    # Wait no. The tensor for matrix multiplication is defined such that:
    # C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And this should equal sum_j A[i,j] * B[j,k].
    # So the tensor entry T[i,j,k] = 1 for all i,j,k (it's the identity in the middle).
    
    # Let me verify numerically.
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_computed[i, k] += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
