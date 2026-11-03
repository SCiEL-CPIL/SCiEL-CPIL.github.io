"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all (i,j,k)."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _construct_schoolbook():
    """Standard rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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

def _construct_44():
    """
    Rank-44 decomposition for 2x3x8 multiplication.
    
    Strategy: Partition the 3 columns of A and 8 rows of B into two blocks.
    Block 1: A_left (2x2, cols 0,1) * B_top (2x4, rows 0,1) using Strassen rank 7
    Block 2: A_left (2x2, cols 0,1) * B_bot (2x4, rows 2,3) using Strassen rank 7
    Block 3: A_right (2x1, col 2) * B_full (1x8) using schoolbook rank 8
    
    Total: 7 + 7 + 8 = 22 mults per half of B (cols 0-3), and similarly for cols 4-7.
    Wait, that's 44 for all 8 columns.
    
    Actually let me reconsider the partition:
    A is 2x3, B is 3x8.
    
    Partition A columns: A_L (cols 0,1) is 2x2, A_R (col 2) is 2x1.
    Partition B rows: B_L (rows 0,1) is 2x8, B_R (row 2) is 1x8.
    
    C = A_L * B_L + A_R * B_R
    
    For A_L * B_L (2x2 * 2x8): Use Strassen-like approach.
    Split B_L into two 2x4 blocks: B_L1 (cols 0-3), B_L2 (cols 4-7).
    For each 2x4 block, use rank-7 Strassen on the 2x2 * 2x2 part... 
    But 2x4 is not 2x2. Let me think differently.
    
    For 2x2 * 2x4: We can use the fact that this is like two 2x2 * 2x2 multiplications side by side.
    Each 2x2 * 2x2 needs rank 7 (Strassen). Two such = rank 14.
    But maybe we can do better by sharing?
    
    Alternative: Just use schoolbook for the 2x2*2x4 part = 2*2*4 = 16 mults.
    A_L * B_L: 2*2*8 = 32 mults (schoolbook)
    A_R * B_R: 2*1*8 = 16 mults (schoolbook)
    Total schoolbook: 48.
    
    Using Strassen for 2x2*2x4: 
    Actually, for multiplying 2x2 by 2x4, we can view it as 2x2 * [B1 | B2] where B1, B2 are 2x2.
    We need to compute A*B1 and A*B2. Using Strassen for each: 7+7=14 mults.
    So A_L * B_L: 14 mults.
    A_L * B_R: A_L is 2x2, B_R is 1x8. This is (2x2)*(1x8) = 2*1*8=16 schoolbook, but since B_R has only 1 row, it's just scaling. Actually A_L * B_R where B_R is 1x8 means each column of B_R is a scalar. So it's 2*8=16 mults... wait no.
    
    Let me be more careful. B is 3x8. B_R is the third row of B, which is 1x8.
    A_L is 2x2. A_L * B_R doesn't make sense dimensionally. 
    
    C = A * B where A is 2x3, B is 3x8.
    C[i,k] = sum_j A[i,j] * B[j,k]
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    So C = A[:,0:2] @ B[0:2,:] + A[:,2:3] @ B[2:3,:]
    = A_L @ B_L + A_R @ B_R
    where A_L is 2x2, B_L is 2x8, A_R is 2x1, B_R is 1x8.
    
    A_L @ B_L: 2x2 * 2x8. Schoolbook: 2*2*8 = 32 mults.
    A_R @ B_R: 2x1 * 1x8. Schoolbook: 2*1*8 = 16 mults.
    Total: 48.
    
    For A_L @ B_L (2x2 * 2x8): Split B_L into four 2x2 blocks: B_L0, B_L1, B_L2, B_L3 (cols 0-1, 2-3, 4-5, 6-7).
    A_L @ B_L0 needs rank 7 (Strassen).
    Similarly for B_L1, B_L2, B_L3: 7 each = 28 mults.
    
    A_R @ B_R: 2x1 * 1x8 = 16 mults (schoolbook, since it's just outer product of column and row).
    
    Total: 28 + 16 = 44.
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    U = [[0] * 44 for _ in range(n * m)]   # 6 x 44
    V = [[0] * 44 for _ in range(m * p)]   # 24 x 44
    W = [[0] * 44 for _ in range(p * n)]   # 16 x 44
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1 + M5 - M6
    # C10 = M2 + M4 - M7 + M6
    # C11 = M3 + M2 + M1 - M4
    
    col_offset = 0
    for block in range(4):  # 4 blocks of 2 columns each
        # For block b, columns are 2*b, 2*b+1
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen mults for A_L (2x2) * B_L_block (2x2)
        # A_L indices: A[0,0]->U[0], A[0,1]->U[1], A[1,0]->U[3], A[1,1]->U[4]
        # B_L_block indices: B[0,k0]->V[0*8+k0]=V[k0], B[0,k1]->V[k1], B[1,k0]->V[8+k0], B[1,k1]->V[8+k1]
        # C indices: C[0,k0]->W[k0*2+0]=W[k0*2], C[0,k1]->W[k1*2], C[1,k0]->W[k0*2+1], C[1,k1]->W[k1*2+1]
        
        # M1 = (A00+A11)*(B00+B11)
        c = col_offset
        U[0][c] = 1  # A00
        U[4][c] = 1  # A11
        V[k0][c] = 1  # B[0,k0]
        V[8+k1][c] = 1  # B[1,k1]
        W[k0*2][c] = 1  # C[0,k0]
        W[k1*2+1][c] = 1  # C[1,k1]
        col_offset += 1
        
        # M2 = (A10+A11)*B00
        c = col_offset
        U[3][c] = 1  # A10
        U[4][c] = 1  # A11
        V[k0][c] = 1  # B[0,k0]
        W[k0*2+1][c] = 1  # C[1,k0]
        col_offset += 1
        
        # M3 = A00*(B10-B11)
        c = col_offset
        U[0][c] = 1  # A00
        V[8+k0][c] = 1  # B[1,k0]
        V[8+k1][c] = -1  # B[1,k1]
        W[k1*2][c] = 1  # C[0,k1]
        col_offset += 1
        
        # M4 = A11*(B01-B00)
        c = col_offset
        U[4][c] = 1  # A11
        V[k1][c] = 1  # B[0,k1]
        V[k0][c] = -1  # B[0,k0]
        W[k0*2][c] = 1  # C[0,k0]
        W[k0*2+1][c] = 1  # C[1,k0]
        col_offset += 1
        
        # M5 = (A00+A01)*B11
        c = col_offset
        U[0][c] = 1  # A00
        U[1][c] = 1  # A01
        V[8+k1][c] = 1  # B[1,k1]
        W[k0*2][c] = -1  # C[0,k0]
        W[k1*2][c] = 1  # C[0,k1]
        col_offset += 1
        
        # M6 = (A10-A00)*(B00+B01)
        c = col_offset
        U[3][c] = 1  # A10
        U[0][c] = -1  # A00
        V[k0][c] = 1  # B[0,k0]
        V[k1][c] = 1  # B[0,k1]
        W[k1*2][c] = -1  # C[0,k1]
        W[k0*2+1][c] = 1  # C[1,k0]
        col_offset += 1
        
        # M7 = (A01-A11)*(B10+B11)
        c = col_offset
        U[1][c] = 1  # A01
        U[4][c] = -1  # A11
        V[8+k0][c] = 1  # B[1,k0]
        V[8+k1][c] = 1  # B[1,k1]
        W[k0*2][c] = 1  # C[0,k0]
        W[k0*2+1][c] = -1  # C[1,k0]
        col_offset += 1
    
    # Now A_R @ B_R: A_R is column 2 of A (2x1), B_R is row 2 of B (1x8)
    # C[i,k] += A[i,2] * B[2,k]
    # U[2] and U[5] are A[0,2] and A[1,2]
    # V[16+k] is B[2,k]
    # W[k*2+i] is C[i,k]
    for k in range(8):
        c = col_offset
        U[2][c] = 1  # A[0,2]
        V[16+k][c] = 1  # B[2,k]
        W[k*2][c] = 1  # C[0,k]
        col_offset += 1
        
        c = col_offset
        U[5][c] = 1  # A[1,2]
        V[16+k][c] = 1  # B[2,k]
        W[k*2+1][c] = 1  # C[1,k]
        col_offset += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Try rank-44 construction first
    try:
        result = _construct_44()
        U, V, W = result["U"], result["V"], result["W"]
        n, m, p = DIMENSIONS
        if _verify(U, V, W, n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook rank-48
    return _construct_schoolbook()
# EVOLVE-BLOCK-END
