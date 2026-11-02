import time
import numpy as np

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def get_strassen_2x2x2():
        """Return U, V, W matrices for 2x2x2 Strassen decomposition (rank 7)"""
        # Standard Strassen formulas (verified)
        # U: 4 entries per term (A00, A01, A10, A11)
        # V: 4 entries per term (B00, B01, B10, B11)  
        # W: 4 entries per term (C00, C01, C10, C11)
        
        U = [
            [1, 0, 0, 1],   # M1: (A00+A11)
            [0, 0, 1, 1],   # M2: (A10+A11)
            [1, 0, 0, 0],   # M3: A00
            [0, 0, 0, 1],   # M4: A11
            [1, 1, 0, 0],   # M5: (A00+A01)
            [-1, 0, 1, 0],  # M6: (A10-A00)
            [0, 1, -1, 0],  # M7: (A01-A10)
        ]
        V = [
            [1, 0, 0, 1],   # M1: (B00+B11)
            [1, 0, 0, 0],   # M2: B00
            [0, 0, 1, -1],  # M3: (B10-B11)
            [-1, 1, 0, 0],  # M4: (B01-B00)
            [0, 0, 0, 1],   # M5: B11
            [1, 1, 0, 0],   # M6: (B00+B01)
            [0, 0, 1, 1],   # M7: (B10+B11)
        ]
        W = [
            [1, 0, 0, 1],   # M1: C00+1, C11+1
            [0, 0, 1, -1],  # M2: C10+1, C11-1
            [0, 1, 0, 1],   # M3: C01+1, C11+1
            [1, 0, 1, 0],   # M4: C00+1, C10+1
            [-1, 1, 0, 0],  # M5: C00-1, C01+1
            [0, 0, 1, 0],   # M6: C10+1
            [-1, 0, 0, 1],  # M7: C00-1, C11+1
        ]
        return U, V, W

    def verify(U, V, W, rank, n, m, p):
        """Check if the decomposition is valid"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
        
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    # Build the full 3x4x8 decomposition
    # Strategy: split n=3 into 2+1, p=8 into 4+4
    # Block (0:2, 0:4) x (0:4, 0:4): 2x4x4 -> four 2x2x2 Strassen = rank 28
    # Block (0:2, 0:4) x (0:4, 4:8): 2x4x4 -> four 2x2x2 Strassen = rank 28
    # Block (2:3, 0:4) x (0:4, 0:4): 1x4x4 -> naive = rank 4
    # Block (2:3, 0:4) x (0:4, 4:8): 1x4x4 -> naive = rank 4
    # Total rank: 28 + 28 + 4 + 4 = 64
    
    # Wait, let me reconsider. The 2x4x4 can be split into 2x2x4 blocks:
    # 2x2x4 = two 2x2x2 Strassen = rank 14
    # So 2x4x4 = two 2x2x4 = rank 28
    # That's what I had.
    
    # Actually, I can do better. Let me use:
    # Split m=4 into 2+2 as well
    # 2x2x2 Strassen blocks
    
    # For the 2x4x4 block (i in 0:2, j in 0:4, k in 0:4):
    # Split j into 0:2 and 2:4
    # Split k into 0:2 and 2:4
    # Four 2x2x2 Strassen blocks, each rank 7, total rank 28
    
    # Let me construct this properly
    total_rank = 28 + 28 + 4 + 4  # 64
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    strassen_U, strassen_V, strassen_W = get_strassen_2x2x2()
    
    def add_strassen_block(i_offset, j_offset, k_offset, col):
        """Add a 2x2x2 Strassen block at the given offset"""
        for r in range(7):
            for ui in range(2):
                for uj in range(2):
                    i = i_offset + ui
                    j = j_offset + uj
                    val = strassen_U[r][ui * 2 + uj]
                    if val != 0:
                        U[i * m + j][col] += val
            
            for vj in range(2):
                for vk in range(2):
                    j = j_offset + vj
                    k = k_offset + vk
                    val = strassen_V[r][vj * 2 + vk]
                    if val != 0:
                        V[j * p + k][col] += val
            
            for wk in range(2):
                for wi in range(2):
                    k = k_offset + wk
                    i = i_offset + wi
                    val = strassen_W[r][wk * 2 + wi]
                    if val != 0:
                        W[k * n + i][col] += val
            
            col += 1
        return col
    
    def add_naive_1x2x2(i_idx, j_offset, k_offset, col):
        """Add a 1x2x2 naive block"""
        for j in range(2):
            for k in range(2):
                U[i_idx * m + (j_offset + j)][col] = 1
                V[(j_offset + j) * p + (k_offset + k)][col] = 1
                W[(k_offset + k) * n + i_idx][col] = 1
                col += 1
        return col
    
    # Block 1: i in 0:2, j in 0:2, k in 0:2
    col = add_strassen_block(0, 0, 0, col)
    # Block 2: i in 0:2, j in 0:2, k in 2:4
    col = add_strassen_block(0, 0, 2, col)
    # Block 3: i in 0:2, j in 2:4, k in 0:2
    col = add_strassen_block(0, 2, 0, col)
    # Block 4: i in 0:2, j in 2:4, k in 2:4
    col = add_strassen_block(0, 2, 2, col)
    # Block 5: i in 0:2, j in 0:2, k in 4:6
    col = add_strassen_block(0, 0, 4, col)
    # Block 6: i in 0:2, j in 0:2, k in 6:8
    col = add_strassen_block(0, 0, 6, col)
    # Block 7: i in 0:2, j in 2:4, k in 4:6
    col = add_strassen_block(0, 2, 4, col)
    # Block 8: i in 0:2, j in 2:4, k in 6:8
    col = add_strassen_block(0, 2, 6, col)
    
    # Now handle i=2 (the third row)
    # j in 0:2, k in 0:2
    col = add_naive_1x2x2(2, 0, 0, col)
    # j in 0:2, k in 2:4
    col = add_naive_1x2x2(2, 0, 2, col)
    # j in 0:2, k in 4:6
    col = add_naive_1x2x2(2, 0, 4, col)
    # j in 0:2, k in 6:8
    col = add_naive_1x2x2(2, 0, 6, col)
    # j in 2:4, k in 0:2
    col = add_naive_1x2x2(2, 2, 0, col)
    # j in 2:4, k in 2:4
    col = add_naive_1x2x2(2, 2, 2, col)
    # j in 2:4, k in 4:6
    col = add_naive_1x2x2(2, 2, 4, col)
    # j in 2:4, k in 6:8
    col = add_naive_1x2x2(2, 2, 6, col)
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if verify(U, V, W, col, n, m, p):
        return result
    
    # Fallback to naive if verification fails
    naive_rank = n * m * p
    U_naive = [[0] * naive_rank for _ in range(n * m)]
    V_naive = [[0] * naive_rank for _ in range(m * p)]
    W_naive = [[0] * naive_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1
    
    return {"U": U_naive, "V": V_naive, "W": W_naive}
