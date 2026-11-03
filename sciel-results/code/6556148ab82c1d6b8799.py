"""Matrix multiplication 2x4 by 4x5 tensor decomposition with rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, rank):
    """Verify exact decomposition using numpy for speed."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _strassen_2x2x2():
    """
    Correct Strassen decomposition for 2x2x2.
    U (4x7), V (4x7), W (4x7).
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    # Corrected W matrix
    W = [
        [1, 0, 0, 1, -1, 0, -1], # c00 = M1 + M4 - M5 - M7
        [0, 0, 1, 0, 1, 0, 0],   # c01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # c10 = M2 + M4
        [1, -1, 1, 0, 0, 0, 1]   # c11 = M1 - M2 + M3 + M7
    ]
    
    # Verify
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.dot(U_np[i*2+j], V_np[j*2+k] * W_np[k*2+i])
                if abs(s - 1.0) > 1e-9:
                    return None
    return U, V, W

def _block_2x2x1():
    """
    Decomposition for 2x2x1.
    A (2x2), B (2x1), C (2x1).
    Rank 4 (schoolbook).
    """
    U = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]
    V = [
        [1,0,1,0],
        [0,1,0,1]
    ]
    W = [
        [1,1,0,0],
        [0,0,1,1]
    ]
    return U, V, W

def construct():
    n, m, p = 2, 4, 5
    # Total rank 36
    total_rank = 36
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    sU, sV, sW = _strassen_2x2x2()
    if sU is None:
        return _schoolbook(n, m, p)
    bU, bV, bW = _block_2x2x1()
    
    col_offset = 0
    
    for j_block in [0, 2]:
        # 1. 2x2x2 for k=0,1
        for c in range(7):
            for li in range(2):
                gi = li
                U[gi*4+j_block][col_offset+c] = sU[li*2+0][c] # lj=0, gj=j_block
                # Note: U index is i*m+j. m=4.
                # sU index is li*2+lj.
                # We need to map lj=0 to gj=j_block and lj=1 to gj=j_block+1.
                # So for each li, we need to fill two U rows? No.
                # sU is 4x7. Rows are (0,0), (0,1), (1,0), (1,1).
                # U rows are i*4+j.
                # For a fixed j_block, we use j=j_block and j=j_block+1.
                
            # Let's do it properly
            # Loop over local indices
            pass
            
        # Proper mapping
        # U[gi*4+gj, col] = sU[li*2+lj, c]
        # where gj = j_block + lj
        
        # Block 1: k=0,1
        for c in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = li
                    gj = j_block + lj
                    U[gi*4+gj][col_offset+c] = sU[li*2+lj][c]
            for lj in range(2):
                gj = j_block + lj
                for lk in range(2):
                    gk = lk
                    V[gj*5+gk][col_offset+c] = sV[lj*2+lk][c]
            for lk in range(2):
                gk = lk
                for li in range(2):
                    gi = li
                    W[gk*2+gi][col_offset+c] = sW[lk*2+li][c]
        col_offset += 7
        
        # Block 2: k=2,3
        for c in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = li
                    gj = j_block + lj
                    U[gi*4+gj][col_offset+c] = sU[li*2+lj][c]
            for lj in range(2):
                gj = j_block + lj
                for lk in range(2):
                    gk = 2 + lk
                    V[gj*5+gk][col_offset+c] = sV[lj*2+lk][c]
            for lk in range(2):
                gk = 2 + lk
                for li in range(2):
                    gi = li
                    W[gk*2+gi][col_offset+c] = sW[lk*2+li][c]
        col_offset += 7
        
        # Block 3: k=4
        for c in range(4):
            for li in range(2):
                for lj in range(2):
                    gi = li
                    gj = j_block + lj
                    U[gi*4+gj][col_offset+c] = bU[li*2+lj][c]
            for lj in range(2):
                gj = j_block + lj
                lk = 0
                gk = 4
                V[gj*5+gk][col_offset+c] = bV[lj*2+lk][c]
            lk = 0
            gk = 4
            for li in range(2):
                gi = li
                W[gk*2+gi][col_offset+c] = bW[lk*2+li][c]
        col_offset += 4

    if _verify(U, V, W, n, m, p, total_rank):
        return {"U": U, "V": V, "W": W}
    else:
        return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
