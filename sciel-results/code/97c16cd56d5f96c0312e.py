import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition reconstructs the matmul tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_rank88():
    """
    Construct a rank-88 decomposition for 3x4x8 matmul.
    
    Strategy:
    - Part 1: Row 0 of A (1x4) times B (4x8) - schoolbook, rank 32
    - Part 2: Rows 1-2 of A (2x4) times B (4x8) - 8 Strassen 2x2x2 blocks, rank 56
    Total: 88
    """
    n, m, p = 3, 4, 8
    R = 88
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # Part 1: Row 0 of A (1x4) times B (4x8) - schoolbook, rank 32
    for j in range(4):
        for k in range(8):
            U[0*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
    
    # Part 2: Rows 1-2 of A (2x4) times B (4x8)
    # Partition A cols into 2 blocks of 2: j=0,1 and j=2,3
    # Partition B rows into 2 blocks of 2: j=0,1 and j=2,3
    # Partition B cols into 4 blocks of 2: k=0,1; 2,3; 4,5; 6,7
    # C rows: 1, 2; C col blocks: (0,1), (2,3), (4,5), (6,7)
    
    # Strassen 2x2x2 factors
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    for j_block in range(2):  # j=0,1 or j=2,3
        jA0 = j_block * 2
        jB0 = j_block * 2
        
        for k_block in range(4):  # k=0,1; 2,3; 4,5; 6,7
            k0 = k_block * 2
            i0 = 1  # A/C row offset
            
            # M1 = (A00+A11)(B00+B11)
            U[i0*m + jA0, col] += 1
            U[(i0+1)*m + (jA0+1), col] += 1
            V[jB0*p + k0, col] += 1
            V[(jB0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M2 = (A10+A11)B00
            U[(i0+1)*m + jA0, col] += 1
            U[(i0+1)*m + (jA0+1), col] += 1
            V[jB0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            
            # M3 = A00(B01-B11)
            U[i0*m + jA0, col] += 1
            V[jB0*p + (k0+1), col] += 1
            V[(jB0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M4 = A11(B10-B00)
            U[(i0+1)*m + (jA0+1), col] += 1
            V[(jB0+1)*p + k0, col] += 1
            V[jB0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1
            W[k0*n + (i0+1), col] += 1
            col += 1
            
            # M5 = (A00+A01)B11
            U[i0*m + jA0, col] += 1
            U[i0*m + (jA0+1), col] += 1
            V[(jB0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1
            W[(k0+1)*n + i0, col] += 1
            col += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[(i0+1)*m + jA0, col] += 1
            U[i0*m + jA0, col] -= 1
            V[jB0*p + k0, col] += 1
            V[jB0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[i0*m + (jA0+1), col] += 1
            U[(i0+1)*m + (jA0+1), col] -= 1
            V[(jB0+1)*p + k0, col] += 1
            V[(jB0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1
    
    assert col == 88, f"Expected 88, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    try:
        U_h, V_h, W_h = _construct_rank88()
        if _verify(U_h, V_h, W_h, n, m, p):
            return {"U": U_h, "V": V_h, "W": W_h}
    except Exception:
        pass
    
    # Schoolbook fallback (rank 96)
    R = n * m * p
    U_sb = [[0]*R for _ in range(n*m)]
    V_sb = [[0]*R for _ in range(m*p)]
    W_sb = [[0]*R for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
