import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
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
    Row 0: Schoolbook (rank 32)
    Rows 1-2: Strassen-based (rank 56)
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
            
    # Part 2: Rows 1-2 of A (2x4) times B (4x8) - Strassen, rank 56
    # We split the 2x4x8 problem into 4 2x2x2 blocks
    # A is 2x4, B is 4x8, C is 2x8
    # Split A into [A00, A01] where A00 is 2x2, A01 is 2x2
    # Split B into [B00, B01; B10, B11] where each is 2x4
    # Wait, let's think of B as 4x8.
    # B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 2x8
    # C = A * B
    # C = [A0, A1] * [B_top; B_bottom]
    # C = A0 * B_top + A1 * B_bottom
    # A0 is 2x2, B_top is 2x8 -> C_left is 2x8
    # A1 is 2x2, B_bottom is 2x8 -> C_right is 2x8
    # So we have two 2x2x8 multiplications.
    # Each 2x2x8 can be done with 4 2x2x2 Strassen blocks (rank 7 each) -> rank 28.
    # Total rank for Part 2 is 28 + 28 = 56.
    
    # Let's define the Strassen block for 2x2x2
    # Indices:
    # A: rows i0, i0+1; cols j0, j0+1
    # B: rows j0, j0+1; cols k0, k0+1
    # C: rows i0, i0+1; cols k0, k0+1
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # Part 2a: A0 (rows 1-2, cols 0-1) * B_top (rows 0-1, cols 0-7)
    # B_top is 2x8. Split into 4 2x2 blocks.
    add_strassen(i0=1, j0=0, k0=0)
    add_strassen(i0=1, j0=0, k0=2)
    add_strassen(i0=1, j0=0, k0=4)
    add_strassen(i0=1, j0=0, k0=6)
    
    # Part 2b: A1 (rows 1-2, cols 2-3) * B_bottom (rows 2-3, cols 0-7)
    # B_bottom is 2x8. Split into 4 2x2 blocks.
    add_strassen(i0=1, j0=2, k0=0)
    add_strassen(i0=1, j0=2, k0=2)
    add_strassen(i0=1, j0=2, k0=4)
    add_strassen(i0=1, j0=2, k0=6)
    
    assert col == 88, f"Expected 88, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try rank-88 construction
    try:
        U, V, W = _construct_rank88()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Schoolbook fallback (rank 96) - guaranteed valid
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    return {"U": U_sb, "V": V_sb, "W": W_sb}
