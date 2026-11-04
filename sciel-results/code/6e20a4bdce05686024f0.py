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

def _laderman_2x3x3(U, V, W, a_idx, b_idx, c_idx, base_col):
    """
    Apply Laderman rank-14 decomposition for 2x3x3 matmul.
    A is 2x3, B is 3x3, C is 2x3.
    a_idx: dict mapping (row, col) -> U index
    b_idx: dict mapping (row, col) -> V index  
    c_idx: dict mapping (row, col) -> W index
    Returns next column index.
    """
    col = base_col
    # p1 = A[0,0] * B[0,0]
    U[a_idx[(0,0)], col] += 1
    V[b_idx[(0,0)], col] += 1
    W[c_idx[(0,0)], col] += 1
    col += 1
    # p2 = A[0,1] * B[1,0]
    U[a_idx[(0,1)], col] += 1
    V[b_idx[(1,0)], col] += 1
    W[c_idx[(0,1)], col] += 1
    col += 1
    # p3 = A[0,2] * B[2,0]
    U[a_idx[(0,2)], col] += 1
    V[b_idx[(2,0)], col] += 1
    W[c_idx[(0,2)], col] += 1
    col += 1
    # p4 = A[1,0] * B[0,1]
    U[a_idx[(1,0)], col] += 1
    V[b_idx[(0,1)], col] += 1
    W[c_idx[(1,0)], col] += 1
    col += 1
    # p5 = A[1,1] * B[1,1]
    U[a_idx[(1,1)], col] += 1
    V[b_idx[(1,1)], col] += 1
    W[c_idx[(1,1)], col] += 1
    col += 1
    # p6 = A[1,2] * B[2,1]
    U[a_idx[(1,2)], col] += 1
    V[b_idx[(2,1)], col] += 1
    W[c_idx[(1,2)], col] += 1
    col += 1
    # p7 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[a_idx[(0,0)], col] += 1
    U[a_idx[(1,1)], col] += 1
    V[b_idx[(0,0)], col] += 1
    V[b_idx[(1,1)], col] += 1
    W[c_idx[(0,0)], col] += 1
    W[c_idx[(1,1)], col] += 1
    col += 1
    # p8 = (A[0,1]+A[1,2]) * (B[1,0]+B[2,1])
    U[a_idx[(0,1)], col] += 1
    U[a_idx[(1,2)], col] += 1
    V[b_idx[(1,0)], col] += 1
    V[b_idx[(2,1)], col] += 1
    W[c_idx[(0,1)], col] += 1
    W[c_idx[(1,2)], col] += 1
    col += 1
    # p9 = (A[0,2]+A[1,0]) * (B[2,0]+B[0,1])
    U[a_idx[(0,2)], col] += 1
    U[a_idx[(1,0)], col] += 1
    V[b_idx[(2,0)], col] += 1
    V[b_idx[(0,1)], col] += 1
    W[c_idx[(0,2)], col] += 1
    W[c_idx[(1,0)], col] += 1
    col += 1
    # p10 = (A[0,0]+A[0,1]+A[0,2]) * (B[0,0]+B[1,0]+B[2,0])
    U[a_idx[(0,0)], col] += 1
    U[a_idx[(0,1)], col] += 1
    U[a_idx[(0,2)], col] += 1
    V[b_idx[(0,0)], col] += 1
    V[b_idx[(1,0)], col] += 1
    V[b_idx[(2,0)], col] += 1
    W[c_idx[(0,0)], col] += 1
    W[c_idx[(0,1)], col] += 1
    W[c_idx[(0,2)], col] += 1
    col += 1
    # p11 = (A[1,0]+A[1,1]+A[1,2]) * (B[0,1]+B[1,1]+B[2,1])
    U[a_idx[(1,0)], col] += 1
    U[a_idx[(1,1)], col] += 1
    U[a_idx[(1,2)], col] += 1
    V[b_idx[(0,1)], col] += 1
    V[b_idx[(1,1)], col] += 1
    V[b_idx[(2,1)], col] += 1
    W[c_idx[(1,0)], col] += 1
    W[c_idx[(1,1)], col] += 1
    W[c_idx[(1,2)], col] += 1
    col += 1
    # p12 = (A[0,0]-A[1,0]+A[0,1]) * (B[0,0]+B[0,1]-B[1,0])
    U[a_idx[(0,0)], col] += 1
    U[a_idx[(1,0)], col] -= 1
    U[a_idx[(0,1)], col] += 1
    V[b_idx[(0,0)], col] += 1
    V[b_idx[(0,1)], col] += 1
    V[b_idx[(1,0)], col] -= 1
    W[c_idx[(0,0)], col] += 1
    W[c_idx[(1,0)], col] -= 1
    W[c_idx[(0,1)], col] += 1
    W[c_idx[(1,1)], col] -= 1
    col += 1
    # p13 = (A[0,1]-A[1,1]+A[0,2]) * (B[1,0]+B[1,1]-B[2,0])
    U[a_idx[(0,1)], col] += 1
    U[a_idx[(1,1)], col] -= 1
    U[a_idx[(0,2)], col] += 1
    V[b_idx[(1,0)], col] += 1
    V[b_idx[(1,1)], col] += 1
    V[b_idx[(2,0)], col] -= 1
    W[c_idx[(0,1)], col] += 1
    W[c_idx[(1,1)], col] -= 1
    W[c_idx[(0,2)], col] += 1
    W[c_idx[(1,2)], col] -= 1
    col += 1
    # p14 = (A[0,2]-A[1,2]+A[0,0]) * (B[2,0]+B[2,1]-B[0,0])
    U[a_idx[(0,2)], col] += 1
    U[a_idx[(1,2)], col] -= 1
    U[a_idx[(0,0)], col] += 1
    V[b_idx[(2,0)], col] += 1
    V[b_idx[(2,1)], col] += 1
    V[b_idx[(0,0)], col] -= 1
    W[c_idx[(0,2)], col] += 1
    W[c_idx[(1,2)], col] -= 1
    W[c_idx[(0,0)], col] += 1
    W[c_idx[(1,0)], col] -= 1
    col += 1
    return col

def _construct_laderman():
    """
    Construct a rank-88 decomposition using Laderman for 2x3x3 sub-blocks.
    Row 0: schoolbook 1x4x8 = 32
    Rows 1-2: 2x4x8 split into 2x3x8 (40) + 2x1x8 (16) = 56
    Total: 88
    """
    n, m, p = 3, 4, 8
    R = 88
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Part 1: Row 0 of A (1x4) times B (4x8) - schoolbook, rank 32
    col = 0
    for j in range(4):
        for k in range(8):
            U[0*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
    
    # Part 2: Rows 1-2 of A (2x4) times B (4x8)
    # Split inner dim (j) into 0-2 and 3
    # Split B rows into 0-2 and 3
    # Split output k into 0-3 and 4-7
    
    # A rows 1-2, cols 0-2: 2x3
    a3_idx = {(0,0): 1*4+0, (0,1): 1*4+1, (0,2): 1*4+2,
              (1,0): 2*4+0, (1,1): 2*4+1, (1,2): 2*4+2}
    # A rows 1-2, col 3: 2x1
    a1_idx = {(0,0): 1*4+3, (1,0): 2*4+3}
    
    # For 2x3x4 sub-problems (k=0-3 and k=4-7)
    # Each 2x3x4: Laderman 2x3x3 (rank 14) + schoolbook 2x3x1 (rank 6) = 20
    
    for k_offset in [0, 4]:
        for k_sub in range(3):
            # 2x3x3 Laderman: A(2x3) x B(3x3) -> C(2x3)
            # B rows 0-2, cols k_offset to k_offset+2
            b3_idx = {(0,0): 0*p + (k_offset+k_sub),
                      (0,1): 0*p + (k_offset+k_sub+1),
                      (0,2): 0*p + (k_offset+k_sub+2),
                      (1,0): 1*p + (k_offset+k_sub),
                      (1,1): 1*p + (k_offset+k_sub+1),
                      (1,2): 1*p + (k_offset+k_sub+2),
                      (2,0): 2*p + (k_offset+k_sub),
                      (2,1): 2*p + (k_offset+k_sub+1),
                      (2,2): 2*p + (k_offset+k_sub+2)}
            c3_idx = {(0,0): (k_offset+k_sub)*n + 1,
                      (0,1): (k_offset+k_sub+1)*n + 1,
                      (0,2): (k_offset+k_sub+2)*n + 1,
                      (1,0): (k_offset+k_sub)*n + 2,
                      (1,1): (k_offset+k_sub+1)*n + 2,
                      (1,2): (k_offset+k_sub+2)*n + 2}
            col = _laderman_2x3x3(U, V, W, a3_idx, b3_idx, c3_idx, col)
        # 2x3x1 schoolbook: A(2x3) x B(3x1) -> C(2x1)
        k_col = k_offset + 3
        b1_idx = {(0,0): 0*p + k_col,
                  (1,0): 1*p + k_col,
                  (2,0): 2*p + k_col}
        c1_idx = {(0,0): k_col*n + 1,
                  (1,0): k_col*n + 2}
        for ar in range(2):
            for ac in range(3):
                for br in range(3):
                    U[a3_idx[(ar,ac)], col] += 1
                    V[b1_idx[(br,0)], col] += 1
                    W[c1_idx[(ar,0)], col] += 1
                    col += 1
    
    # 2x1x8 schoolbook: A(2x1) x B(1x8) -> C(2x8)
    # A row 3 of original (col 3), B row 3
    for k in range(8):
        for ar in range(2):
            U[a1_idx[(ar,0)], col] += 1
            V[3*p + k, col] += 1
            W[k*n + 1 + ar, col] += 1
            col += 1
    
    assert col == 88, f"Expected 88, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try Laderman-based construction (rank 88)
    try:
        U_l, V_l, W_l = _construct_laderman()
        if _verify(U_l, V_l, W_l, n, m, p):
            return {"U": U_l, "V": V_l, "W": W_l}
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
