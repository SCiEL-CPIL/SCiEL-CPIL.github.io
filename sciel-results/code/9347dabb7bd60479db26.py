"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    # A: 2x4, flat index = i*m + j = i*4 + j
    # B: 4x4, flat index = j*p + k = j*4 + k  
    # C: 2x4, W flat index = k*n + i = k*2 + i
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]   # 8 x 28
    V = [[0]*rank for _ in range(m*p)]   # 16 x 28
    W = [[0]*rank for _ in range(p*n)]   # 8 x 28
    
    def do_strassen(col_start, a_rows, a_cols, b_rows, b_cols, w_z00, w_z01, w_z10, w_z11):
        """Add 7 Strassen rank-1 terms starting at col_start."""
        x00 = a_rows[0]*m + a_cols[0]
        x01 = a_rows[0]*m + a_cols[1]
        x10 = a_rows[1]*m + a_cols[0]
        x11 = a_rows[1]*m + a_cols[1]
        
        y00 = b_rows[0]*p + b_cols[0]
        y01 = b_rows[0]*p + b_cols[1]
        y10 = b_rows[1]*p + b_cols[0]
        y11 = b_rows[1]*p + b_cols[1]
        
        col = col_start
        # M1 = (X00+X11)(Y00+Y11) -> Z00+=M1, Z11+=M1
        for idx in [x00, x11]: U[idx][col] += 1
        for idx in [y00, y11]: V[idx][col] += 1
        W[w_z00][col] += 1; W[w_z11][col] += 1
        col += 1
        # M2 = (X10+X11)Y00 -> Z10+=M2, Z11-=M2
        for idx in [x10, x11]: U[idx][col] += 1
        V[y00][col] += 1
        W[w_z10][col] += 1; W[w_z11][col] -= 1
        col += 1
        # M3 = X00(Y01-Y11) -> Z01+=M3, Z11+=M3
        U[x00][col] += 1
        V[y01][col] += 1; V[y11][col] -= 1
        W[w_z01][col] += 1; W[w_z11][col] += 1
        col += 1
        # M4 = X11(Y10-Y00) -> Z00+=M4, Z10+=M4
        U[x11][col] += 1
        V[y10][col] += 1; V[y00][col] -= 1
        W[w_z00][col] += 1; W[w_z10][col] += 1
        col += 1
        # M5 = (X00+X01)Y11 -> Z00-=M5, Z01+=M5
        for idx in [x00, x01]: U[idx][col] += 1
        V[y11][col] += 1
        W[w_z00][col] -= 1; W[w_z01][col] += 1
        col += 1
        # M6 = (X10-X00)(Y00+Y01) -> Z11+=M6
        U[x10][col] += 1; U[x00][col] -= 1
        V[y00][col] += 1; V[y01][col] += 1
        W[w_z11][col] += 1
        col += 1
        # M7 = (X01-X11)(Y10+Y11) -> Z00+=M7
        U[x01][col] += 1; U[x11][col] -= 1
        V[y10][col] += 1; V[y11][col] += 1
        W[w_z00][col] += 1
        col += 1
        return col
    
    col = 0
    # 1: A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]
    # W: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    col = do_strassen(col, [0,1], [0,1], [0,1], [0,1], 0, 2, 1, 3)
    # 2: A[0:2,2:4] x B[2:4,0:2] -> C[0:2,0:2]
    col = do_strassen(col, [0,1], [2,3], [2,3], [0,1], 0, 2, 1, 3)
    # 3: A[0:2,0:2] x B[0:2,2:4] -> C[0:2,2:4]
    # W: C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    col = do_strassen(col, [0,1], [0,1], [0,1], [2,3], 4, 6, 5, 7)
    # 4: A[0:2,2:4] x B[2:4,2:4] -> C[0:2,2:4]
    col = do_strassen(col, [0,1], [2,3], [2,3], [2,3], 4, 6, 5, 7)
    
    assert col == 28, f"Expected 28, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
