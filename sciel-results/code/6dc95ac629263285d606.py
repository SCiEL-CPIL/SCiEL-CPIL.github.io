import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a block decomposition
    # a_offset: row offset in A (0 or 2)
    # b_offset: row offset in B (0 or 2)
    # c_offset: col offset in C (0 to 6)
    # type: 'strassen' or 'schoolbook_2x2x1'
    # b_width: 2 for strassen, 1 for schoolbook
    
    col = 0
    
    def add_strassen(a_off, b_off, c_off):
        nonlocal col
        # U indices: a_off*2 + 0..3 (A is 2x2, flat index i*2+j? No, A is 2x4. 
        # In our local 2x2 block, rows are 0,1. Cols are a_off, a_off+1.
        # U row index: i * m + j = i * 4 + j.
        # So U rows: i*4 + a_off, i*4 + a_off+1.
        
        # V indices: b_off*7 + 0..1 (B is 4x7. Rows b_off, b_off+1. Cols c_off, c_off+1).
        # V row index: j * p + k = j * 7 + k.
        # So V rows: b_off*7 + c_off, b_off*7 + c_off+1, (b_off+1)*7 + c_off, (b_off+1)*7 + c_off+1.
        
        # W indices: (c_off)*n + i, (c_off+1)*n + i.
        # W row index: k * n + i = k * 2 + i.
        # So W rows: c_off*2 + 0, c_off*2 + 1, (c_off+1)*2 + 0, (c_off+1)*2 + 1.
        
        # Strassen factors
        # U (4x7)
        u_0 = [1, 0, 1, 0, 1, 0, 0]
        u_1 = [0, 1, 0, 0, 1, -1, 1]
        u_2 = [0, 0, 1, 1, 0, 1, 0]
        u_3 = [1, 0, 0, 1, 0, 0, -1]
        
        # V (4x7)
        v_0 = [1, 0, 0, 1]
        v_1 = [0, 1, 0, 0]
        v_2 = [1, 1, 0, 0]
        v_3 = [1, 0, -1, 0]
        v_4 = [0, 0, 1, 1]
        v_5 = [0, 1, 0, -1]
        v_6 = [1, 0, 1, 0]
        v_7 = [0, 1, 1, 0]
        
        # Wait, I need to map my previous derivation to these vectors.
        # My previous derivation:
        # M1: U=[1,0,1,0], V=[1,0,0,1] (B00, B11)
        # M2: U=[0,0,1,1], V=[1,0,0,0] (B00)
        # M3: U=[1,0,0,0], V=[0,1,0,-1] (B01, B11)
        # M4: U=[0,0,0,1], V=[-1,0,1,0] (B00, B10)
        # M5: U=[1,1,0,0], V=[0,0,0,1] (B11)
        # M6: U=[-1,0,1,0], V=[1,1,0,0] (B00, B01)
        # M7: U=[0,1,0,-1], V=[0,0,1,1] (B10, B11)
        
        # Let's use these exact vectors.
        U_cols = [
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        V_cols = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        W_cols = [
            [1, 0, 0, 1], # C00, C11
            [0, 0, 1, 0], # C10
            [0, 1, 0, 1], # C01, C11
            [1, 0, 1, 0], # C00, C10
            [-1, 1, 0, 0], # C00, C01
            [0, 0, 0, 1], # C11
            [1, 0, 0, 0]  # C00
        ]
        
        # Map to global indices
        # U rows: i=0,1. j=a_off, a_off+1.
        u_rows = [a_off, a_off+1, 4+a_off, 4+a_off+1]
        # V rows: j=b_off, b_off+1. k=c_off, c_off+1.
        v_rows = [b_off*7 + c_off, b_off*7 + c_off+1, (b_off+1)*7 + c_off, (b_off+1)*7 + c_off+1]
        # W rows: k=c_off, c_off+1. i=0,1.
        w_rows = [c_off*2, c_off*2+1, (c_off+1)*2, (c_off+1)*2+1]
        
        for r in range(7):
            for i in range(4):
                U[u_rows[i]][col] += U_cols[r][i]
                V[v_rows[i]][col] += V_cols[r][i]
                W[w_rows[i]][col] += W_cols[r][i]
            col += 1

    def add_schoolbook_2x2x1(a_off, b_off, c_off):
        nonlocal col
        # A 2x2, B 2x1, C 2x1
        # U rows: a_off, a_off+1, 4+a_off, 4+a_off+1
        # V rows: b_off*7+c_off, (b_off+1)*7+c_off
        # W rows: c_off*2, c_off*2+1
        
        u_rows = [a_off, a_off+1, 4+a_off, 4+a_off+1]
        v_rows = [b_off*7+c_off, (b_off+1)*7+c_off]
        w_rows = [c_off*2, c_off*2+1]
        
        # Terms:
        # 0: A0 B0 C0
        # 1: A1 B1 C0
        # 2: A2 B0 C1
        # 3: A3 B1 C1
        
        U_cols = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        V_cols = [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1]
        ]
        W_cols = [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1]
        ]
        
        for r in range(4):
            for i in range(4):
                U[u_rows[i]][col] += U_cols[r][i]
            for i in range(2):
                V[v_rows[i]][col] += V_cols[r][i]
                W[w_rows[i]][col] += W_cols[r][i]
            col += 1

    # Block L: A cols 0,1. B rows 0,1.
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    add_schoolbook_2x2x1(0, 0, 6)
    
    # Block R: A cols 2,3. B rows 2,3.
    add_strassen(2, 2, 0)
    add_strassen(2, 2, 2)
    add_strassen(2, 2, 4)
    add_schoolbook_2x2x1(2, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
