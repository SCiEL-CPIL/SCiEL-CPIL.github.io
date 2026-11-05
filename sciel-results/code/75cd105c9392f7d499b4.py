import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    rank = 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    def add_strassen_222(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        nonlocal col
        # M1 = (a00+a11)(b00+b11)
        U[A_row_off*m + A_col_off][col] = 1
        U[(A_row_off+1)*m + (A_col_off+1)][col] = 1
        V[B_row_off*p + B_col_off][col] = 1
        V[(B_row_off+1)*p + (B_col_off+1)][col] = 1
        W[(B_col_off)*n + C_row_off][col] = 1
        W[(B_col_off+1)*n + (C_row_off+1)][col] = 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[(A_row_off+1)*m + A_col_off][col] = 1
        U[(A_row_off+1)*m + (A_col_off+1)][col] = 1
        V[B_row_off*p + B_col_off][col] = 1
        W[(B_col_off)*n + (C_row_off+1)][col] = 1
        W[(B_col_off+1)*n + (C_row_off+1)][col] = -1
        col += 1
        
        # M3 = a00(b01-b11)
        U[A_row_off*m + A_col_off][col] = 1
        V[B_row_off*p + (B_col_off+1)][col] = 1
        V[(B_row_off+1)*p + (B_col_off+1)][col] = -1
        W[(B_col_off+1)*n + C_row_off][col] = 1
        W[(B_col_off+1)*n + (C_row_off+1)][col] = 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[(A_row_off+1)*m + (A_col_off+1)][col] = 1
        V[(B_row_off+1)*p + B_col_off][col] = 1
        V[B_row_off*p + B_col_off][col] = -1
        W[(B_col_off)*n + (C_row_off+1)][col] = 1
        W[(B_col_off+1)*n + (C_row_off+1)][col] = 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[A_row_off*m + A_col_off][col] = 1
        U[A_row_off*m + (A_col_off+1)][col] = 1
        V[(B_row_off+1)*p + (B_col_off+1)][col] = 1
        W[(B_col_off+1)*n + C_row_off][col] = -1
        W[(B_col_off+1)*n + (C_row_off+1)][col] = -1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[(A_row_off+1)*m + A_col_off][col] = 1
        U[A_row_off*m + A_col_off][col] = -1
        V[B_row_off*p + B_col_off][col] = 1
        V[B_row_off*p + (B_col_off+1)][col] = 1
        W[(B_col_off+1)*n + (C_row_off+1)][col] = 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[A_row_off*m + (A_col_off+1)][col] = 1
        U[(A_row_off+1)*m + (A_col_off+1)][col] = -1
        V[(B_row_off+1)*p + B_col_off][col] = 1
        V[(B_row_off+1)*p + (B_col_off+1)][col] = 1
        W[(B_col_off)*n + C_row_off][col] = 1
        col += 1
    
    def add_223(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        nonlocal col
        for i in range(2):
            for k in range(2):
                for j in range(3):
                    U[(A_row_off+i)*m + (A_col_off+k)][col] = 1
                    V[(B_row_off+k)*p + (B_col_off+j)][col] = 1
                    W[(B_col_off+j)*n + (C_row_off+i)][col] = 1
                    col += 1
    
    def add_123(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        nonlocal col
        for k in range(2):
            for j in range(3):
                U[A_row_off*m + (A_col_off+k)][col] = 1
                V[(B_row_off+k)*p + (B_col_off+j)][col] = 1
                W[(B_col_off+j)*n + C_row_off][col] = 1
                col += 1
    
    def add_122(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        nonlocal col
        for k in range(2):
            for j in range(2):
                U[A_row_off*m + (A_col_off+k)][col] = 1
                V[(B_row_off+k)*p + (B_col_off+j)][col] = 1
                W[(B_col_off+j)*n + C_row_off][col] = 1
                col += 1
    
    # Assemble blocks
    add_223(0, 0, 0, 0, 0, 0)
    add_223(0, 2, 2, 0, 0, 0)
    add_strassen_222(0, 0, 0, 3, 0, 3)
    add_strassen_222(0, 2, 2, 3, 0, 3)
    add_123(2, 0, 0, 0, 2, 0)
    add_123(2, 2, 2, 0, 2, 0)
    add_122(2, 0, 0, 3, 2, 3)
    add_122(2, 2, 2, 3, 2, 3)
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    if time.time() > deadline:
        pass  # We're done constructing, just return
    
    return {"U": U, "V": V, "W": W}
