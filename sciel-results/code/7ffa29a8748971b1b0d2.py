DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Try rank 15 first
    rank = 15
    U = [[0] * rank for _ in range(n * m)]  # 6 x 15
    V = [[0] * rank for _ in range(m * p)]  # 9 x 15
    W = [[0] * rank for _ in range(p * n)]  # 6 x 15
    
    def set_entry(mat, i, j, val):
        mat[i][j] += val
    
    # Strassen for 2x2 block (cols 0-6)
    # A: 0=A11, 1=A12, 2=A13, 3=A21, 4=A22, 5=A23
    # B: 0=B11, 1=B12, 2=B13, 3=B21, 4=B22, 5=B23, 6=B31, 7=B32, 8=B33
    # C (W index): C[row,col] -> col*n+row
    # C11=0, C12=2, C13=4, C21=1, C22=3, C23=5
    
    # Col 0: M1 = (A11+A22)(B11+B22)
    set_entry(U, 0, 0, 1); set_entry(U, 4, 0, 1)
    set_entry(V, 0, 0, 1); set_entry(V, 4, 0, 1)
    set_entry(W, 0, 0, 1); set_entry(W, 3, 0, 1)
    
    # Col 1: M2 = (A21+A22)B11
    set_entry(U, 3, 1, 1); set_entry(U, 4, 1, 1)
    set_entry(V, 0, 1, 1)
    set_entry(W, 1, 1, 1); set_entry(W, 3, 1, -1)
    
    # Col 2: M3 = A11(B12-B22)
    set_entry(U, 0, 2, 1)
    set_entry(V, 1, 2, 1); set_entry(V, 4, 2, -1)
    set_entry(W, 2, 2, 1); set_entry(W, 3, 2, 1)
    
    # Col 3: M4 = A22(B21-B11)
    set_entry(U, 4, 3, 1)
    set_entry(V, 3, 3, 1); set_entry(V, 0, 3, -1)
    set_entry(W, 0, 3, 1); set_entry(W, 1, 3, 1)
    
    # Col 4: M5 = (A11+A12)B22
    set_entry(U, 0, 4, 1); set_entry(U, 1, 4, 1)
    set_entry(V, 4, 4, 1)
    set_entry(W, 0, 4, -1); set_entry(W, 2, 4, 1)
    
    # Col 5: M6 = (A21-A11)(B11+B12)
    set_entry(U, 3, 5, 1); set_entry(U, 0, 5, -1)
    set_entry(V, 0, 5, 1); set_entry(V, 1, 5, 1)
    set_entry(W, 3, 5, 1)
    
    # Col 6: M7 = (A12-A22)(B21+B22)
    set_entry(U, 1, 6, 1); set_entry(U, 4, 6, -1)
    set_entry(V, 3, 6, 1); set_entry(V, 4, 6, 1)
    set_entry(W, 0, 6, 1)
    
    # Boundary terms (cols 7-14)
    # Col 7: A13*(B31+B33) -> C11 + C13
    set_entry(U, 2, 7, 1)
    set_entry(V, 6, 7, 1); set_entry(V, 8, 7, 1)
    set_entry(W, 0, 7, 1); set_entry(W, 4, 7, 1)
    
    # Col 8: A13*B32 -> C12
    set_entry(U, 2, 8, 1)
    set_entry(V, 7, 8, 1)
    set_entry(W, 2, 8, 1)
    
    # Col 9: A23*(B31+B33) -> C21 + C23
    set_entry(U, 5, 9, 1)
    set_entry(V, 6, 9, 1); set_entry(V, 8, 9, 1)
    set_entry(W, 1, 9, 1); set_entry(W, 5, 9, 1)
    
    # Col 10: A23*B32 -> C22
    set_entry(U, 5, 10, 1)
    set_entry(V, 7, 10, 1)
    set_entry(W, 3, 10, 1)
    
    # Col 11: A11*B13 -> C13
    set_entry(U, 0, 11, 1)
    set_entry(V, 2, 11, 1)
    set_entry(W, 4, 11, 1)
    
    # Col 12: A21*B13 -> C23
    set_entry(U, 3, 12, 1)
    set_entry(V, 2, 12, 1)
    set_entry(W, 5, 12, 1)
    
    # Col 13: A12*B23 -> C13
    set_entry(U, 1, 13, 1)
    set_entry(V, 5, 13, 1)
    set_entry(W, 4, 13, 1)
    
    # Col 14: A22*B23 -> C23
    set_entry(U, 4, 14, 1)
    set_entry(V, 5, 14, 1)
    set_entry(W, 5, 14, 1)
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for a_row in range(n):
        for b_col in range(p):
            for j in range(m):
                a_idx = a_row * m + j
                b_idx = j * p + b_col
                c_idx = b_col * n + a_row
                T_target[a_idx, b_idx, c_idx] = 1.0
    
    T_comp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_comp += np.einsum('i,j,k->ijk', Ua[:, r], Va[:, r], Wa[:, r])
    
    if not np.allclose(T_comp, T_target, atol=1e-9):
        # Fall back to rank 17
        rank = 17
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Rebuild with original rank-17
        def set_entry2(mat, i, j, val):
            mat[i][j] += val
        
        set_entry2(U, 0, 0, 1); set_entry2(U, 4, 0, 1)
        set_entry2(V, 0, 0, 1); set_entry2(V, 4, 0, 1)
        set_entry2(W, 0, 0, 1); set_entry2(W, 3, 0, 1)
        
        set_entry2(U, 3, 1, 1); set_entry2(U, 4, 1, 1)
        set_entry2(V, 0, 1, 1)
        set_entry2(W, 1, 1, 1); set_entry2(W, 3, 1, -1)
        
        set_entry2(U, 0, 2, 1)
        set_entry2(V, 1, 2, 1); set_entry2(V, 4, 2, -1)
        set_entry2(W, 2, 2, 1); set_entry2(W, 3, 2, 1)
        
        set_entry2(U, 4, 3, 1)
        set_entry2(V, 3, 3, 1); set_entry2(V, 0, 3, -1)
        set_entry2(W, 0, 3, 1); set_entry2(W, 1, 3, 1)
        
        set_entry2(U, 0, 4, 1); set_entry2(U, 1, 4, 1)
        set_entry2(V, 4, 4, 1)
        set_entry2(W, 0, 4, -1); set_entry2(W, 2, 4, 1)
        
        set_entry2(U, 3, 5, 1); set_entry2(U, 0, 5, -1)
        set_entry2(V, 0, 5, 1); set_entry2(V, 1, 5, 1)
        set_entry2(W, 3, 5, 1)
        
        set_entry2(U, 1, 6, 1); set_entry2(U, 4, 6, -1)
        set_entry2(V, 3, 6, 1); set_entry2(V, 4, 6, 1)
        set_entry2(W, 0, 6, 1)
        
        set_entry2(U, 2, 7, 1); set_entry2(V, 6, 7, 1)
        set_entry2(W, 0, 7, 1)
        set_entry2(U, 2, 8, 1); set_entry2(V, 7, 8, 1)
        set_entry2(W, 2, 8, 1)
        set_entry2(U, 2, 9, 1); set_entry2(V, 8, 9, 1)
        set_entry2(W, 4, 9, 1)
        set_entry2(U, 5, 10, 1); set_entry2(V, 6, 10, 1)
        set_entry2(W, 1, 10, 1)
        set_entry2(U, 5, 11, 1); set_entry2(V, 7, 11, 1)
        set_entry2(W, 3, 11, 1)
        set_entry2(U, 5, 12, 1); set_entry2(V, 8, 12, 1)
        set_entry2(W, 5, 12, 1)
        set_entry2(U, 0, 13, 1); set_entry2(V, 2, 13, 1)
        set_entry2(W, 4, 13, 1)
        set_entry2(U, 3, 14, 1); set_entry2(V, 2, 14, 1)
        set_entry2(W, 5, 14, 1)
        set_entry2(U, 1, 15, 1); set_entry2(V, 5, 15, 1)
        set_entry2(W, 4, 15, 1)
        set_entry2(U, 4, 16, 1); set_entry2(V, 5, 16, 1)
        set_entry2(W, 5, 16, 1)
        
        result = {"U": U, "V": V, "W": W}
    
    return result
