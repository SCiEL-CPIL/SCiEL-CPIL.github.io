"""Rewritten program for matrix multiplication 2x4x5 with reduced rank."""

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # Strassen 7 mults for 2x2x2
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        U[i_off*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00 -> C10, C11
        U[i_off*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M3 = A00(B01-B11) -> C01, C11
        U[i_off*m + j_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        W[(k_off+1)*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M4 = A11(B10-B00) -> C00, C10
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        V[j_off*p + k_off][col] = -1
        W[k_off*n + i_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11 -> C00, C01
        U[i_off*m + j_off][col] = 1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = -1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01) -> C11
        U[i_off*m + (j_off+1)][col] = 1
        U[i_off*m + j_off][col] = -1
        V[j_off*p + k_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11) -> C00, C11
        U[i_off*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1

    def add_schoolbook_241(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for j in range(4):
                U[(i_off+i)*m + (j_off+j)][col] = 1
                V[(j_off+j)*p + (k_off)][col] = 1
                W[(k_off)*n + (i_off+i)][col] = 1
                col += 1

    # 4 Strassen blocks for 2x4x4 part
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 2)
    
    # 1 Schoolbook block for 2x4x1 part
    add_schoolbook_241(0, 0, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
