import numpy as np

def construct():
    n, m, p = 4, 4, 6
    # Target rank 88
    rank = 88
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(i_off, j_off, k_off, col_start):
        # i_off: row offset in A (0 or 2)
        # j_off: col offset in A (0 or 2)
        # k_off: col offset in B (0, 3)
        # Actually, B is 2x3. k_off is offset in B's columns.
        # A is 2x2.
        # B is 2x2 (first 2 cols of 2x3).
        
        # Strassen indices
        # A: a=A[i_off, j_off], b=A[i_off, j_off+1], c=A[i_off+1, j_off], d=A[i_off+1, j_off+1]
        # B: e=B[j_off, k_off], f=B[j_off, k_off+1], g=B[j_off+1, k_off], h=B[j_off+1, k_off+1]
        # C: p=C[k_off, i_off], q=C[k_off, i_off+1], r=C[k_off+1, i_off], s=C[k_off+1, i_off+1]
        
        # M1 = (a+d)(e+h)
        U[i_off*m + j_off, col_start] = 1
        U[(i_off+1)*m + (j_off+1), col_start] = 1
        V[j_off*p + k_off, col_start] = 1
        V[(j_off+1)*p + (k_off+1), col_start] = 1
        W[k_off*n + i_off, col_start] = 1
        W[(k_off+1)*n + (i_off+1), col_start] = 1
        col_start += 1
        
        # M2 = (c+d)e
        U[(i_off+1)*m + j_off, col_start] = 1
        U[(i_off+1)*m + (j_off+1), col_start] = 1
        V[j_off*p + k_off, col_start] = 1
        W[k_off*n + (i_off+1), col_start] = 1
        col_start += 1
        
        # M3 = a(f-h)
        U[i_off*m + j_off, col_start] = 1
        V[j_off*p + (k_off+1), col_start] = 1
        V[(j_off+1)*p + k_off, col_start] = -1
        W[k_off*n + i_off, col_start] = 1
        W[(k_off+1)*n + i_off, col_start] = 1 # Wait, M3 contributes to q and r?
        # q = M3 + M5
        # r = M2 + M4
        # M3 = a(f-h) = af - ah
        # C[k_off, i_off+1] gets M3.
        # C[k_off+1, i_off] gets M3? No.
        # Let's re-verify Strassen output mapping.
        # p = M1 + M4 - M5 + M7
        # q = M3 + M5
        # r = M2 + M4
        # s = M1 - M2 + M3 + M6
        # So M3 contributes to q (C[k_off, i_off+1]) and s (C[k_off+1, i_off+1]).
        W[k_off*n + (i_off+1), col_start] = 1
        W[(k_off+1)*n + (i_off+1), col_start] = 1
        col_start += 1
        
        # M4 = d(g-e)
        U[(i_off+1)*m + (j_off+1), col_start] = 1
        V[(j_off+1)*p + k_off, col_start] = 1
        V[j_off*p + k_off, col_start] = -1
        W[k_off*n + i_off, col_start] = 1
        W[(k_off+1)*n + (i_off+1), col_start] = 1
        col_start += 1
        
        # M5 = (a+b)h
        U[i_off*m + j_off, col_start] = 1
        U[i_off*m + (j_off+1), col_start] = 1
        V[(j_off+1)*p + (k_off+1), col_start] = 1
        W[k_off*n + i_off, col_start] = -1
        W[k_off*n + (i_off+1), col_start] = 1
        col_start += 1
        
        # M6 = (c-a)(e+f)
        U[(i_off+1)*m + j_off, col_start] = 1
        U[i_off*m + j_off, col_start] = -1
        V[j_off*p + k_off, col_start] = 1
        V[j_off*p + (k_off+1), col_start] = 1
        W[(k_off+1)*n + (i_off+1), col_start] = 1
        col_start += 1
        
        # M7 = (b-d)(g+h)
        U[i_off*m + (j_off+1), col_start] = 1
        U[(i_off+1)*m + (j_off+1), col_start] = -1
        V[(j_off+1)*p + k_off, col_start] = 1
        V[(j_off+1)*p + (k_off+1), col_start] = 1
        W[k_off*n + i_off, col_start] = 1
        W[(k_off+1)*n + (i_off+1), col_start] = 1
        col_start += 1
        
        return col_start

    # Helper to add 2x2x1 schoolbook block
    def add_schoolbook_221(i_off, j_off, k_off, col_start):
        # A 2x2, B 2x1 (col k_off), C 2x1 (col k_off)
        # C[k_off, i_off] = A[i_off, j_off]B[j_off, k_off] + A[i_off, j_off+1]B[j_off+1, k_off]
        # C[k_off, i_off+1] = A[i_off+1, j_off]B[j_off, k_off] + A[i_off+1, j_off+1]B[j_off+1, k_off]
        
        # Term 1: A[i_off, j_off] B[j_off, k_off] -> C[k_off, i_off]
        U[i_off*m + j_off, col_start] = 1
        V[j_off*p + k_off, col_start] = 1
        W[k_off*n + i_off, col_start] = 1
        col_start += 1
        
        # Term 2: A[i_off, j_off+1] B[j_off+1, k_off] -> C[k_off, i_off]
        U[i_off*m + (j_off+1), col_start] = 1
        V[(j_off+1)*p + k_off, col_start] = 1
        W[k_off*n + i_off, col_start] = 1
        col_start += 1
        
        # Term 3: A[i_off+1, j_off] B[j_off, k_off] -> C[k_off, i_off+1]
        U[(i_off+1)*m + j_off, col_start] = 1
        V[j_off*p + k_off, col_start] = 1
        W[k_off*n + (i_off+1), col_start] = 1
        col_start += 1
        
        # Term 4: A[i_off+1, j_off+1] B[j_off+1, k_off] -> C[k_off, i_off+1]
        U[(i_off+1)*m + (j_off+1), col_start] = 1
        V[(j_off+1)*p + k_off, col_start] = 1
        W[k_off*n + (i_off+1), col_start] = 1
        col_start += 1
        
        return col_start

    col = 0
    # 8 blocks of 2x2x3
    # A_L (4x2) * B_T (2x6)
    # A_L1 (2x2) * B_T1 (2x3) -> i_off=0, j_off=0, k_off=0
    col = add_strassen_222(0, 0, 0, col)
    col = add_schoolbook_221(0, 0, 2, col)
    
    # A_L1 (2x2) * B_T2 (2x3) -> i_off=0, j_off=0, k_off=3
    col = add_strassen_222(0, 0, 3, col)
    col = add_schoolbook_221(0, 0, 5, col)
    
    # A_L2 (2x2) * B_T1 (2x3) -> i_off=2, j_off=0, k_off=0
    col = add_strassen_222(2, 0, 0, col)
    col = add_schoolbook_221(2, 0, 2, col)
    
    # A_L2 (2x2) * B_T2 (2x3) -> i_off=2, j_off=0, k_off=3
    col = add_strassen_222(2, 0, 3, col)
    col = add_schoolbook_221(2, 0, 5, col)
    
    # A_R (4x2) * B_B (2x6)
    # A_R1 (2x2) * B_B1 (2x3) -> i_off=0, j_off=2, k_off=0
    col = add_strassen_222(0, 2, 0, col)
    col = add_schoolbook_221(0, 2, 2, col)
    
    # A_R1 (2x2) * B_B2 (2x3) -> i_off=0, j_off=2, k_off=3
    col = add_strassen_222(0, 2, 3, col)
    col = add_schoolbook_221(0, 2, 5, col)
    
    # A_R2 (2x2) * B_B1 (2x3) -> i_off=2, j_off=2, k_off=0
    col = add_strassen_222(2, 2, 0, col)
    col = add_schoolbook_221(2, 2, 2, col)
    
    # A_R2 (2x2) * B_B2 (2x3) -> i_off=2, j_off=2, k_off=3
    col = add_strassen_222(2, 2, 3, col)
    col = add_schoolbook_221(2, 2, 5, col)
    
    return {"U": U, "V": V, "W": W}
