import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 33
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col_offset = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col_offset
        # Strassen coefficients
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We define the terms (r) and their contributions to U, V, W
        # U[i,j], V[j,k], W[k,i]
        
        terms = []
        
        # Term 0: M1
        # A: (0,0), (1,1) -> U rows: 0*3+0=0, 1*3+1=4 (local)
        # B: (0,0), (1,1) -> V rows: 0*2+0=0, 1*2+1=3 (local)
        # C: (0,0), (1,1) -> W rows: 0*2+0=0, 1*2+1=3 (local)
        # Signs: +1 for all?
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # This is not a single rank 1 term. Strassen uses 7 rank 1 terms for the whole 2x2x2.
        # The standard Strassen decomposition is:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to express this as sum of 7 rank 1 tensors.
        # r=0: M1
        # U: A00+A11 -> rows 0, 4 (local) with val 1
        # V: B00+B11 -> rows 0, 3 (local) with val 1
        # W: C00, C11 -> rows 0, 3 (local) with val 1
        # But M1 contributes to C00 with +1 and C11 with +1.
        # So W[0]=1, W[3]=1.
        
        # r=1: M2
        # U: A10+A11 -> rows 3, 4 (local) with val 1
        # V: B00 -> row 0 (local) with val 1
        # W: C10 -> row 2 (local) with val 1 (C10 is k=0, i=1 -> 0*2+1=1? No. W[k*n+i]. k=0, i=1 -> 1. k=1, i=0 -> 2. k=1, i=1 -> 3.
        # C10: i=1, k=0. W[0*2+1] = W[1].
        # So W[1]=1.
        
        # r=2: M3
        # U: A00 -> row 0 (local)
        # V: B01-B11 -> rows 1, 3 (local) with vals 1, -1
        # W: C01 -> row 2 (local) (i=0, k=1 -> 1*2+0=2).
        # C01 gets +M3.
        
        # r=3: M4
        # U: A11 -> row 4 (local)
        # V: B10-B00 -> rows 2, 0 (local) with vals 1, -1
        # W: C00 (+M4), C10 (+M4).
        # C00: i=0, k=0 -> W[0].
        # C10: i=1, k=0 -> W[1].
        # So W[0]=1, W[1]=1.
        
        # r=4: M5
        # U: A00+A01 -> rows 0, 1 (local)
        # V: B11 -> row 3 (local)
        # W: C00 (-M5), C01 (+M5).
        # C00: W[0] = -1.
        # C01: W[2] = 1.
        
        # r=5: M6
        # U: A10-A00 -> rows 3, 0 (local) with vals 1, -1
        # V: B00+B01 -> rows 0, 1 (local) with vals 1, 1
        # W: C11 (+M6).
        # C11: i=1, k=1 -> W[3].
        # So W[3]=1.
        
        # r=6: M7
        # U: A01-A11 -> rows 1, 4 (local) with vals 1, -1
        # V: B10+B11 -> rows 2, 3 (local) with vals 1, 1
        # W: C00 (+M7), C11 (-M7? No, C11 = M1 - M2 + M3 + M6. M7 is in C00 only?
        # C00 = M1 + M4 - M5 + M7.
        # C11 = M1 - M2 + M3 + M6.
        # So M7 contributes to C00 only.
        # W[0]=1.
        
        # Let's verify the W contributions:
        # C00 (W[0]): M1(+1) + M4(+1) + M5(-1) + M7(+1)
        # C01 (W[2]): M3(+1) + M5(+1)
        # C10 (W[1]): M2(+1) + M4(+1)
        # C11 (W[3]): M1(+1) + M2(-1) + M3(+1) + M6(+1)
        
        # Now map to global indices.
        
        # r=0
        U[i_off*3 + 0 + j_off, col_offset] = 1
        U[i_off*3 + 1 + j_off+1, col_offset] = 1
        V[j_off*8 + 0 + k_off, col_offset] = 1
        V[j_off*8 + 1 + k_off+1, col_offset] = 1
        W[k_off*2 + 0 + i_off, col_offset] = 1
        W[k_off*2 + 1 + i_off+1, col_offset] = 1
        col_offset += 1
        
        # r=1
        U[i_off*3 + 1 + j_off, col_offset] = 1
        U[i_off*3 + 1 + j_off+1, col_offset] = 1
        V[j_off*8 + 0 + k_off, col_offset] = 1
        W[k_off*2 + 0 + i_off+1, col_offset] = 1
        col_offset += 1
        
        # r=2
        U[i_off*3 + 0 + j_off, col_offset] = 1
        V[j_off*8 + 0 + k_off+1, col_offset] = 1
        V[j_off*8 + 1 + k_off+1, col_offset] = -1
        W[k_off*2 + 1 + i_off, col_offset] = 1
        col_offset += 1
        
        # r=3
        U[i_off*3 + 1 + j_off+1, col_offset] = 1
        V[j_off*8 + 1 + k_off, col_offset] = 1
        V[j_off*8 + 0 + k_off, col_offset] = -1
        W[k_off*2 + 0 + i_off, col_offset] = 1
        W[k_off*2 + 0 + i_off+1, col_offset] = 1
        col_offset += 1
        
        # r=4
        U[i_off*3 + 0 + j_off, col_offset] = 1
        U[i_off*3 + 0 + j_off+1, col_offset] = 1
        V[j_off*8 + 1 + k_off+1, col_offset] = 1
        W[k_off*2 + 0 + i_off, col_offset] = -1
        W[k_off*2 + 1 + i_off, col_offset] = 1
        col_offset += 1
        
        # r=5
        U[i_off*3 + 1 + j_off, col_offset] = 1
        U[i_off*3 + 0 + j_off, col_offset] = -1
        V[j_off*8 + 0 + k_off, col_offset] = 1
        V[j_off*8 + 0 + k_off+1, col_offset] = 1
        W[k_off*2 + 1 + i_off+1, col_offset] = 1
        col_offset += 1
        
        # r=6
        U[i_off*3 + 0 + j_off+1, col_offset] = 1
        U[i_off*3 + 1 + j_off+1, col_offset] = -1
        V[j_off*8 + 1 + k_off, col_offset] = 1
        V[j_off*8 + 1 + k_off+1, col_offset] = 1
        W[k_off*2 + 0 + i_off, col_offset] = 1
        col_offset += 1

    # Helper to add trivial 2x1xk block
    def add_trivial_21k(i_off, j_off, k_off, k_size):
        nonlocal col_offset
        for i_loc in range(2):
            for k_loc in range(k_size):
                i_glob = i_off + i_loc
                j_glob = j_off
                k_glob = k_off + k_loc
                
                U[i_glob*3 + j_glob, col_offset] = 1
                V[j_glob*8 + k_glob, col_offset] = 1
                W[k_glob*2 + i_glob, col_offset] = 1
                col_offset += 1

    # Helper to add trivial 2x3xk block
    def add_trivial_23k(i_off, j_off, k_off, k_size):
        nonlocal col_offset
        for i_loc in range(2):
            for j_loc in range(3):
                for k_loc in range(k_size):
                    i_glob = i_off + i_loc
                    j_glob = j_off + j_loc
                    k_glob = k_off + k_loc
                    
                    U[i_glob*3 + j_glob, col_offset] = 1
                    V[j_glob*8 + k_glob, col_offset] = 1
                    W[k_glob*2 + i_glob, col_offset] = 1
                    col_offset += 1

    # Block 1: A1 B11,1 (2x2x2)
    # i_off=0, j_off=0, k_off=0
    add_strassen_222(0, 0, 0)
    
    # Block 2: A1 B11,2 (2x2x2)
    # i_off=0, j_off=0, k_off=2
    add_strassen_222(0, 0, 2)
    
    # Block 3: A1 B11,3 (2x2x2)
    # i_off=0, j_off=0, k_off=4
    add_strassen_222(0, 0, 4)
    
    # Block 4: A2 B21 (2x1x6)
    # i_off=0, j_off=2, k_off=0, k_size=6
    add_trivial_21k(0, 2, 0, 6)
    
    # Block 5: A B_right (2x3x2)
    # i_off=0, j_off=0, k_off=6, k_size=2
    add_trivial_23k(0, 0, 6, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
