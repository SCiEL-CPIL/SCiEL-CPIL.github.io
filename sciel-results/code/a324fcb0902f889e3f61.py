import numpy as np

def construct():
    n, m, p = 3, 4, 8
    rank_total = 88
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col_offset = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen(i_off, j_off, k_off):
        nonlocal col_offset
        # i_off: row offset in A/C
        # j_off: col offset in A / row offset in B
        # k_off: col offset in B/C
        
        # Strassen terms
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B10-B11)
        # M4: A11(B01-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # We need to define U, V, W for each of the 7 terms.
        # U indices: (i, j)
        # V indices: (j, k)
        # W indices: (k, i)
        
        # Term 1
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: (0,0)=1, (1,1)=1
        U[i_off*2 + 0, col_offset] = 1
        U[i_off*2 + 1, col_offset] = 1
        V[j_off*2 + 0, col_offset] = 1
        V[j_off*2 + 1, col_offset] = 1
        W[k_off*2 + 0, col_offset] = 1
        W[k_off*2 + 1, col_offset] = 1
        col_offset += 1
        
        # Term 2
        # U: (1,0)=1, (1,1)=1
        # V: (0,0)=1
        # W: (0,1)=1
        U[i_off*2 + 1, col_offset] = 1
        U[i_off*2 + 1, col_offset] = 1 # (1,1) also? No, (1,0) and (1,1)
        # Wait, U is indexed by (i, j).
        # (1,0) is i=1, j=0. (1,1) is i=1, j=1.
        U[i_off*2 + 1, col_offset] = 1 # This is (1,0) if we map i*m+j?
        # My U mapping: U[i*m+j].
        # For 2x2 block, local i, j in 0..1.
        # Global i = i_off + i. Global j = j_off + j.
        # Index = (i_off+i)*m + (j_off+j).
        
        # Let's rewrite the helper to use explicit global indices.
        
        # I will implement the addition logic directly in the main loop for clarity and correctness.
        pass

    # I will write the full code in the final block.
    return {"U": U, "V": V, "W": W}
