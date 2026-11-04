import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Base Strassen decomposition for 2x2x2
    # U, V, W are 4x7
    U_base = np.zeros((4, 7))
    V_base = np.zeros((4, 7))
    W_base = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    # A rows: 00, 11. B rows: 00, 11. C: 00, 11.
    U_base[0, 0] = 1; U_base[3, 0] = 1
    V_base[0, 0] = 1; V_base[3, 0] = 1
    W_base[0, 0] = 1; W_base[3, 0] = 1
    
    # M2: (A10+A11)B00
    U_base[2, 1] = 1; U_base[3, 1] = 1
    V_base[0, 1] = 1
    W_base[1, 1] = 1; W_base[3, 1] = -1
    
    # M3: A00(B01-B11)
    U_base[0, 2] = 1
    V_base[1, 2] = 1; V_base[3, 2] = -1
    W_base[1, 2] = 1; W_base[3, 2] = 1
    
    # M4: A11(B10-B00)
    U_base[3, 3] = 1
    V_base[2, 3] = 1; V_base[0, 3] = -1
    W_base[1, 3] = 1; W_base[3, 3] = 1
    
    # M5: (A00+A01)B11
    U_base[0, 4] = 1; U_base[1, 4] = 1
    V_base[3, 4] = 1
    W_base[0, 4] = -1; W_base[3, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U_base[2, 5] = 1; U_base[0, 5] = -1
    V_base[0, 5] = 1; V_base[1, 5] = 1
    W_base[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U_base[1, 6] = 1; U_base[3, 6] = -1
    V_base[2, 6] = 1; V_base[3, 6] = 1
    W_base[0, 6] = 1; W_base[1, 6] = 1
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_offset = 0
    
    # A1 B1: A cols 0,1. B rows 0,1.
    # 4 blocks of B1
    for b in range(4):
        for r in range(7):
            # Local indices
            # U_base row index: i*2 + j
            # V_base row index: j*2 + k
            # W_base row index: k*2 + i
            
            # We need to map local to global
            # This is done by iterating over non-zero entries in U_base, V_base, W_base?
            # No, we just copy the column and permute rows.
            
            # U global row: i*4 + j
            # V global row: j*8 + (2b + k)
            # W global row: (2b + k)*2 + i
            
            # We can just create index maps
            u_map = np.zeros(4, dtype=int)
            v_map = np.zeros(4, dtype=int)
            w_map = np.zeros(4, dtype=int)
            
            for i in range(2):
                for j in range(2):
                    u_map[i*2 + j] = i*4 + j
                    w_map[j*2 + i] = (2*b + j)*2 + i # Wait, W local index is k*2+i.
                    # W_base row is k*2 + i.
                    # So for each local row l in W_base, we need to find k, i.
                    # l = k*2 + i.
            
            # Let's just do it element by element for safety
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # U local (i, j)
                        u_loc = i * 2 + j
                        u_glob = i * 4 + j
                        
                        # V local (j, k)
                        v_loc = j * 2 + k
                        v_glob = j * 8 + (2 * b + k)
                        
                        # W local (k, i)
                        w_loc = k * 2 + i
                        w_glob = (2 * b + k) * 2 + i
                        
                        # We need to add U_base[u_loc, r] to U[u_glob, col_offset]
                        # But U_base is sparse.
                        # Actually, we can just construct the full column.
                        # U_col[u_glob] = U_base[u_loc, r]
                        # This is not correct because U_base has entries for specific (i,j).
                        # U_base[u_loc, r] is the value.
                        
                        # So:
                        U[u_glob, col_offset] += U_base[u_loc, r]
                        V[v_glob, col_offset] += V_base[v_loc, r]
                        W[w_glob, col_offset] += W_base[w_loc, r]
                        
        col_offset += 7
        
    # A2 B2: A cols 2,3. B rows 2,3.
    for b in range(4):
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # A2 local j' -> Global J = j' + 2
                        j_glob_a = j + 2
                        
                        u_loc = i * 2 + j
                        u_glob = i * 4 + j_glob_a
                        
                        v_loc = j * 2 + k
                        # B2 local j' -> Global J = j' + 2
                        j_glob_b = j + 2
                        v_glob = j_glob_b * 8 + (2 * b + k)
                        
                        w_loc = k * 2 + i
                        w_glob = (2 * b + k) * 2 + i
                        
                        U[u_glob, col_offset] += U_base[u_loc, r]
                        V[v_glob, col_offset] += V_base[v_loc, r]
                        W[w_glob, col_offset] += W_base[w_loc, r]
                        
        col_offset += 7
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
