import numpy as np

def construct():
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[:, 2] * B[2, :]
    # i in 0..1, k in 0..4
    for i in range(n):
        for k in range(p):
            # U index i*m + 2
            U[i * m + 2][col] = 1
            # V index 2*p + k
            V[2 * p + k][col] = 1
            # W index k*n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: A[:, 0:2] * B[0:2, :]
    # Split B[0:2, :] into B[0:2, 0:2] (Strassen), B[0:2, 2:4] (Strassen), B[0:2, 4:5] (Schoolbook)
    
    # Helper to add Strassen block for A[0:2, 0:2] * B[0:2, k_offset:k_offset+2]
    def add_strassen_block(k_offset, start_col):
        # U indices for A[0:2, 0:2]
        # A00: 0, A01: 1, A10: 3, A11: 4
        u_idx = {
            "A00": 0, "A01": 1, "A10": 3, "A11": 4
        }
        
        # V indices for B[0:2, k_offset:k_offset+2]
        # B0,ko: 0*p + ko = ko
        # B0,k1: 0*p + ko+1 = ko+1
        # B1,ko: 1*p + ko = p+ko
        # B1,k1: 1*p + ko+1 = p+ko+1
        v_idx = {
            "B00": k_offset,
            "B01": k_offset + 1,
            "B10": p + k_offset,
            "B11": p + k_offset + 1
        }
        
        # W indices for C[0:2, k_offset:k_offset+2]
        # C0,ko: ko*n + 0 = 2*ko
        # C0,k1: (ko+1)*n + 0 = 2*ko + 2
        # C1,ko: ko*n + 1 = 2*ko + 1
        # C1,k1: (ko+1)*n + 1 = 2*ko + 3
        w_idx = {
            "C00": 2 * k_offset,
            "C01": 2 * k_offset + 2,
            "C10": 2 * k_offset + 1,
            "C11": 2 * k_offset + 3
        }
        
        # Strassen Terms
        # 1. M1 = (A00+A11)(B00+B11)
        U[u_idx["A00"]][start_col] += 1
        U[u_idx["A11"]][start_col] += 1
        V[v_idx["B00"]][start_col] += 1
        V[v_idx["B11"]][start_col] += 1
        W[w_idx["C00"]][start_col] += 1
        W[w_idx["C11"]][start_col] += 1
        col = start_col + 1
        
        # 2. M2 = (A10+A11)B00
        U[u_idx["A10"]][col] += 1
        U[u_idx["A11"]][col] += 1
        V[v_idx["B00"]][col] += 1
        W[w_idx["C10"]][col] += 1
        col += 1
        
        # 3. M3 = A00(B01-B11)
        U[u_idx["A00"]][col] += 1
        V[v_idx["B01"]][col] += 1
        V[v_idx["B11"]][col] += -1
        W[w_idx["C01"]][col] += 1
        W[w_idx["C11"]][col] += -1
        col += 1
        
        # 4. M4 = A11(B10-B00)
        U[u_idx["A11"]][col] += 1
        V[v_idx["B10"]][col] += 1
        V[v_idx["B00"]][col] += -1
        W[w_idx["C00"]][col] += -1
        W[w_idx["C10"]][col] += 1
        col += 1
        
        # 5. M5 = (A00+A01)B11
        U[u_idx["A00"]][col] += 1
        U[u_idx["A01"]][col] += 1
        V[v_idx["B11"]][col] += 1
        W[w_idx["C01"]][col] += 1
        col += 1
        
        # 6. M6 = (A10-A00)(B00+B01)
        U[u_idx["A10"]][col] += 1
        U[u_idx["A00"]][col] += -1
        V[v_idx["B00"]][col] += 1
        V[v_idx["B01"]][col] += 1
        W[w_idx["C00"]][col] += 1
        col += 1
        
        # 7. M7 = (A01-A11)(B10+B11)
        U[u_idx["A01"]][col] += 1
        U[u_idx["A11"]][col] += -1
        V[v_idx["B10"]][col] += 1
        V[v_idx["B11"]][col] += 1
        W[w_idx["C11"]][col] += 1
        col += 1
        
        return col

    col = add_strassen_block(0, col)
    col = add_strassen_block(2, col)
    
    # Block 2c: A[:, 0:2] * B[0:2, 4]
    # Schoolbook rank 4
    # Terms:
    # A00 B04 -> C04
    U[0][col] += 1
    V[4][col] += 1
    W[8][col] += 1
    col += 1
    
    # A01 B14 -> C04
    U[1][col] += 1
    V[9][col] += 1
    W[8][col] += 1
    col += 1
    
    # A10 B04 -> C14
    U[3][col] += 1
    V[4][col] += 1
    W[9][col] += 1
    col += 1
    
    # A11 B14 -> C14
    U[4][col] += 1
    V[9][col] += 1
    W[9][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
