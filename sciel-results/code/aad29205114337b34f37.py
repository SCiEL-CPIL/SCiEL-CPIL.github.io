import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build the decomposition using block structure
    # Split: n=2 (single block), m=5 -> 2+3, p=6 -> 2+2+2
    
    # We'll build U, V, W matrices with a common rank
    # Each sub-block contributes its own terms
    
    # For 2x2x2: Strassen gives rank 7
    # For 2x3x2: let's use schoolbook rank 12 for now, total = 3*7 + 3*12 = 57
    
    # Let's try to get 2x3x2 down to rank 11
    
    # First, let's construct the Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        """Returns U, V, W for 2x2x2 matrix multiplication with rank 7"""
        # U has 4 rows (2x2), V has 4 rows (2x2), W has 4 rows (2x2)
        # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 products
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1  # A00, A11
        V[0, 0] = 1; V[3, 0] = 1  # B00, B11
        W[0, 0] = 1; W[3, 0] = 1  # C00, C11
        W[0, 0] = 1  # C00 gets +M1
        W[3, 0] = 1  # C11 gets +M1
        
        # Let me redo this more carefully
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # C00+=M1, C11+=M1
        
        # M2 = (A10+A11)B00
        U[1, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1  # C10+=M2, C11-=M2
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[3, 2] = 1  # C01+=M3, C11+=M3
        
        # M4 = A11(B01-B10)
        U[3, 3] = 1
        V[1, 3] = 1; V[2, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1  # C00+=M4, C10+=M4
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[0, 5] = 1  # Wait, this doesn't work
        
        # I'm making errors. Let me look up Strassen carefully.
        
        # Standard Strassen:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B10)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B10)
        # M7 = (A01 - A11) * (B01 + B11)
        #
        # C00 = M1 - M5 + M7
        # C01 = M4 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[1, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[1, 3] = 1; V[2, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    U_s, V_s, W_s = strassen_2x2x2()
    
    # Verify Strassen
    T_expected = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_expected[i*2+j, j*2+k, k*2+i] = 1.0
    
    T_actual = np.zeros((4, 4, 4))
    for r in range(7):
        T_actual += np.outer(np.outer(U_s[:, r], V_s[:, r]), W_s[:, r])
    
    if not np.allclose(T_expected, T_actual):
        print("Strassen verification failed!")
        # Fallback to schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now build the full 2x5x6 decomposition
    # Split m=5 into [0,1] and [2,3,4]
    # Split p=6 into [0,1], [2,3], [4,5]
    
    # Block (m=[0,1], p=[0,1]): 2x2x2, use Strassen
    # Block (m=[0,1], p=[2,3]): 2x2x2, use Strassen
    # Block (m=[0,1], p=[4,5]): 2x2x2, use Strassen
    # Block (m=[2,3,4], p=[0,1]): 2x3x2, use schoolbook (rank 12)
    # Block (m=[2,3,4], p=[2,3]): 2x3x2, use schoolbook (rank 12)
    # Block (m=[2,3,4], p=[4,5]): 2x3x2, use schoolbook (rank 12)
    
    total_rank = 3*7 + 3*12 = 57
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    col = 0
    
    # 2x2x2 blocks (m=[0,1], various p)
    for p_offset in [0, 2, 4]:
        # U indices: i*5+j for i in [0,1], j in [0,1] -> [0,1, 5,6]
        # V indices: j*6+k for j in [0,1], k in [p_offset, p_offset+1]
        # W indices: k*2+i for k in [p_offset, p_offset+1], i in [0,1]
        
        for r in range(7):
            # U
            for u_row in range(4):
                # u_row corresponds to i*2+j in the 2x2 subproblem
                i_sub = u_row // 2
                j_sub = u_row % 2
                i_full = i_sub
                j_full = j_sub
                U_full[i_full*5 + j_full, col] = U_s[u_row, r]
            
            # V
            for v_row in range(4):
                j_sub = v_row // 2
                k_sub = v_row % 2
                j_full = j_sub
                k_full = p_offset + k_sub
                V_full[j_full*6 + k_full, col] = V_s[v_row, r]
            
            # W
            for w_row in range(4):
                k_sub = w_row // 2
                i_sub = w_row % 2
                k_full = p_offset + k_sub
                i_full = i_sub
                W_full[k_full*2 + i_full, col] = W_s[w_row, r]
            
            col += 1
    
    # 2x3x2 blocks (m=[2,3,4], various p)
    for p_offset in [0, 2, 4]:
        # Schoolbook for 2x3x2: rank 12
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    j_full = 2 + j
                    k_full = p_offset + k
                    U_full[i*5 + j_full, col] = 1
                    V_full[j_full*6 + k_full, col] = 1
                    W_full[k_full*2 + i, col] = 1
                    col += 1
    
    # Verify
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_actual = np.zeros((n*m, m*p, p*n))
    for r in range(total_rank):
        T_actual += np.outer(np.outer(U_full[:, r], V_full[:, r]), W_full[:, r])
    
    if not np.allclose(T_expected, T_actual):
        print("Full decomposition verification failed!")
        # Fallback
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
