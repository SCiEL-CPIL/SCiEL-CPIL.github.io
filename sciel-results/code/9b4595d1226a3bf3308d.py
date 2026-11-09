import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    
    def compute_T(U, V, W, r):
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        return np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)

    def verify(U, V, W, r, tol=1e-9):
        T = compute_T(U, V, W, r)
        return np.allclose(T, 1.0, atol=tol)

    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Construct rank-55 solution using Strassen for 2x2x10 part and standard for 2x1x10 part
    def construct_rank_55():
        # Strassen factors for 2x2x2
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices: 00, 01, 10, 11
        U_str = np.zeros((4, 7), dtype=int)
        V_str = np.zeros((4, 7), dtype=int)
        W_str = np.zeros((4, 7), dtype=int)
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U_str[0, 0] = 1; U_str[3, 0] = 1
        V_str[0, 0] = 1; V_str[3, 0] = 1
        W_str[0, 0] = 1; W_str[3, 0] = 1
        
        # M2: (A10+A11)B00 -> C10
        U_str[2, 1] = 1; U_str[3, 1] = 1
        V_str[0, 1] = 1
        W_str[1, 1] = 1
        
        # M3: A00(B01-B11) -> C01
        U_str[0, 2] = 1
        V_str[1, 2] = 1; V_str[3, 2] = -1
        W_str[2, 2] = 1
        
        # M4: A11(B10-B00) -> C00, C10
        U_str[3, 3] = 1
        V_str[2, 3] = 1; V_str[0, 3] = -1
        W_str[0, 3] = 1; W_str[1, 3] = 1
        
        # M5: (A00+A01)B11 -> C00(-1), C01
        U_str[0, 4] = 1; U_str[1, 4] = 1
        V_str[3, 4] = 1
        W_str[0, 4] = -1; W_str[2, 4] = 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U_str[2, 5] = 1; U_str[0, 5] = -1
        V_str[0, 5] = 1; V_str[1, 5] = 1
        W_str[3, 5] = 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U_str[1, 6] = 1; U_str[3, 6] = -1
        V_str[2, 6] = 1; V_str[3, 6] = 1
        W_str[0, 6] = 1
        
        # Total rank 55
        # Part 1: 2x2x10 (5 blocks of 2x2x2) -> Rank 35
        # Part 2: 2x1x10 (standard) -> Rank 20
        R = 55
        U_tot = np.zeros((n*m, R), dtype=int)
        V_tot = np.zeros((m*p, R), dtype=int)
        W_tot = np.zeros((p*n, R), dtype=int)
        
        # Part 1: Strassen
        for b in range(5):
            col_offset = b * 7
            # U: A rows 0,1 (cols 0,1 of A)
            # A is 2x3. Cols 0,1 used.
            # U_str is 4x7 (A00, A01, A10, A11)
            # Map to U_tot rows: i*3 + j
            # A00 -> 0*3+0=0, A01 -> 0*3+1=1, A10 -> 1*3+0=3, A11 -> 1*3+1=4
            U_tot[0:5, col_offset:col_offset+7] = U_str[[0, 1, 3, 4], :]
            
            # V: B rows 0,1 (cols 2b, 2b+1 of B)
            # B is 3x10. Rows 0,1 used.
            # V_str is 4x7 (B00, B01, B10, B11)
            # Map to V_tot rows: j*10 + k
            # B00 -> 0*10 + 2b, B01 -> 0*10 + 2b+1
            # B10 -> 1*10 + 2b, B11 -> 1*10 + 2b+1
            rows_V = [2*b, 2*b+1, 10+2*b, 10+2*b+1]
            V_tot[rows_V, col_offset:col_offset+7] = V_str
            
            # W: C cols 2b, 2b+1
            # C is 2x10.
            # W_str is 4x7 (C00, C01, C10, C11)
            # Map to W_tot rows: k*2 + i
            # C00 -> 0*2+0=0, C01 -> 1*2+0=2
            # C10 -> 0*2+1=1, C11 -> 1*2+1=3
            rows_W = [0, 2, 1, 3]
            W_tot[rows_W, col_offset:col_offset+7] = W_str
            
        # Part 2: Standard 2x1x10
        # A col 2, B row 2
        # Rank 20
        for i in range(2):
            for k in range(10):
                col_idx = 35 + i * 10 + k
                U_tot[i*3 + 2, col_idx] = 1
                V_tot[2*10 + k, col_idx] = 1
                W_tot[k*2 + i, col_idx] = 1
                
        return U_tot, V_tot, W_tot

    # Try rank 55
    try:
        U, V, W = construct_rank_55()
        if verify(U, V, W, 55):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
