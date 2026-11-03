import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 8
    
    # Schoolbook fallback
    def schoolbook():
        rank_sb = n * m * p  # 96
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, idx] = 1
                    V_sb[j * p + k, idx] = 1
                    W_sb[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    def verify(U, V, W, rank, tol=1e-9):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        if U_np.shape != (n * m, rank) or V_np.shape != (m * p, rank) or W_np.shape != (p * n, rank):
            return False
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        rng = np.random.default_rng(12345)
        for _ in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C = A @ B
            M = U_3d * A[:, :, np.newaxis]
            N = V_3d * B[:, :, np.newaxis]
            S = np.einsum('ijc,jkc->ikc', M, N)
            S_T = np.transpose(S, (1, 0, 2))
            C_dec = np.sum(S_T * W_3d, axis=2)
            if np.max(np.abs(C - C_dec)) > tol:
                return False
        return True
    
    def to_clean_list(mat_np):
        result = []
        for row in mat_np:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                elif abs(v * 2 - round(v * 2)) < 1e-9:
                    new_row.append(round(v * 2) / 2)
                else:
                    new_row.append(float(v))
            result.append(new_row)
        return result
    
    # Hybrid decomposition: rank 88
    # Part 1: 1×4×8 schoolbook for row 0 (rank 32)
    # Part 2: 2×4×8 via Strassen on 2×2×2 blocks (rank 56)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec, v_vec, w_vec are 1D numpy arrays."""
        U_rows.append(u_vec.copy())
        V_rows.append(v_vec.copy())
        W_rows.append(w_vec.copy())
    
    # Part 1: Schoolbook for row i=0
    # C[0,k] = sum_j A[0,j] * B[j,k]
    for j in range(m):
        for k in range(p):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            u[0 * m + j] = 1
            v[j * p + k] = 1
            w[k * n + 0] = 1
            add_term(u, v, w)
    
    # Part 2: Strassen for rows 1,2
    # For each half of B (cols 0-3 and cols 4-7):
    #   A_L = A[1:3, 0:2], A_R = A[1:3, 2:4]
    #   B_TL, B_TR, B_BL, B_BR (2x2 each)
    #   C_L = A_L*B_TL + A_R*B_BL
    #   C_R = A_L*B_TR + A_R*B_BR
    
    def strassen_2x2x2(U_indices, V_indices, W_indices):
        """
        Add Strassen rank-7 terms for a 2x2x2 block.
        U_indices: list of 4 (i,j) indices for A sub-block
        V_indices: list of 4 (j,k) indices for B sub-block  
        W_indices: list of 4 (k,i) indices for C sub-block
        The 2x2 blocks are ordered: [top-left, top-right, bottom-left, bottom-right]
        """
        # Strassen 7 products:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # U_indices = [A11, A12, A21, A22] as (i,j)
        # V_indices = [B11, B12, B21, B22] as (j,k)
        # W_indices = [C11, C12, C21, C22] as (k,i)
        
        A11, A12, A21, A22 = U_indices
        B11, B12, B21, B22 = V_indices
        C11, C12, C21, C22 = W_indices
        
        # M1 = (A11+A22)(B11+B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[A11] = 1; u[A22] = 1
        v[B11] = 1; v[B22] = 1
        w[C11] = 1; w[C12] = 0; w[C21] = 0; w[C22] = 1
        # C11 gets +1, C22 gets +1
        w[C11] = 1
        w[C22] = 1
        add_term(u, v, w)
        
        # M2 = (A21+A22)B11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[A21] = 1; u[A22] = 1
        v[B11] = 1
        w[C21] = 1
        add_term(u, v, w)
        
        # M3 = A11(B21-B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[A11] = 1
        v[B21] = 1; v[B22] = -1
        w[C12] = 1; w[C22] = 1
        add_term(u, v, w)
        
        # M4 = A22(B12-B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[A22] = 1
        v[B12] = 1; v[B11] = -1
        w[C11] = 1; w[C21] = 1
        add_term(u, v, w)
        
        # M5 = (A11+A12)B22
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[A11] = 1; u[A12] = 1
        v[B22] = 1
        w[C11] = -1; w[C12] = 1
        add_term(u, v, w)
        
        # M6 = (A21-A11)(B11+B12)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[A21] = 1; u[A11] = -1
        v[B11] = 1; v[B12] = 1
        w[C22] = 1
        add_term(u, v, w)
        
        # M7 = (A12-A22)(B21+B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[A12] = 1; u[A22] = -1
        v[B21] = 1; v[B22] = 1
        w[C11] = 1; w[C12] = 0
        # C11 gets +1
        w[C11] = 1
        add_term(u, v, w)
    
    # Process left half of B (cols 0-3)
    # A_L: rows 1,2; cols 0,1 -> U indices: (1,0),(1,1),(2,0),(2,1) = 4,5,8,9
    # A_R: rows 1,2; cols 2,3 -> U indices: (1,2),(1,3),(2,2),(2,3) = 6,7,10,11
    # B_TL: rows 0,1; cols 0,1 -> V indices: (0,0),(0,1),(1,0),(1,1) = 0,1,8,9
    # B_BL: rows 2,3; cols 0,1 -> V indices: (2,0),(2,1),(3,0),(3,1) = 16,17,24,25
    # B_TR: rows 0,1; cols 2,3 -> V indices: (0,2),(0,3),(1,2),(1,3) = 2,3,10,11
    # B_BR: rows 2,3; cols 2,3 -> V indices: (2,2),(2,3),(3,2),(3,3) = 18,19,26,27
    # C_L: rows 1,2; cols 0,1 -> W indices: (0,1),(0,2),(1,1),(1,2) = 1,2,4,5
    # C_R: rows 1,2; cols 2,3 -> W indices: (2,1),(2,2),(3,1),(3,2) = 7,8,10,11
    
    # C_L = A_L*B_TL + A_R*B_BL
    # A_L*B_TL: U=[4,5,8,9], V=[0,1,8,9], W=[1,2,4,5]
    strassen_2x2x2([4,5,8,9], [0,1,8,9], [1,2,4,5])
    # A_R*B_BL: U=[6,7,10,11], V=[16,17,24,25], W=[1,2,4,5]
    strassen_2x2x2([6,7,10,11], [16,17,24,25], [1,2,4,5])
    
    # C_R = A_L*B_TR + A_R*B_BR
    # A_L*B_TR: U=[4,5,8,9], V=[2,3,10,11], W=[7,8,10,11]
    strassen_2x2x2([4,5,8,9], [2,3,10,11], [7,8,10,11])
    # A_R*B_BR: U=[6,7,10,11], V=[18,19,26,27], W=[7,8,10,11]
    strassen_2x2x2([6,7,10,11], [18,19,26,27], [7,8,10,11])
    
    # Process right half of B (cols 4-7)
    # B_TL_right: rows 0,1; cols 4,5 -> V indices: (0,4),(0,5),(1,4),(1,5) = 4,5,12,13
    # B_TR_right: rows 0,1; cols 6,7 -> V indices: (0,6),(0,7),(1,6),(1,7) = 6,7,14,15
    # B_BL_right: rows 2,3; cols 4,5 -> V indices: (2,4),(2,5),(3,4),(3,5) = 20,21,28,29
    # B_BR_right: rows 2,3; cols 6,7 -> V indices: (2,6),(2,7),(3,6),(3,7) = 22,23,30,31
    # C_L_right: rows 1,2; cols 4,5 -> W indices: (4,1),(4,2),(5,1),(5,2) = 13,14,16,17
    # C_R_right: rows 1,2; cols 6,7 -> W indices: (6,1),(6,2),(7,1),(7,2) = 19,20,22,23
    
    # C_L_right = A_L*B_TL_right + A_R*B_BL_right
    # A_L*B_TL_right: U=[4,5,8,9], V=[4,5,12,13], W=[13,14,16,17]
    strassen_2x2x2([4,5,8,9], [4,5,12,13], [13,14,16,17])
    # A_R*B_BL_right: U=[6,7,10,11], V=[20,21,28,29], W=[13,14,16,17]
    strassen_2x2x2([6,7,10,11], [20,21,28,29], [13,14,16,17])
    
    # C_R_right = A_L*B_TR_right + A_R*B_BR_right
    # A_L*B_TR_right: U=[4,5,8,9], V=[6,7,14,15], W=[19,20,22,23]
    strassen_2x2x2([4,5,8,9], [6,7,14,15], [19,20,22,23])
    # A_R*B_BR_right: U=[6,7,10,11], V=[22,23,30,31], W=[19,20,22,23]
    strassen_2x2x2([6,7,10,11], [22,23,30,31], [19,20,22,23])
    
    # Convert to arrays
    rank = len(U_rows)
    U_np = np.array(U_rows).T
    V_np = np.array(V_rows).T
    W_np = np.array(W_rows).T
    
    # Check shapes
    if U_np.shape != (12, rank) or V_np.shape != (32, rank) or W_np.shape != (24, rank):
        return schoolbook()
    
    # Verify
    if verify(U_np.tolist(), V_np.tolist(), W_np.tolist(), rank):
        return {
            "U": to_clean_list(U_np),
            "V": to_clean_list(V_np),
            "W": to_clean_list(W_np)
        }
    
    # Fallback to schoolbook
    return schoolbook()
