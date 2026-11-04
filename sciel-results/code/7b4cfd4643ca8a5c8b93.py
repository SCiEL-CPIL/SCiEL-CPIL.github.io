import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    deadline = time.time() + 110  # 10s margin before 120s limit
    
    # Construct the target tensor T of shape (n*m, m*p, n*p)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Function to verify a decomposition
    def verify_decomp(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (n*p, R)
        # T_reconstructed[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        R = U.shape[1]
        T_rec = np.zeros_like(T)
        for r in range(R):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T, T_rec, atol=1e-9)

    # Schoolbook decomposition
    def get_schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R), dtype=np.float64)
        V = np.zeros((m * p, R), dtype=np.float64)
        W = np.zeros((n * p, R), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W

    # Strassen decomposition for 2x2x2
    def get_strassen_222():
        # Returns U, V, W for 2x2x2 matrix multiplication
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices: U[i*2+j], V[i*2+j], W[i*2+j]
        # Strassen's 7 multiplications:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7), dtype=np.float64)
        V = np.zeros((4, 7), dtype=np.float64)
        W = np.zeros((4, 7), dtype=np.float64)
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1  # A00, A11
        V[0, 0] = 1; V[3, 0] = 1  # B00, B11
        W[0, 0] = 1; W[3, 0] = 1  # C00, C11
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1  # A10, A11
        V[0, 1] = 1               # B00
        W[2, 1] = 1; W[3, 1] = -1 # C10, -C11
        
        # M3
        U[0, 2] = 1               # A00
        V[2, 2] = 1; V[3, 2] = -1 # B10, -B11
        W[1, 2] = 1; W[2, 2] = 1  # C01, C10
        
        # M4
        U[3, 3] = 1               # A11
        V[1, 3] = 1; V[0, 3] = -1 # B01, -B00
        W[0, 3] = 1; W[2, 3] = 1  # C00, C10
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1  # A00, A01
        V[3, 4] = 1               # B11
        W[0, 4] = -1; W[1, 4] = 1 # -C00, C01
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1 # A10, -A00
        V[0, 5] = 1; V[1, 5] = 1  # B00, B01
        W[3, 5] = 1               # C11
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1 # A01, -A11
        V[2, 6] = 1; V[3, 6] = 1  # B10, B11
        W[0, 6] = 1               # C00
        
        return U, V, W

    # Combine blocks to form the full decomposition
    def combine_blocks():
        # Partition 3x5x6 into blocks:
        # Block 1: 2x2x2 (i=0,1; j=0,1; k=0,1) -> Use Strassen (rank 7)
        # Block 2: 2x2x4 (i=0,1; j=0,1; k=2,3,4,5) -> Schoolbook (rank 16)
        # Block 3: 1x2x2 (i=2; j=0,1; k=0,1) -> Schoolbook (rank 4)
        # Block 4: 1x2x4 (i=2; j=0,1; k=2,3,4,5) -> Schoolbook (rank 8)
        # Block 5: 2x3x2 (i=0,1; j=2,3,4; k=0,1) -> Schoolbook (rank 12)
        # Block 6: 2x3x4 (i=0,1; j=2,3,4; k=2,3,4,5) -> Schoolbook (rank 24)
        # Block 7: 1x3x2 (i=2; j=2,3,4; k=0,1) -> Schoolbook (rank 6)
        # Block 8: 1x3x4 (i=2; j=2,3,4; k=2,3,4,5) -> Schoolbook (rank 12)
        # Total: 7 + 16 + 4 + 8 + 12 + 24 + 6 + 12 = 89
        
        # Wait, let's recalculate. The total schoolbook is 90.
        # If we replace one 2x2x2 block with Strassen (7 instead of 8), we save 1.
        # So total should be 89.
        
        R_total = 7 + 16 + 4 + 8 + 12 + 24 + 6 + 12
        U = np.zeros((n * m, R_total), dtype=np.float64)
        V = np.zeros((m * p, R_total), dtype=np.float64)
        W = np.zeros((n * p, R_total), dtype=np.float64)
        
        col = 0
        
        # Block 1: 2x2x2 (i=0,1; j=0,1; k=0,1) -> Strassen
        U_str, V_str, W_str = get_strassen_222()
        for r in range(7):
            # Map local indices to global
            # Local i=0,1 -> global i=0,1
            # Local j=0,1 -> global j=0,1
            # Local k=0,1 -> global k=0,1
            for li in range(2):
                for lj in range(2):
                    la = li * 2 + lj
                    ga = li * m + lj
                    U[ga, col + r] = U_str[la, r]
            for li in range(2):
                for lk in range(2):
                    lb = li * 2 + lk
                    gb = li * p + lk
                    V[gb, col + r] = V_str[lb, r]
            for li in range(2):
                for lk in range(2):
                    lc = lk * 2 + li
                    gc = lk * n + li
                    W[gc, col + r] = W_str[lc, r]
        col += 7
        
        # Block 2: 2x2x4 (i=0,1; j=0,1; k=2,3,4,5) -> Schoolbook
        for li in range(2):
            for lj in range(2):
                for lk in range(4):
                    gi = li
                    gj = lj
                    gk = 2 + lk
                    ga = gi * m + gj
                    gb = gj * p + gk
                    gc = gk * n + gi
                    U[ga, col] = 1.0
                    V[gb, col] = 1.0
                    W[gc, col] = 1.0
                    col += 1
        
        # Block 3: 1x2x2 (i=2; j=0,1; k=0,1) -> Schoolbook
        for lj in range(2):
            for lk in range(2):
                gi = 2
                gj = lj
                gk = lk
                ga = gi * m + gj
                gb = gj * p + gk
                gc = gk * n + gi
                U[ga, col] = 1.0
                V[gb, col] = 1.0
                W[gc, col] = 1.0
                col += 1
        
        # Block 4: 1x2x4 (i=2; j=0,1; k=2,3,4,5) -> Schoolbook
        for lj in range(2):
            for lk in range(4):
                gi = 2
                gj = lj
                gk = 2 + lk
                ga = gi * m + gj
                gb = gj * p + gk
                gc = gk * n + gi
                U[ga, col] = 1.0
                V[gb, col] = 1.0
                W[gc, col] = 1.0
                col += 1
        
        # Block 5: 2x3x2 (i=0,1; j=2,3,4; k=0,1) -> Schoolbook
        for li in range(2):
            for lj in range(3):
                for lk in range(2):
                    gi = li
                    gj = 2 + lj
                    gk = lk
                    ga = gi * m + gj
                    gb = gj * p + gk
                    gc = gk * n + gi
                    U[ga, col] = 1.0
                    V[gb, col] = 1.0
                    W[gc, col] = 1.0
                    col += 1
        
        # Block 6: 2x3x4 (i=0,1; j=2,3,4; k=2,3,4,5) -> Schoolbook
        for li in range(2):
            for lj in range(3):
                for lk in range(4):
                    gi = li
                    gj = 2 + lj
                    gk = 2 + lk
                    ga = gi * m + gj
                    gb = gj * p + gk
                    gc = gk * n + gi
                    U[ga, col] = 1.0
                    V[gb, col] = 1.0
                    W[gc, col] = 1.0
                    col += 1
        
        # Block 7: 1x3x2 (i=2; j=2,3,4; k=0,1) -> Schoolbook
        for lj in range(3):
            for lk in range(2):
                gi = 2
                gj = 2 + lj
                gk = lk
                ga = gi * m + gj
                gb = gj * p + gk
                gc = gk * n + gi
                U[ga, col] = 1.0
                V[gb, col] = 1.0
                W[gc, col] = 1.0
                col += 1
        
        # Block 8: 1x3x4 (i=2; j=2,3,4; k=2,3,4,5) -> Schoolbook
        for lj in range(3):
            for lk in range(4):
                gi = 2
                gj = 2 + lj
                gk = 2 + lk
                ga = gi * m + gj
                gb = gj * p + gk
                gc = gk * n + gi
                U[ga, col] = 1.0
                V[gb, col] = 1.0
                W[gc, col] = 1.0
                col += 1
        
        return U, V, W

    best_U, best_V, best_W = get_schoolbook()
    best_rank = n * m * p
    
    # Try the block decomposition
    if time.time() < deadline - 5:
        U_block, V_block, W_block = combine_blocks()
        if verify_decomp(U_block, V_block, W_block):
            best_U, best_V, best_W = U_block, V_block, W_block
            best_rank = U_block.shape[1]
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
