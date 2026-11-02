import time
import numpy as np

def verify_decomposition(U, V, W, n=2, m=2, p=7):
    """
    Verify if U, V, W form a valid decomposition of the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill

    n, m, p = 2, 2, 7
    schoolbook_rank = n * m * p  # 28

    # Schoolbook construction (rank 28)
    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = make_schoolbook()
    best_rank = schoolbook_rank

    # Rank 25 construction using Strassen blocks for pairs of columns
    def make_rank25():
        rank = 25
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        def add_strassen_block(col_start, k_start):
            """
            Add Strassen rank-7 terms for a 2x2 block of B (columns k_start, k_start+1).
            col_start: starting column index in U, V, W for this block's terms.
            k_start: starting column index in B (and C).
            """
            # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1
            V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1
            W[2*(k_start+1) + 1, col_start+0] += 1

            # M2 = (A10 + A11) * B0,ks
            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1
            W[2*(k_start+1) + 1, col_start+1] -= 1

            # M3 = A00 * (B0,ks+1 - B1,ks+1)
            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1
            V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1
            W[2*(k_start+1) + 1, col_start+2] += 1

            # M4 = A11 * (B1,ks - B0,ks)
            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1
            V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1
            W[2*k_start + 1, col_start+3] += 1

            # M5 = (A00 + A01) * B1,ks+1
            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1
            W[2*(k_start+1) + 0, col_start+4] += 1

            # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1
            V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1

            # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1
            V[7*1 + (k_start+1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1

        # Apply Strassen to pairs (0,1), (2,3), (4,5)
        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        add_strassen_block(14, 4)
        
        # Apply schoolbook to column 6
        col_start = 21
        k = 6
        
        # C0,6 = A00*B0,6 + A01*B1,6
        U[0, col_start+0] += 1
        V[7*0 + k, col_start+0] += 1
        W[2*k + 0, col_start+0] += 1
        
        U[1, col_start+1] += 1
        V[7*1 + k, col_start+1] += 1
        W[2*k + 0, col_start+1] += 1
        
        # C1,6 = A10*B0,6 + A11*B1,6
        U[2, col_start+2] += 1
        V[7*0 + k, col_start+2] += 1
        W[2*k + 1, col_start+2] += 1
        
        U[3, col_start+3] += 1
        V[7*1 + k, col_start+3] += 1
        W[2*k + 1, col_start+3] += 1

        return U.tolist(), V.tolist(), W.tolist()

    # Try rank 25
    U25, V25, W25 = make_rank25()
    if verify_decomposition(U25, V25, W25):
        best_U, best_V, best_W = U25, V25, W25
        best_rank = 25
    else:
        # If rank 25 fails verification (shouldn't happen), fall back to schoolbook
        pass

    # Try to find rank 24 by using a unified decomposition across all 7 columns.
    # Strategy: Use 3 overlapping Strassen-like blocks with shared bilinear forms.
    # Block 1: columns 0,1 (rank 7)
    # Block 2: columns 1,2 (rank 7, shares some with block 1)
    # Block 3: columns 3,4 (rank 7)
    # Block 4: columns 4,5 (rank 7, shares some with block 3)
    # Block 5: columns 5,6 (rank 7, shares some with block 4)
    # Total unique: 7 + 7 + 7 + 7 + 7 = 35, but with sharing we aim for 24.
    
    # Actually, let's try a different approach: use 4 groups of 2 columns with 1 overlap
    # Group A: cols 0,1 -> rank 7
    # Group B: cols 2,3 -> rank 7  
    # Group C: cols 4,5 -> rank 7
    # Col 6: rank 4
    # Total: 25 (same as before)
    
    # To get 24, we need to save 1 more term. Let's try mixing col 5 and 6.
    # Group A: cols 0,1 -> rank 7
    # Group B: cols 2,3 -> rank 7
    # Group C: cols 4,5 -> rank 6 (try to reduce by 1)
    # Col 6: rank 4
    # Total: 24
    
    def make_rank24():
        rank = 24
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        def add_strassen_block(col_start, k1, k2, use_6=True):
            """
            Add Strassen-like terms for columns k1, k2.
            If use_6=True, try to use only 6 terms instead of 7.
            """
            if use_6:
                # Try a rank-6 decomposition for 2x2x2 (might not work, but let's try)
                # Standard Strassen uses 7. For 6, we need a clever combination.
                # This is known to be impossible for exact 2x2x2, but let's try anyway.
                pass
            else:
                # Standard Strassen rank 7
                # M1 = (A00 + A11) * (B0,k1 + B1,k2)
                U[0, col_start+0] += 1
                U[3, col_start+0] += 1
                V[7*0 + k1, col_start+0] += 1
                V[7*1 + k2, col_start+0] += 1
                W[2*k1 + 0, col_start+0] += 1
                W[2*k2 + 1, col_start+0] += 1

                # M2 = (A10 + A11) * B0,k1
                U[2, col_start+1] += 1
                U[3, col_start+1] += 1
                V[7*0 + k1, col_start+1] += 1
                W[2*k1 + 1, col_start+1] += 1
                W[2*k2 + 1, col_start+1] -= 1

                # M3 = A00 * (B0,k2 - B1,k2)
                U[0, col_start+2] += 1
                V[7*0 + k2, col_start+2] += 1
                V[7*1 + k2, col_start+2] -= 1
                W[2*k2 + 0, col_start+2] += 1
                W[2*k2 + 1, col_start+2] += 1

                # M4 = A11 * (B1,k1 - B0,k1)
                U[3, col_start+3] += 1
                V[7*1 + k1, col_start+3] += 1
                V[7*0 + k1, col_start+3] -= 1
                W[2*k1 + 0, col_start+3] += 1
                W[2*k1 + 1, col_start+3] += 1

                # M5 = (A00 + A01) * B1,k2
                U[0, col_start+4] += 1
                U[1, col_start+4] += 1
                V[7*1 + k2, col_start+4] += 1
                W[2*k1 + 0, col_start+4] -= 1
                W[2*k2 + 0, col_start+4] += 1

                # M6 = (A10 - A00) * (B0,k1 + B0,k2)
                U[2, col_start+5] += 1
                U[0, col_start+5] -= 1
                V[7*0 + k1, col_start+5] += 1
                V[7*0 + k2, col_start+5] += 1
                W[2*k2 + 1, col_start+5] += 1

                # M7 = (A01 - A11) * (B1,k1 + B1,k2)
                U[1, col_start+6] += 1
                U[3, col_start+6] -= 1
                V[7*1 + k1, col_start+6] += 1
                V[7*1 + k2, col_start+6] += 1
                W[2*k1 + 0, col_start+6] += 1

        # Apply standard Strassen to pairs (0,1), (2,3), (4,5)
        add_strassen_block(0, 0, 1, use_6=False)
        add_strassen_block(7, 2, 3, use_6=False)
        
        # For columns 4,5, try a modified approach
        # Use 6 terms by sharing with column 6
        col_start = 14
        k4, k5 = 4, 5
        
        # M1 = (A00 + A11) * (B0,k4 + B1,k5)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k4, col_start+0] += 1
        V[7*1 + k5, col_start+0] += 1
        W[2*k4 + 0, col_start+0] += 1
        W[2*k5 + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,k4
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k4, col_start+1] += 1
        W[2*k4 + 1, col_start+1] += 1
        W[2*k5 + 1, col_start+1] -= 1

        # M3 = A00 * (B0,k5 - B1,k5)
        U[0, col_start+2] += 1
        V[7*0 + k5, col_start+2] += 1
        V[7*1 + k5, col_start+2] -= 1
        W[2*k5 + 0, col_start+2] += 1
        W[2*k5 + 1, col_start+2] += 1

        # M4 = A11 * (B1,k4 - B0,k4)
        U[3, col_start+3] += 1
        V[7*1 + k4, col_start+3] += 1
        V[7*0 + k4, col_start+3] -= 1
        W[2*k4 + 0, col_start+3] += 1
        W[2*k4 + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,k5
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + k5, col_start+4] += 1
        W[2*k4 + 0, col_start+4] -= 1
        W[2*k5 + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,k4 + B0,k5)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k4, col_start+5] += 1
        V[7*0 + k5, col_start+5] += 1
        W[2*k5 + 1, col_start+5] += 1

        # Column 6: use 4 schoolbook terms
        col_start = 20
        k6 = 6
        
        # C0,6 = A00*B0,6 + A01*B1,6
        U[0, col_start+0] += 1
        V[7*0 + k6, col_start+0] += 1
        W[2*k6 + 0, col_start+0] += 1
        
        U[1, col_start+1] += 1
        V[7*1 + k6, col_start+1] += 1
        W[2*k6 + 0, col_start+1] += 1
        
        # C1,6 = A10*B0,6 + A11*B1,6
        U[2, col_start+2] += 1
        V[7*0 + k6, col_start+2] += 1
        W[2*k6 + 1, col_start+2] += 1
        
        U[3, col_start+3] += 1
        V[7*1 + k6, col_start+3] += 1
        W[2*k6 + 1, col_start+3] += 1

        return U.tolist(), V.tolist(), W.tolist()

    U24, V24, W24 = make_rank24()
    if verify_decomposition(U24, V24, W24):
        best_U, best_V, best_W = U24, V24, W24
        best_rank = 24

    return {"U": best_U, "V": best_V, "W": best_W}
