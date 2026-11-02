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
    # Strategy: Use 2 Strassen blocks (rank 7 each) for cols 0,1 and 2,3.
    # Then try to decompose cols 4,5,6 with rank 10.
    # Total rank: 7 + 7 + 10 = 24.
    
    def make_rank24_v2():
        rank = 24
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        def add_strassen_block(col_start, k1, k2):
            # Standard Strassen rank 7
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0 + k1, col_start+0] += 1
            V[7*1 + k2, col_start+0] += 1
            W[2*k1 + 0, col_start+0] += 1
            W[2*k2 + 1, col_start+0] += 1

            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0 + k1, col_start+1] += 1
            W[2*k1 + 1, col_start+1] += 1
            W[2*k2 + 1, col_start+1] -= 1

            U[0, col_start+2] += 1
            V[7*0 + k2, col_start+2] += 1
            V[7*1 + k2, col_start+2] -= 1
            W[2*k2 + 0, col_start+2] += 1
            W[2*k2 + 1, col_start+2] += 1

            U[3, col_start+3] += 1
            V[7*1 + k1, col_start+3] += 1
            V[7*0 + k1, col_start+3] -= 1
            W[2*k1 + 0, col_start+3] += 1
            W[2*k1 + 1, col_start+3] += 1

            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1 + k2, col_start+4] += 1
            W[2*k1 + 0, col_start+4] -= 1
            W[2*k2 + 0, col_start+4] += 1

            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0 + k1, col_start+5] += 1
            V[7*0 + k2, col_start+5] += 1
            W[2*k2 + 1, col_start+5] += 1

            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[7*1 + k1, col_start+6] += 1
            V[7*1 + k2, col_start+6] += 1
            W[2*k1 + 0, col_start+6] += 1

        # Block 1: Cols 0, 1
        add_strassen_block(0, 0, 1)
        # Block 2: Cols 2, 3
        add_strassen_block(7, 2, 3)
        
        # Block 3: Cols 4, 5, 6 with Rank 10
        # This is a heuristic attempt. If it fails, verification will catch it.
        # We use 10 terms for the last 3 columns.
        col_start = 14
        k4, k5, k6 = 4, 5, 6
        
        # We can try to combine schoolbook terms for these 3 columns.
        # Schoolbook is 12 terms. We need to save 2.
        # Let's try a specific pattern.
        
        # Terms 0-9 for cols 4,5,6
        # This part is experimental.
        
        # If we can't find a valid rank 10 block, this will fail verification.
        # For now, let's just leave it as schoolbook (12 terms) to ensure validity,
        # but we are trying to fit it in 10 slots.
        # Actually, let's just return a valid rank 25 solution if this doesn't work.
        # To avoid wasting time on a likely invalid construction, we will skip this 
        # if we don't have a known valid rank 10 decomposition.
        
        # Since I don't have a guaranteed valid rank 10 decomposition for 2x2x3,
        # I will not attempt to fill these 10 slots with garbage.
        # Instead, I will return None to indicate failure.
        return None

    U24, V24, W24 = make_rank24_v2()
    if U24 is not None and verify_decomposition(U24, V24, W24):
        best_U, best_V, best_W = U24, V24, W24
        best_rank = 24

    return {"U": best_U, "V": best_V, "W": best_W}
