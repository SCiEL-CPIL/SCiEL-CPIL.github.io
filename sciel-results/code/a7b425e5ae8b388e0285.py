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

    # Schoolbook construction (rank 28) as fallback
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

    # Try to achieve rank 24 using 2x2x4 (rank 11) + 2x2x3 (rank 8) = 19? 
    # Actually, let me first try to get a valid rank 24 decomposition.
    # Strategy: Use Strassen (rank 7) for columns 0-1, Strassen (rank 7) for columns 2-3,
    # and a rank 10 decomposition for columns 4-6 (2x2x3).
    # Total: 7 + 7 + 10 = 24.
    
    # First, let me construct a rank-8 decomposition for 2x2x3.
    # Known optimal: rank 8 for 2x2x3.
    
    def make_rank24():
        rank = 24
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        def add_strassen_block(col_start, k_start, k_end):
            """
            Add Strassen rank-7 terms for columns k_start to k_end (exclusive).
            Must be exactly 2 columns.
            """
            # M1 = (A00 + A11) * (B0,ks + B1,ke-1)
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1
            V[7*1 + (k_end-1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1
            W[2*(k_end-1) + 1, col_start+0] += 1

            # M2 = (A10 + A11) * B0,ks
            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1
            W[2*(k_end-1) + 1, col_start+1] -= 1

            # M3 = A00 * (B0,ke-1 - B1,ke-1)
            U[0, col_start+2] += 1
            V[7*0 + (k_end-1), col_start+2] += 1
            V[7*1 + (k_end-1), col_start+2] -= 1
            W[2*(k_end-1) + 0, col_start+2] += 1
            W[2*(k_end-1) + 1, col_start+2] += 1

            # M4 = A11 * (B1,ks - B0,ks)
            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1
            V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1
            W[2*k_start + 1, col_start+3] += 1

            # M5 = (A00 + A01) * B1,ke-1
            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1 + (k_end-1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1
            W[2*(k_end-1) + 0, col_start+4] += 1

            # M6 = (A10 - A00) * (B0,ks + B0,ke-1)
            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1
            V[7*0 + (k_end-1), col_start+5] += 1
            W[2*(k_end-1) + 1, col_start+5] += 1

            # M7 = (A01 - A11) * (B1,ks + B1,ke-1)
            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1
            V[7*1 + (k_end-1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1

        # Apply Strassen to pairs (0,1) and (2,3)
        add_strassen_block(0, 0, 2)
        add_strassen_block(7, 2, 4)
        
        # For columns 4,5,6: use a rank-10 decomposition for 2x2x3
        # This is more complex. Let me use a known rank-8 decomposition for 2x2x3.
        # Actually, let me try to find a rank 8 decomposition for 2x2x3.
        
        # Known: 2x2x3 has rank 8.
        # Let me construct it.
        
        col_start = 14
        k_start = 4
        
        # I'll use a construction based on the known optimal algorithm.
        # For 2x2x3, we can use:
        # M1 = (A00+A11)*(B0,4+B1,5) -> C0,4, C1,5
        # M2 = (A10+A11)*B0,4 -> C1,4, C1,5
        # M3 = A00*(B0,5-B1,5) -> C0,5, C1,5
        # M4 = A11*(B1,4-B0,4) -> C0,4, C1,4
        # M5 = (A00+A01)*B1,5 -> C0,4, C0,5
        # M6 = (A10-A00)*(B0,4+B0,5) -> C1,5
        # M7 = (A01-A11)*(B1,4+B1,5) -> C0,4
        # That's 7, but we need to handle column 6 too.
        
        # Let me just use schoolbook for the last 3 columns to be safe: rank 12
        # Total would be 7+7+12 = 26, which is worse than 25.
        
        # So let me stick with the rank 25 approach but try to optimize.
        # Actually, the current rank 25 is already good. Let me just return it.
        
        # For now, fall back to the known rank 25 construction
        return None

    # Use the proven rank 25 construction
    def make_rank25():
        rank = 25
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        def add_strassen_block(col_start, k_start):
            """
            Add Strassen rank-7 terms for a 2x2 block of B (columns k_start, k_start+1).
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

    return {"U": best_U, "V": best_V, "W": best_W}
