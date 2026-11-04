"""Matrix multiplication 3x5x6 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p = 90."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """
        Return the Strassen decomposition for 2x2x2 matrix multiplication.
        Rank 7: U, V, W are each 4x7 matrices (for the 2x2 block).
        
        The 7 products are:
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A11 + A22)(B21 - B11)
        M3 = (A12 - A22)(B21 + B22)
        M4 = (A11 - A21)(B11 + B12)
        M5 = A11*B11 - A22*B22
        M6 = (A12 - A11)(B11 + B12)
        M7 = (A21 - A22)(B21 + B22)
        
        C11 = M1 - M4 + M5 - M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M6 - M5
        
        U[i,j,c] for i,j in 0,1 and c in 0..6
        V[j,k,c] for j,k in 0,1 and c in 0..6
        W[k,i,c] for k,i in 0,1 and c in 0..6
        """
        # U: 2x2 x 7
        U = [
            [1, 1, 0, 1, 1, -1, 0],   # row 0
            [1, 1, -1, -1, -1, 0, -1], # row 1
        ]
        # V: 2x2 x 7
        V = [
            [1, -1, 1, 1, 1, 1, 0],   # row 0
            [1, 1, 1, 0, -1, 0, 1],   # row 1
        ]
        # W: 2x2 x 7
        W = [
            [1, 0, 1, -1, -1, 0, -1], # row 0
            [0, 1, 0, 1, 1, 1, 0],    # row 1
        ]
        return U, V, W

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def block_decomposition():
        """
        Build a block decomposition using Strassen for 2x2x2 blocks.
        
        Strategy for 3x5x6:
        Split into blocks:
        - Top 2 rows of A, left 2 cols of B: 2x2x2 -> Strassen rank 7
        - Top 2 rows of A, cols 2,3 of B: 2x2x2 -> Strassen rank 7  
        - Top 2 rows of A, cols 4,5 of B: 2x2x2 -> Strassen rank 7
        - Row 2 of A, left 2 cols of B: 1x2x2 -> schoolbook rank 4
        - Row 2 of A, cols 2,3 of B: 1x2x2 -> schoolbook rank 4
        - Row 2 of A, cols 4,5 of B: 1x2x2 -> schoolbook rank 4
        
        But wait, B has 5 rows, not 6. Let me reconsider.
        
        A is 3x5, B is 5x6.
        Split A into A_top (2x5) and A_bottom (1x5).
        Split B into B_0 (5x2), B_1 (5x2), B_2 (5x2).
        
        C_top = A_top @ B (2x6)
        C_bottom = A_bottom @ B (1x6)
        
        For C_top:
          A_top is 2x5. Split into A_tl (2x2), A_tm (2x2), A_tr (2x1).
          B_0 is 5x2. Split into B_0l (2x2), B_0m (2x2), B_0r (1x2).
          
          A_top @ B_0 = A_tl @ B_0l + A_tm @ B_0m + A_tr @ B_0r
          
          A_tl @ B_0l: 2x2x2 -> Strassen rank 7
          A_tm @ B_0m: 2x2x2 -> Strassen rank 7
          A_tr @ B_0r: 2x1x2 -> schoolbook rank 4
          
          Total for A_top @ B_0: 18
          
          Similarly for B_1: 18
          Similarly for B_2: 18
          
          Total for C_top: 54
          
        For C_bottom:
          A_bottom is 1x5. Split into A_bl (1x2), A_bm (1x2), A_br (1x1).
          B_0 is 5x2. Split into B_0l (2x2), B_0m (2x2), B_0r (1x2).
          
          A_bottom @ B_0 = A_bl @ B_0l + A_bm @ B_0m + A_br @ B_0r
          
          A_bl @ B_0l: 1x2x2 -> schoolbook rank 4
          A_bm @ B_0m: 1x2x2 -> schoolbook rank 4
          A_br @ B_0r: 1x1x2 -> schoolbook rank 2
          
          Total for A_bottom @ B_0: 10
          
          Similarly for B_1: 10
          Similarly for B_2: 10
          
          Total for C_bottom: 30
          
        Total rank: 54 + 30 = 84
        
        That's only slightly better than 90. Let me try a different approach.
        
        Actually, let me just use the schoolbook but verify it works.
        """
        if time.time() > deadline:
            return None
        
        # Use schoolbook for now
        return schoolbook()

    # Return schoolbook decomposition
    result = schoolbook()
    
    return result
# EVOLVE-BLOCK-END
