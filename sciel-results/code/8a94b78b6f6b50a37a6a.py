import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    n, m, p = DIMENSIONS  # 3, 4, 6

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def make_strassen_2x2():
        """
        Return U, V, W for 2x2x2 Strassen with shapes (4,7), (4,7), (4,7).
        U indexed as U[i*2+j], V as V[j*2+k], W as W[k*2+i] for i,j,k in {0,1}.
        Must satisfy: for all (i,j,k) in {0,1}^3, sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1

        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1

        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1

        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1

        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1

        return U, V, W

    def verify_decomposition(U, V, W, rank, n, m, p):
        """Check: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += W[k * n + i, r] * U[i * m + j, r] * V[j * p + k, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def make_blocked_strassen():
        """
        Build a blocked Strassen decomposition for 3x4x6.
        Split: A rows [0,1] and [2], A cols [0,1] and [2,3], B cols [0,1],[2,3],[4,5]
        6 blocks of type (2,2,2) using Strassen (rank 7 each)
        6 blocks of type (1,2,2) using schoolbook (rank 4 each)
        Total rank: 6*7 + 6*4 = 66
        """
        n, m, p = 3, 4, 6
        total_rank = 6 * 7 + 6 * 4  # 66
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)

        # Get Strassen 2x2x2 decomposition
        sU, sV, sW = make_strassen_2x2()

        col = 0
        a_row_blocks = [(0, 2), (2, 3)]  # (start, end)
        a_col_blocks = [(0, 2), (2, 4)]  # (start, end)
        b_col_blocks = [(0, 2), (2, 4), (4, 6)]  # (start, end)

        for (ar_s, ar_e) in a_row_blocks:
            ar_size = ar_e - ar_s  # 2 or 1
            for (ac_s, ac_e) in a_col_blocks:
                ac_size = ac_e - ac_s  # 2
                b_row_s = ac_s  # B rows match A cols
                b_row_e = ac_e
                for (bc_s, bc_e) in b_col_blocks:
                    bc_size = bc_e - bc_s  # 2

                    if ar_size == 2:
                        # Use Strassen for (2,2,2) block
                        for r in range(7):
                            for ii in range(2):
                                for jj in range(2):
                                    U[(ar_s + ii) * m + (ac_s + jj), col] = sU[ii * 2 + jj, r]
                            for jj in range(2):
                                for kk in range(2):
                                    V[(b_row_s + jj) * p + (bc_s + kk), col] = sV[jj * 2 + kk, r]
                            for kk in range(2):
                                for ii in range(2):
                                    W[(bc_s + kk) * n + (ar_s + ii), col] = sW[kk * 2 + ii, r]
                            col += 1
                    else:
                        # ar_size == 1: use schoolbook for (1,2,2) block, rank 4
                        i0 = ar_s
                        for jj in range(2):
                            for kk in range(2):
                                U[i0 * m + (ac_s + jj), col] = 1.0
                                V[(b_row_s + jj) * p + (bc_s + kk), col] = 1.0
                                W[(bc_s + kk) * n + i0, col] = 1.0
                                col += 1

        assert col == total_rank, f"Column count {col} != expected {total_rank}"

        if not verify_decomposition(U, V, W, total_rank, n, m, p):
            return None

        return U, V, W, total_rank

    def make_laderman_3x3():
        """
        Laderman's 23-product algorithm for 3x3 matrix multiplication.
        Reference: Laderman, 1976.
        This is a known decomposition with rank 23.
        
        The 23 products are:
        M1  = (a11+a22+a33)(b11+b22+b33)
        M2  = (a11+a22)(b11+b22)
        M3  = (a11+a33)(b11+b33)
        M4  = (a22+a33)(b22+b33)
        M5  = (a11+a12+a31)(b22+b12+b31)
        M6  = (a11+a13+a21)(b22+b13+b21)
        M7  = (a22+a23+a31)(b11+b23+b31)
        M8  = (a22+a33+a12)(b11+b33+b12)
        M9  = (a22+a33+a13)(b11+b33+b13)
        M10 = (a11+a22+a12+a21)(b11+b22+b12+b21)
        M11 = (a11+a33+a13+a31)(b11+b33+b13+b31)
        M12 = (a22+a33+a23+a32)(b22+b33+b23+b32)
        M13 = (a11+a12+a21+a22+a31+a33)(b11+b12+b21+b22+b31+b33)
        M14 = (a11+a13+a21+a22+a31+a33)(b11+b13+b21+b22+b31+b33)
        M15 = (a22+a23+a32+a33+a11+a12)(b22+b23+b32+b33+b11+b12)
        M16 = (a12+a23)(b12+b23)
        M17 = (a13+a22)(b13+b22)
        M18 = (a21+a33)(b21+b33)
        M19 = (a23+a31)(b23+b31)
        M20 = (a12+a33)(b12+b33)
        M21 = (a13+a21)(b13+b21)
        M22 = (a21+a32)(b21+b32)
        M23 = (a23+a31)(b23+b31) -- Wait, let me check the exact Laderman formula.
        
        Actually, let me use a verified set from literature. The standard Laderman decomposition
        for 3x3x3 with rank 23 is complex. Let me use a different approach.
        
        Instead, let's use the fact that 3x3x3 can be decomposed as:
        - 2x2x2 Strassen (rank 7) for the top-left 2x2 block
        - Plus corrections for the remaining parts
        
        Or better: use a known decomposition.
        
        Let me try a different blocking strategy that might be simpler to verify.
        """
        return None  # Fallback to other methods

    def make_hybrid_decomposition():
        """
        Try a hybrid approach:
        3x4x6 = 3x(2+2)x(2+2+2)
        
        Split A columns into [0,1] and [2,3]
        Split B rows into [0,1] and [2,3]
        Split B columns into [0,1], [2,3], [4,5]
        
        This gives 2x2x2 blocks (6 of them) plus 1x2x2 blocks (6 of them)
        Same as before: 6*7 + 6*4 = 66
        
        Let's try a different split:
        A rows: [0,1,2] (no split)
        A cols: [0,1,2,3] -> [0,1,2] and [3]
        B rows: [0,1,2,3] -> [0,1,2] and [3]
        B cols: [0,1,2,3,4,5] -> [0,1,2] and [3,4,5]
        
        This gives:
        - 3x3x3 block (top-left): need rank for 3x3x3
        - 3x3x3 block (top-right): need rank for 3x3x3
        - 3x1x3 block (bottom-left): rank 1
        - 3x1x3 block (bottom-right): rank 1
        
        If we can get 3x3x3 with rank 23 (Laderman), total = 23+23+1+1 = 48.
        
        Since implementing Laderman is complex, let's try another approach.
        
        Alternative: Use Strassen recursively.
        3x3x3 can be seen as (2+1)x(2+1)x(2+1).
        This gives 8 sub-problems:
        2x2x2: rank 7
        2x2x1: rank 4 (schoolbook for 2x2 matrix times 1x1)
        2x1x2: rank 4
        2x1x1: rank 2
        1x2x2: rank 4
        1x2x1: rank 2
        1x1x2: rank 2
        1x1x1: rank 1
        Total: 7+4+4+2+4+2+2+1 = 26.
        
        So 3x3x3 with rank 26 is easy.
        Then 3x4x6 = 3x3x3 + 3x3x3 + 3x1x3 + 3x1x3 = 26+26+1+1 = 54.
        
        This is better than 66!
        """
        n, m, p = 3, 4, 6
        # 3x3x3 with rank 26 (Strassen-based)
        # 3x1x3 with rank 1
        # Total: 26 + 26 + 1 + 1 = 54
        total_rank = 26 + 26 + 1 + 1  # 54
        
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        
        # Helper to add a 3x3x3 block using Strassen-based decomposition (rank 26)
        def add_3x3x3_block(ar_s, ac_s, bc_s, rank_offset):
            nonlocal col
            # This block handles C[ar_s:ar_s+3, bc_s:bc_s+3] = A[ar_s:ar_s+3, ac_s:ac_s+3] * B[ac_s:ac_s+3, bc_s:bc_s+3]
            # Use the 26-rank decomposition for 3x3x3
            sU, sV, sW = make_strassen_3x3()
            for r in range(26):
                for ii in range(3):
                    for jj in range(3):
                        U[(ar_s + ii) * m + (ac_s + jj), col] = sU[ii * 3 + jj, r]
                for jj in range(3):
                    for kk in range(3):
                        V[(ac_s + jj) * p + (bc_s + kk), col] = sV[jj * 3 + kk, r]
                for kk in range(3):
                    for ii in range(3):
                        W[(bc_s + kk) * n + (ar_s + ii), col] = sW[kk * 3 + ii, r]
                col += 1
        
        def add_3x1x3_block(ar_s, ac_s, bc_s):
            nonlocal col
            # This block handles C[ar_s:ar_s+3, bc_s:bc_s+3] = A[ar_s:ar_s+3, ac_s] * B[ac_s, bc_s:bc_s+3]
            # A is 3x1, B is 1x3
            # C[i,k] = A[i,0] * B[0,k]
            # Rank 1: U[i,0]=1, V[k,0]=1, W[3k+i,0]=1
            for ii in range(3):
                U[(ar_s + ii) * m + ac_s, col] = 1.0
            for kk in range(3):
                V[ac_s * p + (bc_s + kk), col] = 1.0
            for kk in range(3):
                for ii in range(3):
                    W[(bc_s + kk) * n + (ar_s + ii), col] = 1.0
            col += 1
        
        # Block 1: C[0:3, 0:3] = A[0:3, 0:3] * B[0:3, 0:3]
        add_3x3x3_block(0, 0, 0, 0)
        
        # Block 2: C[0:3, 3:6] = A[0:3, 0:3] * B[0:3, 3:6]
        add_3x3x3_block(0, 0, 3, 26)
        
        # Block 3: C[0:3, 0:3] += A[0:3, 3] * B[3, 0:3]
        add_3x1x3_block(0, 3, 0)
        
        # Block 4: C[0:3, 3:6] += A[0:3, 3] * B[3, 3:6]
        add_3x1x3_block(0, 3, 3)
        
        assert col == total_rank, f"Column count {col} != expected {total_rank}"
        
        if not verify_decomposition(U, V, W, total_rank, n, m, p):
            return None
        
        return U, V, W, total_rank

    def make_strassen_3x3():
        """
        Decompose 3x3x3 matrix multiplication using Strassen-style blocking.
        3 = 2 + 1
        This gives 8 sub-problems:
        2x2x2: rank 7
        2x2x1: rank 4
        2x1x2: rank 4
        2x1x1: rank 2
        1x2x2: rank 4
        1x2x1: rank 2
        1x1x2: rank 2
        1x1x1: rank 1
        Total: 26
        """
        U = np.zeros((9, 26))
        V = np.zeros((9, 26))
        W = np.zeros((9, 26))
        
        sU, sV, sW = make_strassen_2x2()
        col = 0
        
        # 2x2x2 block: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        for r in range(7):
            for ii in range(2):
                for jj in range(2):
                    U[ii * 3 + jj, col] = sU[ii * 2 + jj, r]
            for jj in range(2):
                for kk in range(2):
                    V[jj * 3 + kk, col] = sV[jj * 2 + kk, r]
            for kk in range(2):
                for ii in range(2):
                    W[kk * 3 + ii, col] = sW[kk * 2 + ii, r]
            col += 1
        
        # 2x2x1 block: A[0:2,0:2] * B[0:2,2] -> C[0:2,2]
        # C[i,2] = sum_j A[i,j] B[j,2] for i,j in {0,1}
        # This is like multiplying a 2x2 matrix by a 2x1 matrix
        # Rank 4: schoolbook
        for ii in range(2):
            for jj in range(2):
                U[ii * 3 + jj, col] = 1.0
                V[jj * 3 + 2, col] = 1.0
                W[2 * 3 + ii, col] = 1.0
                col += 1
        
        # 2x1x2 block: A[0:2,2] * B[2,0:2] -> C[0:2,0:2]
        # C[i,k] = A[i,2] B[2,k] for i,k in {0,1}
        # Rank 4: schoolbook
        for ii in range(2):
            for kk in range(2):
                U[ii * 3 + 2, col] = 1.0
                V[2 * 3 + kk, col] = 1.0
                W[kk * 3 + ii, col] = 1.0
                col += 1
        
        # 2x1x1 block: A[0:2,2] * B[2,2] -> C[0:2,2]
        # C[i,2] = A[i,2] B[2,2] for i in {0,1}
        # Rank 2
        for ii in range(2):
            U[ii * 3 + 2, col] = 1.0
            V[2 * 3 + 2, col] = 1.0
            W[2 * 3 + ii, col] = 1.0
            col += 1
        
        # 1x2x2 block: A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
        # C[2,k] = sum_j A[2,j] B[j,k] for j,k in {0,1}
        # Rank 4
        for jj in range(2):
            for kk in range(2):
                U[2 * 3 + jj, col] = 1.0
                V[jj * 3 + kk, col] = 1.0
                W[kk * 3 + 2, col] = 1.0
                col += 1
        
        # 1x2x1 block: A[2,0:2] * B[0:2,2] -> C[2,2]
        # C[2,2] = sum_j A[2,j] B[j,2] for j in {0,1}
        # Rank 2
        for jj in range(2):
            U[2 * 3 + jj, col] = 1.0
            V[jj * 3 + 2, col] = 1.0
            W[2 * 3 + 2, col] = 1.0
            col += 1
        
        # 1x1x2 block: A[2,2] * B[2,0:2] -> C[2,0:2]
        # C[2,k] = A[2,2] B[2,k] for k in {0,1}
        # Rank 2
        for kk in range(2):
            U[2 * 3 + 2, col] = 1.0
            V[2 * 3 + kk, col] = 1.0
            W[kk * 3 + 2, col] = 1.0
            col += 1
        
        # 1x1x1 block: A[2,2] * B[2,2] -> C[2,2]
        # C[2,2] = A[2,2] B[2,2]
        # Rank 1
        U[2 * 3 + 2, col] = 1.0
        V[2 * 3 + 2, col] = 1.0
        W[2 * 3 + 2, col] = 1.0
        col += 1
        
        assert col == 26, f"Column count {col} != 26"
        
        return U, V, W

    # Start with schoolbook
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    if time.time() - start_time > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try blocked Strassen (rank 66)
    result = make_blocked_strassen()
    if result is not None:
        bU, bV, bW, brank = result
        if brank < best_rank:
            best_U, best_V, best_W, best_rank = bU, bV, bW, brank
    
    if time.time() - start_time > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try hybrid decomposition (rank 54)
    result = make_hybrid_decomposition()
    if result is not None:
        hU, hV, hW, hrank = result
        if hrank < best_rank:
            best_U, best_V, best_W, best_rank = hU, hV, hW, hrank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
