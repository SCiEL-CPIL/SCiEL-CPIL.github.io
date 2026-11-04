"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_strassen_2x2x2():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0] * rank for _ in range(4)]  # 2*2 rows
        V = [[0] * rank for _ in range(4)]  # 2*2 rows
        W = [[0] * rank for _ in range(4)]  # 2*2 rows

        # Standard Strassen terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)

        # U encodes A coefficients: U[i*2+j, term]
        # V encodes B coefficients: V[j*2+k, term]
        # W encodes C coefficients: W[k*2+i, term]

        # Term 0: M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # Term 1: M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1

        # Term 2: M3 = A00(B10-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1

        # Term 3: M4 = A11(B01-B10)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1

        # Term 4: M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1

        # Term 5: M6 = (A10-A00)(B00+B10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # Term 6: M7 = (A01-A11)(B01+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1

        return U, V, W

    def build_2x2x3():
        """Decomposition for 2x2 * 2x3 -> 2x3.
        Split B into B1 (2x2) and B2 (2x1).
        Use Strassen for 2x2*2x2 and schoolbook for 2x2*2x1.
        """
        U_s, V_s, W_s = build_strassen_2x2x2()
        rank_s = 7

        # Schoolbook for 2x2 * 2x1 -> 2x1 (rank 4)
        rank_sb = 4
        U_sb = [[0] * rank_sb for _ in range(4)]
        V_sb = [[0] * rank_sb for _ in range(2)]
        W_sb = [[0] * rank_sb for _ in range(2)]
        U_sb[0][0] = 1; V_sb[0][0] = 1; W_sb[0][0] = 1
        U_sb[1][1] = 1; V_sb[1][1] = 1; W_sb[0][1] = 1
        U_sb[2][2] = 1; V_sb[0][2] = 1; W_sb[1][2] = 1
        U_sb[3][3] = 1; V_sb[1][3] = 1; W_sb[1][3] = 1

        total_rank = rank_s + rank_sb
        U = [[0] * total_rank for _ in range(4)]   # 2*2
        V = [[0] * total_rank for _ in range(6)]   # 2*3
        W = [[0] * total_rank for _ in range(6)]   # 3*2

        # Strassen part: columns 0..6
        for c in range(rank_s):
            for r in range(4):
                U[r][c] = U_s[r][c]
            for r in range(4):
                V[r][c] = V_s[r][c]
            for r in range(4):
                W[r][c] = W_s[r][c]

        # Schoolbook part: columns 7..10
        # V_sb has 2 rows (j=0,1), but B has 3 columns (k=0,1,2)
        # We want B2 to be the last column (k=2)
        # V_sb[j, c] should map to V[j*3+2, 7+c]
        for c in range(rank_sb):
            for r in range(4):
                U[r][rank_s + c] = U_sb[r][c]
            for j in range(2):
                V[j * 3 + 2][rank_s + c] = V_sb[j][c]
            for r in range(2):
                W[r * 2 + 0][rank_s + c] = 0  # W_sb is 2x4, W is 6x...
                # W_sb[k*2+i] where k=0,1 and i=0,1
                # W has rows indexed by k*2+i for C[k,i]
                # C is 2x3, so W has 3*2=6 rows, indexed by k*2+i
                # W_sb has 2 rows (i=0,1) for fixed k... no
                # W_sb[k*2+i] doesn't make sense since W_sb is 2x4
                # W_sb[i, c] for i in 0..1 (output rows of C_strip)
                # C_strip is 2x1, so output has 2 entries: C[0,2] and C[1,2]
                # In full C (2x3), these are at positions k=2, i=0 and k=2, i=1
                # W row index = k*2 + i = 2*2+0=4 and 2*2+1=5
                pass
            # Fix W mapping
            W[4][rank_s + c] = W_sb[0][c]  # C[0,2]
            W[5][rank_s + c] = W_sb[1][c]  # C[1,2]

        return U, V, W

    def build_4x4x6():
        """Decomposition for 4x4 * 4x6 -> 4x6 using Strassen recursion.
        Split A into 2x2 blocks, B into 2x3 blocks, C into 2x3 blocks.
        A = [[A00, A01], [A10, A11]] each 2x2
        B = [[B00, B01], [B10, B11]] each 2x3
        C = [[C00, C01], [C10, C11]] each 2x3
        
        C00 = A00*B00 + A01*B10
        C01 = A00*B01 + A01*B11
        C10 = A10*B00 + A11*B10
        C11 = A10*B01 + A11*B11
        
        Using Strassen on the 2x2 block structure:
        M1 = (A00+A11)(B00+B10) -> contributes to C00 and C11
        M2 = (A10+A11)B00 -> contributes to C10
        M3 = A00(B10-B11) -> contributes to C01
        M4 = A11(B01-B10) -> contributes to C00
        M5 = (A00+A01)B11 -> contributes to C01
        M6 = (A10-A00)(B00+B10) -> contributes to C11
        M7 = (A01-A11)(B01+B11) -> contributes to C00
        
        Wait, this is for 2x2 * 2x2 -> 2x2 where all blocks are same size.
        Here B and C blocks are 2x3, so we need 2x2 * 2x3 multiplication.
        
        Let me use a simpler approach: just use schoolbook for the whole thing
        but with a better structure. Actually, let me just use the direct
        schoolbook decomposition which is guaranteed correct.
        """
        # Schoolbook: rank = n*m*p = 4*4*6 = 96
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
        
        return U, V, W

    # Use schoolbook as a safe fallback, then try to improve
    U, V, W = build_4x4x6()

    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
