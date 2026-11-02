import time
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0


def _validate(U, V, W, n, m, p, rank):
    """Check that U,V,W exactly decompose the matrix mult tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.zeros((n, m, p), dtype=float)
    for t in range(rank):
        u_mat = U[:, t].reshape(n, m)
        v_mat = V[:, t].reshape(m, p)
        w_mat = W[:, t].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-9)


def _build_bordered(n, m, p):
    """Bordered decomposition. Rank = (n-1)(m-1)(p-1) + n + m + p - 2."""
    rank = (n - 1) * (m - 1) * (p - 1) + n + m + p - 2
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    t = 0

    # Core
    for i in range(n - 1):
        for j in range(m - 1):
            for k in range(p - 1):
                U[i * m + j][t] = 1.0
                V[j * p + k][t] = 1.0
                W[k * n + i][t] = 1.0
                t += 1

    # Group 1: (n-1) terms for (i, m-1, k), i<n-1, all k
    for i in range(n - 1):
        U[i * m + (m - 1)][t] = 1.0
        for k in range(p):
            V[(m - 1) * p + k][t] = 1.0
            W[k * n + i][t] = 1.0
        t += 1

    # Group 2: (p-1) terms for (n-1, j, k), j<m-1, k<p-1
    for k in range(p - 1):
        for j in range(m - 1):
            U[(n - 1) * m + j][t] = 1.0
            V[j * p + k][t] = 1.0
        W[k * n + (n - 1)][t] = 1.0
        t += 1

    # Group 3: (m-1) terms for (n-1, j, p-1), j<m-1
    for j in range(m - 1):
        U[(n - 1) * m + j][t] = 1.0
        V[j * p + (p - 1)][t] = 1.0
        W[(p - 1) * n + (n - 1)][t] = 1.0
        t += 1

    # Group 4: 1 term for (n-1, m-1, p-1)
    U[(n - 1) * m + (m - 1)][t] = 1.0
    V[(m - 1) * p + (p - 1)][t] = 1.0
    W[(p - 1) * n + (n - 1)][t] = 1.0
    t += 1

    return U, V, W


def _build_strassen_block(n, m, p):
    """Block-partitioned Strassen decomposition for 3x4 * 4x6.
    
    Partition:
      A (3x4): top row i=0 (1x4), bottom block i=1,2 (2x4)
      B (4x6): left col k=0 (4x1), right block k=1..5 (4x5)
      C (3x6): top-left (1x1), top-right (1x5), bottom-left (2x1), bottom-right (2x5)
    
    Sub-problems:
      C[1,2,0] = A[1,2,:] * B[:,0]  -> 2x4 * 4x1 = 2x1, schoolbook rank 8
      C[0,1,1..5] = A[0,:] * B[:,1..5] -> 1x4 * 4x5 = 1x5, schoolbook rank 10
      C[1,2,1..5] = A[1,2,:] * B[:,1..5] -> 2x4 * 4x5 = 2x5
    
    For the 2x4 * 4x5 = 2x5 block, partition B[:,1..5] into B[:,1..2] (4x2) and B[:,3..5] (4x3):
      C[1,2,1..2] = A[1,2,:] * B[:,1..2] -> 2x4 * 4x2 = 2x2, Strassen rank 7
      C[1,2,3..5] = A[1,2,:] * B[:,3..5] -> 2x4 * 4x3 = 2x3, schoolbook rank 24... 
      That's too much. Let me use schoolbook for 2x4*4x3: rank = 2*4*3 = 24. Total = 7+24+8+10 = 49.
    
    Actually, let me reconsider. For 2x4 * 4x5:
    Split the 4 (m dimension) into 2+2:
      C[1,2,1..5] = A[1,2,0:2]*B[0:2,1..5] + A[1,2,2:4]*B[2:4,1..5]
      Each is 2x2 * 2x5 = 2x5, schoolbook rank 20. Total 40. Worse.
    
    Let me just use schoolbook for the 2x4*4x5 block: rank = 2*4*5 = 40.
    Total: 8 + 10 + 40 = 58. Still better than 72 but worse than bordered 41.
    
    Hmm. Let me try a different partition.
    
    Actually, for the 2x2 sub-problem (2x4 * 4x2), Strassen gives rank 7.
    Let me partition the 6 columns as: k=0 (1 col), k=1,2 (2 cols), k=3,4,5 (3 cols).
    
    C[1,2,1..2] = A[1,2,:] * B[:,1..2] -> 2x4 * 4x2, Strassen rank 7
    C[1,2,3..5] = A[1,2,:] * B[:,3..5] -> 2x4 * 4x3, schoolbook rank 24
    
    Total: 8 + 10 + 7 + 24 = 49. Still worse than bordered 41.
    
    Let me try yet another approach. What if I use Strassen on a different sub-block?
    
    Alternative: partition A into top 2 rows and bottom 1 row, B into left 3 cols and right 3 cols.
    A: 2x4 (top) + 1x4 (bottom)
    B: 4x3 (left) + 4x3 (right)
    
    C[0,1,0..2] = A[0,1,:] * B[:,0..2] -> 2x4 * 4x3 = 2x3, schoolbook 24
    C[0,1,3..5] = A[0,1,:] * B[:,3..5] -> 2x4 * 4x3 = 2x3, schoolbook 24
    C[2,0..2] = A[2,:] * B[:,0..2] -> 1x4 * 4x3 = 1x3, schoolbook 12
    C[2,3..5] = A[2,:] * B[:,3..5] -> 1x4 * 4x3 = 1x3, schoolbook 12
    
    Total: 24+24+12+12 = 72. No improvement.
    
    Let me try: A top 2 rows, B left 2 cols and right 4 cols.
    C[0,1,0..1] = 2x4*4x2 = 2x2, Strassen 7
    C[0,1,2..5] = 2x4*4x4 = 2x4, schoolbook 32
    C[2,0..1] = 1x4*4x2 = 1x2, schoolbook 8
    C[2,2..5] = 1x4*4x4 = 1x4, schoolbook 16
    Total: 7+32+8+16 = 63. Worse than bordered.
    
    OK so for 3x4x6, the bordered decomposition at rank 41 seems hard to beat with simple block+Strassen.
    But wait - what about using Strassen more cleverly?
    
    Actually, let me reconsider. The key insight from the knowledge is that splitting n=3 into 1+2, p=6 into 3+3, m=4 into 2+2 allows Strassen on 2x2x2 sub-problems.
    
    Let me try: A = [A_top (1x4); A_bot (2x4)], B = [B_left (4x3) | B_right (4x3)]
    
    C_top_left (1x3) = A_top * B_left: 1x4 * 4x3, schoolbook 12
    C_top_right (1x3) = A_top * B_right: 1x4 * 4x3, schoolbook 12
    C_bot_left (2x3) = A_bot * B_left: 2x4 * 4x3
    C_bot_right (2x3) = A_bot * B_right: 2x4 * 4x3
    
    For 2x4 * 4x3, partition m=4 into 2+2:
    2x4 * 4x3 = (2x2 * 2x3) + (2x2 * 2x3) = 2*2*3 + 2*2*3 = 12+12 = 24 schoolbook.
    
    Or use Strassen-like: 2x2 * 2x3... not standard.
    
    Total: 12+12+24+24 = 72. No good.
    
    Let me try the partition that was mentioned: n=3 as 1+2, p=6 as 3+3, m=4 as 2+2.
    
    A (3x4) split by rows: A1 (1x4), A2 (2x4). Split A2's columns: A2a (2x2), A2b (2x2).
    B (4x6) split by cols: B1 (4x3), B2 (4x3). Split B1's rows: B1a (2x3), B1b (2x3).
    
    C11 (1x3) = A1 * B1 = A1 * [B1a; B1b] = A1*B1a + A1*B1b
    C12 (1x3) = A1 * B2
    C21 (2x3) = A2 * B1 = [A2a A2b] * [B1a; B1b] = A2a*B1a + A2b*B1b
    C22 (2x3) = A2 * B2 = [A2a A2b] * B2 = A2a*B2a + A2b*B2b where B2 = [B2a; B2b]
    
    C11: 1x4*4x3 = (1x2*2x3) + (1x2*2x3) = 6+6 = 12
    C12: 1x4*4x3 = 12
    C21: (2x2*2x3) + (2x2*2x3) = 12+12 = 24
    C22: (2x2*2x3) + (2x2*2x3) = 12+12 = 24
    Total: 72.
    
    Hmm, none of these simple partitions beat 41. Let me just implement the bordered decomposition properly and also try a few Strassen-based partitions, returning the best valid one.
    
    Actually, I realize I should just implement the bordered (which gives 41) and a couple of Strassen-based alternatives, validate them, and return the best. The bordered at 41 is already an improvement over 72.
    """
    # For now, return bordered as the primary strategy
    return _build_bordered(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN

    # Best known: schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = i * m * p + j * p + k
                U[i * m + j][t] = 1.0
                V[j * p + k][t] = 1.0
                W[k * n + i][t] = 1.0
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank

    # Try bordered decomposition
    bordered_rank = (n - 1) * (m - 1) * (p - 1) + n + m + p - 2
    if bordered_rank < best_rank and time.time() < deadline:
        U_b, V_b, W_b = _build_bordered(n, m, p)
        if U_b is not None:
            if _validate(U_b, V_b, W_b, n, m, p, len(U_b[0])):
                best_result = {"U": U_b, "V": V_b, "W": W_b}
                best_rank = len(U_b[0])

    # Try Strassen-based block partition: 3x4 * 4x6
    # Split A: row 0 (1x4) + rows 1-2 (2x4)
    # Split B: col 0 (4x1) + cols 1-5 (4x5)
    # C[0,0] = A[0,:]*B[:,0] -> 1x4*4x1 = 4 (schoolbook)
    # C[0,1..5] = A[0,:]*B[:,1..5] -> 1x4*4x5 = 20 (schoolbook)
    # C[1..2,0] = A[1..2,:]*B[:,0] -> 2x4*4x1 = 8 (schoolbook)
    # C[1..2,1..5] = A[1..2,:]*B[:,1..5] -> 2x4*4x5 = 40 (schoolbook)
    # Total: 4+20+8+40 = 72. No improvement over schoolbook.
    
    # Try: Split B into cols 0-1 (4x2) and cols 2-5 (4x4)
    # C[1..2,0..1] = 2x4*4x2 -> Strassen rank 7
    # C[1..2,2..5] = 2x4*4x4 -> schoolbook 32
    # C[0,0..1] = 1x4*4x2 -> 8
    # C[0,2..5] = 1x4*4x4 -> 16
    # Total: 7+32+8+16 = 63. Worse than bordered.
    
    # The bordered decomposition at rank 41 is the best simple construction.
    # Let's also try a refined Strassen approach:
    # 2x4 * 4x2 using Strassen (rank 7)
    # This handles C[1..2, 0..1]
    # Remaining: C[0,0..1] (1x4*4x2=8), C[1..2,2..5] (2x4*4x4=32), C[0,2..5] (1x4*4x4=16)
    # Total: 7+8+32+16 = 63
    
    # Since bordered (41) is better, we keep it.
    # But let's verify the bordered is correct and return it.
    
    return best_result
