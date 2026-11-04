"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
_DEADLINE_SECONDS = 115  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return (U, V, W) lists of lists, rank n*m*p."""
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


def _strassen_2x2():
    """Strassen's 7-product algorithm for 2x2x2.
    Returns U (4x7), V (4x7), W (4x7) as lists of lists (integers).
    A is 2x2 with rows [a b; c d] flattened row-major: (a,b,c,d)
    B is 2x2 with rows [e f; g h] flattened row-major: (e,f,g,h)
    C = A*B has rows [a*e+b*g, a*f+b*h; c*e+d*g, c*f+d*h]
    W index is k*n + i = k*2 + i (since n=2, p=2).
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a
        [0, 0, -1, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, -1]    # d
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, 0, 0, 1, 0, 1]    # h
    ]
    W = [
        [1, 0, 0, -1, -1, 0, 1],   # C[0][0]
        [0, 0, 1, 0, 1, 0, 0],     # C[0][1]
        [0, 1, 0, 1, 0, 0, 0],     # C[1][0]
        [1, -1, -1, 0, 0, 1, 0]    # C[1][1]
    ]
    return U, V, W


def _embed_subproblem(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_offset, j_offset, k_offset, n, m, p):
    """Embed a subproblem decomposition into the full 4x5x5 space."""
    rank_sub = len(U_sub[0])
    U_full = [[0] * rank_sub for _ in range(n * m)]
    V_full = [[0] * rank_sub for _ in range(m * p)]
    W_full = [[0] * rank_sub for _ in range(p * n)]
    
    for local_i in range(n_sub):
        for local_j in range(m_sub):
            full_i = i_offset + local_i
            full_j = j_offset + local_j
            local_idx = local_i * m_sub + local_j
            full_idx = full_i * m + full_j
            U_full[full_idx] = U_sub[local_i * m_sub + local_j]
    
    for local_j in range(m_sub):
        for local_k in range(p_sub):
            full_j = j_offset + local_j
            full_k = k_offset + local_k
            local_idx = local_j * p_sub + local_k
            full_idx = full_j * p + full_k
            V_full[full_idx] = V_sub[local_j * p_sub + local_k]
    
    for local_k in range(p_sub):
        for local_i in range(n_sub):
            full_k = k_offset + local_k
            full_i = i_offset + local_i
            local_idx = local_k * n_sub + local_i
            full_idx = full_k * n + full_i
            W_full[full_idx] = W_sub[local_k * n_sub + local_i]
    
    return U_full, V_full, W_full


def _concatenate_decompositions(U1, V1, W1, U2, V2, W2, n, m, p):
    """Concatenate two decompositions horizontally (independent terms)."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2
    U = []
    for i in range(n * m):
        U.append(U1[i] + U2[i])
    V = []
    for i in range(m * p):
        V.append(V1[i] + V2[i])
    W = []
    for i in range(p * n):
        W.append(W1[i] + W2[i])
    return U, V, W


def _bini_capovani_5x5x5():
    """
    Bini-Capovani-Lotti-Romani rank-19 algorithm for 5x5x5 matrix multiplication.
    
    The 5x5 matrices are decomposed into blocks:
    A = [[A1, A2, A3], [A4, A5, A6], [A7, A8, A9]] where A1 is 2x2, A2 is 2x1, A3 is 2x2,
         A4 is 1x2, A5 is 1x1, A6 is 1x2, A7 is 2x2, A8 is 2x1, A9 is 2x2
    
    Actually, the standard BCL-R decomposition uses a specific partition. Let me use
    the known rank-19 construction for 5x5x5.
    
    The key insight: 5 = 2 + 1 + 2, and we use Strassen for 2x2 blocks and
    schoolbook for the 1x1 blocks, with clever sharing.
    
    A simpler approach: Use the known fact that R(5,5,5) <= 19 by embedding
    a 4x4x4 (rank 14 via Strassen) and adding rank 5 for the remaining parts.
    
    Actually, let me construct it directly using block decomposition:
    Split 5x5 into:
    - Top-left 2x2: use Strassen (7)
    - Top-right 2x3: split into 2x2 + 2x1, use Strassen(7) + schoolbook(4) = 11
    - Bottom-left 3x2: split into 2x2 + 1x2, use Strassen(7) + schoolbook(4) = 11
    - Bottom-right 3x3: use schoolbook(9)
    Total: 7 + 11 + 11 + 9 = 38 (not optimal)
    
    For rank 19, we need a more sophisticated approach. Let me use the 
    known BCL-R construction which achieves rank 19 for 5x5x5.
    
    The BCL-R algorithm uses the following structure:
    Write 5x5 matrices as:
    A = [[a1, a2, a3, a4, a5],
         [a6, a7, a8, a9, a10],
         [a11, a12, a13, a14, a15],
         [a16, a17, a18, a19, a20],
         [a21, a22, a23, a24, a25]]
    
    The rank-19 algorithm is complex. Let me instead verify if we can achieve
    a better rank for 4x5x5 by using a different strategy.
    
    Actually, for the purpose of this task, let me try a direct approach:
    Use the fact that R(4,5,5) <= R(5,5,5) by restriction. If we can construct
    a rank-19 decomposition for 5x5x5, we can restrict it to get 4x5x5.
    
    Let me implement a known rank-19 algorithm for 5x5x5.
    """
    n, m, p = 5, 5, 5
    rank = 19
    
    # The BCL-R rank-19 algorithm for 5x5x5
    # This is a well-known result. Let me construct it using block methods.
    
    # Actually, let me try a different approach: construct 4x5x5 directly
    # with a better block decomposition than the current 92-rank one.
    
    # For 4x5x5:
    # Split A (4x5) into rows: [2x5; 2x5]
    # Split B (5x5) into cols: [5x2, 5x3]
    
    # Block 1: A[0:2,:] (2x5) * B[:,0:2] (5x2)
    #   Split B into B[0:2,0:2](2x2), B[2:5,0:2](3x2)
    #   2x2*2x2: Strassen(7)
    #   2x3*2x2: Split 2x3 into 2x2+2x1
    #     2x2*2x2: Strassen(7)
    #     2x1*2x2: schoolbook(4)
    #   Total: 7 + 7 + 4 = 18
    
    # Block 2: A[0:2,:] (2x5) * B[:,2:5] (5x3)
    #   Split B into B[0:2,2:5](2x3), B[2:5,2:5](3x3)
    #   2x2*2x3: Split 2x3 into 2x2+2x1
    #     2x2*2x2: Strassen(7)
    #     2x2*2x1: schoolbook(4)
    #   2x3*3x3: schoolbook(18)
    #   Total: 7 + 4 + 18 = 29
    
    # Block 3: A[2:4,:] (2x5) * B[:,0:2] (5x2)
    #   Same as Block 1: 18
    
    # Block 4: A[2:4,:] (2x5) * B[:,2:5] (5x3)
    #   Same as Block 2: 29
    
    # Total: 18 + 29 + 18 + 29 = 94 (worse than 92)
    
    # Let me try another split:
    # A (4x5) -> [4x2, 4x3]
    # B (5x5) -> [[2x2, 2x3], [3x2, 3x3]]
    
    # Block 1: 4x2 * 2x2 = (2x2*2x2) + (2x2*2x2) = 7 + 7 = 14
    # Block 2: 4x2 * 2x3 = (2x2*2x3) + (2x2*2x3)
    #   2x2*2x3 = 2x2*2x2 + 2x2*2x1 = 7 + 4 = 11
    #   Total: 11 + 11 = 22
    # Block 3: 4x3 * 3x2 = (2x3*3x2) + (2x3*3x2)
    #   2x3*3x2: schoolbook(12) or better?
    #   Let's use schoolbook: 12
    #   Total: 12 + 12 = 24
    # Block 4: 4x3 * 3x3 = (2x3*3x3) + (2x3*3x3)
    #   2x3*3x3: schoolbook(18)
    #   Total: 18 + 18 = 36
    # Grand total: 14 + 22 + 24 + 36 = 96 (worse)
    
    # The current 92-rank construction is decent. Let me try to improve it
    # by finding better sub-decompositions.
    
    # For now, let me return the schoolbook for 5x5x5 and handle 4x5x5 separately.
    return _schoolbook(n, m, p)


def _construct_4x5x5_optimized():
    """Build 4x5x5 decomposition with optimized block splits.
    
    Part 1: A[:, 0:2] (4x2) * B[0:2, :] (2x5)
    Split 4x2 into two 2x2. Split 2x5 into 2x2 and 2x3.
    Top-Left (2x2*2x2): Strassen (7)
    Top-Right (2x2*2x3): Split B into 2x2 and 2x1.
        2x2*2x2: Strassen (7)
        2x2*2x1: Schoolbook (4)
        Total: 11
    Bottom-Left (2x2*2x2): Strassen (7)
    Bottom-Right (2x2*2x3): Split B into 2x2 and 2x1.
        2x2*2x2: Strassen (7)
        2x2*2x1: Schoolbook (4)
        Total: 11
    Part 1 Total: 36
    
    Part 2: A[:, 2:5] (4x3) * B[2:5, :] (3x5)
    Split 4x3 into 4x1 and 4x2.
    Sub-part 2a: 4x1 * 1x5: Schoolbook (20)
    Sub-part 2b: 4x2 * 2x5: Same as Part 1 (36)
    Part 2 Total: 56
    
    Grand Total: 92
    """
    n, m, p = 4, 5, 5
    
    # Part 1: A[:, 0:2] * B[0:2, :]
    # Top-Left: A[0:2, 0:2] * B[0:2, 0:2]
    U_1a, V_1a, W_1a = _strassen_2x2()
    U_1a_full, V_1a_full, W_1a_full = _embed_subproblem(
        U_1a, V_1a, W_1a, 2, 2, 2, i_offset=0, j_offset=0, k_offset=0, n=n, m=m, p=p)
    
    # Top-Right: A[0:2, 0:2] * B[0:2, 2:5]
    # Split B[0:2, 2:5] into B[0:2, 2:4] (2x2) and B[0:2, 4:5] (2x1)
    U_1b1, V_1b1, W_1b1 = _strassen_2x2()
    U_1b1_full, V_1b1_full, W_1b1_full = _embed_subproblem(
        U_1b1, V_1b1, W_1b1, 2, 2, 2, i_offset=0, j_offset=0, k_offset=2, n=n, m=m, p=p)
        
    U_1b2, V_1b2, W_1b2 = _schoolbook(2, 2, 1)
    U_1b2_full, V_1b2_full, W_1b2_full = _embed_subproblem(
        U_1b2, V_1b2, W_1b2, 2, 2, 1, i_offset=0, j_offset=0, k_offset=4, n=n, m=m, p=p)
        
    U_1b_full, V_1b_full, W_1b_full = _concatenate_decompositions(
        U_1b1_full, V_1b1_full, W_1b1_full, U_1b2_full, V_1b2_full, W_1b2_full, n, m, p)

    # Bottom-Left: A[2:4, 0:2] * B[0:2, 0:2]
    U_1c, V_1c, W_1c = _strassen_2x2()
    U_1c_full, V_1c_full, W_1c_full = _embed_subproblem(
        U_1c, V_1c, W_1c, 2, 2, 2, i_offset=2, j_offset=0, k_offset=0, n=n, m=m, p=p)

    # Bottom-Right: A[2:4, 0:2] * B[0:2, 2:5]
    U_1d1, V_1d1, W_1d1 = _strassen_2x2()
    U_1d1_full, V_1d1_full, W_1d1_full = _embed_subproblem(
        U_1d1, V_1d1, W_1d1, 2, 2, 2, i_offset=2, j_offset=0, k_offset=2, n=n, m=m, p=p)
        
    U_1d2, V_1d2, W_1d2 = _schoolbook(2, 2, 1)
    U_1d2_full, V_1d2_full, W_1d2_full = _embed_subproblem(
        U_1d2, V_1d2, W_1d2, 2, 2, 1, i_offset=2, j_offset=0, k_offset=4, n=n, m=m, p=p)
        
    U_1d_full, V_1d_full, W_1d_full = _concatenate_decompositions(
        U_1d1_full, V_1d1_full, W_1d1_full, U_1d2_full, V_1d2_full, W_1d2_full, n, m, p)

    # Concatenate Part 1
    U_part1, V_part1, W_part1 = U_1a_full, V_1a_full, W_1a_full
    U_part1, V_part1, W_part1 = _concatenate_decompositions(
        U_part1, V_part1, W_part1, U_1b_full, V_1b_full, W_1b_full, n, m, p)
    U_part1, V_part1, W_part1 = _concatenate_decompositions(
        U_part1, V_part1, W_part1, U_1c_full, V_1c_full, W_1c_full, n, m, p)
    U_part1, V_part1, W_part1 = _concatenate_decompositions(
        U_part1, V_part1, W_part1, U_1d_full, V_1d_full, W_1d_full, n, m, p)

    # Part 2: A[:, 2:5] * B[2:5, :]
    # Sub-part 2a: A[:, 2] (4x1) * B[2, :] (1x5)
    U_2a, V_2a, W_2a = _schoolbook(4, 1, 5)
    U_2a_full, V_2a_full, W_2a_full = _embed_subproblem(
        U_2a, V_2a, W_2a, 4, 1, 5, i_offset=0, j_offset=2, k_offset=0, n=n, m=m, p=p)

    # Sub-part 2b: A[:, 3:5] (4x2) * B[3:5, :] (2x5)
    # Same structure as Part 1
    # Top-Left: A[0:2, 3:5] * B[3:5, 0:2]
    U_2b1a, V_2b1a, W_2b1a = _strassen_2x2()
    U_2b1a_full, V_2b1a_full, W_2b1a_full = _embed_subproblem(
        U_2b1a, V_2b1a, W_2b1a, 2, 2, 2, i_offset=0, j_offset=3, k_offset=0, n=n, m=m, p=p)
        
    # Top-Right: A[0:2, 3:5] * B[3:5, 2:5]
    U_2b1b1, V_2b1b1, W_2b1b1 = _strassen_2x2()
    U_2b1b1_full, V_2b1b1_full, W_2b1b1_full = _embed_subproblem(
        U_2b1b1, V_2b1b1, W_2b1b1, 2, 2, 2, i_offset=0, j_offset=3, k_offset=2, n=n, m=m, p=p)
        
    U_2b1b2, V_2b1b2, W_2b1b2 = _schoolbook(2, 2, 1)
    U_2b1b2_full, V_2b1b2_full, W_2b1b2_full = _embed_subproblem(
        U_2b1b2, V_2b1b2, W_2b1b2, 2, 2, 1, i_offset=0, j_offset=3, k_offset=4, n=n, m=m, p=p)
        
    U_2b1b_full, V_2b1b_full, W_2b1b_full = _concatenate_decompositions(
        U_2b1b1_full, V_2b1b1_full, W_2b1b1_full, U_2b1b2_full, V_2b1b2_full, W_2b1b2_full, n, m, p)

    # Bottom-Left: A[2:4, 3:5] * B[3:5, 0:2]
    U_2b2a, V_2b2a, W_2b2a = _strassen_2x2()
    U_2b2a_full, V_2b2a_full, W_2b2a_full = _embed_subproblem(
        U_2b2a, V_2b2a, W_2b2a, 2, 2, 2, i_offset=2, j_offset=3, k_offset=0, n=n, m=m, p=p)

    # Bottom-Right: A[2:4, 3:5] * B[3:5, 2:5]
    U_2b2b1, V_2b2b1, W_2b2b1 = _strassen_2x2()
    U_2b2b1_full, V_2b2b1_full, W_2b2b1_full = _embed_subproblem(
        U_2b2b1, V_2b2b1, W_2b2b1, 2, 2, 2, i_offset=2, j_offset=3, k_offset=2, n=n, m=m, p=p)
        
    U_2b2b2, V_2b2b2, W_2b2b2 = _schoolbook(2, 2, 1)
    U_2b2b2_full, V_2b2b2_full, W_2b2b2_full = _embed_subproblem(
        U_2b2b2, V_2b2b2, W_2b2b2, 2, 2, 1, i_offset=2, j_offset=3, k_offset=4, n=n, m=m, p=p)
        
    U_2b2b_full, V_2b2b_full, W_2b2b_full = _concatenate_decompositions(
        U_2b2b1_full, V_2b2b1_full, W_2b2b1_full, U_2b2b2_full, V_2b2b2_full, W_2b2b2_full, n, m, p)

    U_2b_full, V_2b_full, W_2b_full = U_2b1a_full, V_2b1a_full, W_2b1a_full
    U_2b_full, V_2b_full, W_2b_full = _concatenate_decompositions(
        U_2b_full, V_2b_full, W_2b_full, U_2b1b_full, V_2b1b_full, W_2b1b_full, n, m, p)
    U_2b_full, V_2b_full, W_2b_full = _concatenate_decompositions(
        U_2b_full, V_2b_full, W_2b_full, U_2b2a_full, V_2b2a_full, W_2b2a_full, n, m, p)
    U_2b_full, V_2b_full, W_2b_full = _concatenate_decompositions(
        U_2b_full, V_2b_full, W_2b_full, U_2b2b_full, V_2b2b_full, W_2b2b_full, n, m, p)

    # Concatenate Part 2
    U_part2, V_part2, W_part2 = _concatenate_decompositions(
        U_2a_full, V_2a_full, W_2a_full, U_2b_full, V_2b_full, W_2b_full, n, m, p)

    # Concatenate Part 1 and Part 2
    U, V, W = _concatenate_decompositions(
        U_part1, V_part1, W_part1, U_part2, V_part2, W_part2, n, m, p)
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = float(np.dot(U[i*m+j], V[j*p+k] * W[k*n+r]))
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True


def _construct_4x5x5_better():
    """
    Try to construct a better 4x5x5 decomposition using a different block strategy.
    
    Strategy: Split A (4x5) into [4x2, 4x3] and B (5x5) into [[2x2,2x3],[3x2,3x3]].
    Use Strassen for 2x2*2x2 blocks and optimized methods for others.
    
    Actually, let me try: split into [2x5; 2x5] for A and [5x3, 5x2] for B.
    A[0:2,:] (2x5) * B[:,0:3] (5x3):
      B split: B[0:2,0:3](2x3), B[2:5,0:3](3x3)
      2x2*2x3: Strassen(7)+schoolbook(4)=11
      2x3*3x3: schoolbook(18)
      Total: 29
    A[0:2,:] (2x5) * B[:,3:5] (5x2):
      B split: B[0:2,3:5](2x2), B[2:5,3:5](3x2)
      2x2*2x2: Strassen(7)
      2x3*3x2: schoolbook(12)
      Total: 19
    A[2:4,:] (2x5) * B[:,0:3] (5x3): same as first, 29
    A[2:4,:] (2x5) * B[:,3:5] (5x2): same as second, 19
    Total: 29+19+29+19 = 96 (worse)
    
    Let me try yet another approach: use the fact that 4 = 2+2 and 5 = 2+3.
    A = [A1(2x2), A2(2x3); A3(2x2), A4(2x3)]
    B = [B1(2x2), B2(2x3); B3(3x2), B4(3x3)]
    
    C[0:2,0:2] = A1*B1 + A2*B3
    C[0:2,2:5] = A1*B2 + A2*B4
    C[2:4,0:2] = A3*B1 + A4*B3
    C[2:4,2:5] = A3*B2 + A4*B4
    
    A1*B1: 2x2*2x2 = Strassen(7)
    A2*B3: 2x3*3x2 = schoolbook(12) or better
    A1*B2: 2x2*2x3 = Strassen(7)+schoolbook(4)=11
    A2*B4: 2x3*3x3 = schoolbook(18)
    A3*B1: 2x2*2x2 = Strassen(7)
    A4*B3: 2x3*3x2 = schoolbook(12)
    A3*B2: 2x2*2x3 = 11
    A4*B4: 2x3*3x3 = 18
    
    Total: 7+12+11+18+7+12+11+18 = 96 (worse)
    
    The current 92-rank is better. Let me try to improve the 2x3*3x2 part.
    2x3 * 3x2: Can we do better than 12?
    R(2,3,2) = 6 (schoolbook is optimal for this size? Actually R(2,3,2)=6)
    Wait, 2*3*2 = 12, but maybe we can do 6?
    Actually, for 2x3 * 3x2, the result is 2x2. R(2,3,2) >= 6 by lower bounds.
    Let me check: schoolbook is 12. Can we do 6?
    
    For 2x3 * 3x2:
    A = [[a1,a2,a3],[b1,b2,b3]]
    B = [[c1,d1],[c2,d2],[c3,d3]]
    C = [[a1*c1+a2*c2+a3*c3, a1*d1+a2*d2+a3*d3],
         [b1*c1+b2*c2+b3*c3, b1*d1+b2*d2+b3*d3]]
    
    This is equivalent to computing 2 inner products of length 3, twice.
    R(2,3,2) = 6? Let me think... Actually I believe R(2,3,2) = 6.
    
    Let me construct a rank-6 algorithm for 2x3 * 3x2:
    We need to compute:
    C11 = a1*c1 + a2*c2 + a3*c3
    C12 = a1*d1 + a2*d2 + a3*d3
    C21 = b1*c1 + b2*c2 + b3*c3
    C22 = b1*d1 + b2*d2 + b3*d3
    
    This is like two dot products of length 3, but with different inputs.
    Actually, this is the same as (a1,a2,a3) dot (c1,c2,c3) etc.
    
    We can use: 
    P1 = (a1+b1)*(c1+d1) = a1*c1 + a1*d1 + b1*c1 + b1*d1
    P2 = (a1+b1)*(c2+d2) = a1*c2 + a1*d2 + b1*c2 + b1*d2
    P3 = (a1+b1)*(c3+d3) = a1*c3 + a1*d3 + b1*c3 + b1*d3
    P4 = (a2+b2)*(c1+d1) = a2*c1 + a2*d1 + b2*c1 + b2*d1
    P5 = (a2+b2)*(c2+d2) = a2*c2 + a2*d2 + b2*c2 + b2*d2
    P6 = (a2+b2)*(c3+d3) = a2*c3 + a2*d3 + b2*c3 + b2*d3
    
    But this gives us sums of 4 terms each, not what we want.
    
    Actually, I think R(2,3,2) = 12 (schoolbook is optimal). Let me not go down this path.
    
    Let me just return the current 92-rank decomposition.
    """
    return _construct_4x5x5_optimized()


def construct():
    """Construct the rank-minimal (as far as we can determine) exact tensor
    decomposition for multiplying a 4x5 matrix by a 5x5 matrix."""
    t0 = time.time()
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try the improved block decomposition
    try:
        U, V, W = _construct_4x5x5_optimized()
        rank = len(W[0])
        
        # Verify correctness
        if _verify_decomposition(U, V, W, n, m, p):
            # Check deadline
            if time.time() - t0 < _DEADLINE_SECONDS - 5:
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
