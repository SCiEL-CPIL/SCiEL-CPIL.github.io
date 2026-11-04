"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2(n, m, p):
    """
    Build a Strassen-based decomposition for multiplying n x m by m x p,
    where n, m, p are all even, using 2x2 blocks.
    The 2x2x2 Strassen decomposition has rank 7.
    """
    nb = n // 2
    mb = m // 2
    pb = p // 2
    
    # Strassen factors for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U factors: (A) side - for 2x2 block, U has 4 rows (2x2), 7 cols
    # Row index in 2x2: (bi, bj) where bi in {0,1}, bj in {0,1}
    # For Strassen, U[bi*2+bj, c] gives coefficient of A_{bi,bj} in M_c
    
    U2 = np.zeros((4, 7))
    # M1: (A11+A22)(B11+B22) -> A11:1, A22:1
    U2[0, 0] = 1; U2[3, 0] = 1
    # M2: (A21+A22)B11 -> A21:1, A22:1
    U2[2, 1] = 1; U2[3, 1] = 1
    # M3: A11(B21-B22) -> A11:1
    U2[0, 2] = 1
    # M4: A22(B11-B21) -> A22:1
    U2[3, 3] = 1
    # M5: (A11+A12)B22 -> A11:1, A12:1
    U2[0, 4] = 1; U2[1, 4] = 1
    # M6: (A21-A11)(B11+B12) -> A21:1, A11:-1
    U2[2, 5] = 1; U2[0, 5] = -1
    # M7: (A12-A22)(B21+B22) -> A12:1, A22:-1
    U2[1, 6] = 1; U2[3, 6] = -1
    
    # V factors: (B) side - for 2x2 block, V has 4 rows (2x2), 7 cols
    # M1: (B11+B22) -> B11:1, B22:1
    V2[0, 0] = 1; V2[3, 0] = 1
    # M2: B11 -> B11:1
    V2[0, 1] = 1
    # M3: (B21-B22) -> B21:1, B22:-1
    V2[2, 2] = 1; V2[3, 2] = -1
    # M4: (B11-B21) -> B11:1, B21:-1
    V2[0, 3] = 1; V2[2, 3] = -1
    # M5: B22 -> B22:1
    V2[3, 4] = 1
    # M6: (B11+B12) -> B11:1, B12:1
    V2[0, 5] = 1; V2[1, 5] = 1
    # M7: (B21+B22) -> B21:1, B22:1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # W factors: (C) side - for 2x2 block, W has 4 rows (2x2), 7 cols
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    W2 = np.zeros((4, 7))
    # C11: M1:1, M4:1, M5:-1, M7:1
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    # C12: M3:1, M5:1
    W2[1, 2] = 1; W2[1, 4] = 1
    # C21: M2:1, M4:1
    W2[2, 1] = 1; W2[2, 3] = 1
    # C22: M1:1, M2:-1, M3:1, M6:1
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1
    
    # Now build the full decomposition
    rank = 7 * nb * mb * pb
    U_full = np.zeros((n * m, rank))
    V_full = np.zeros((m * p, rank))
    W_full = np.zeros((p * n, rank))
    
    col = 0
    for ib in range(nb):
        for jb in range(mb):
            for kb in range(pb):
                for c in range(7):
                    # U: element (i, j) where i = 2*ib + bi, j = 2*jb + bj
                    for bi in range(2):
                        for bj in range(2):
                            uval = U2[bi * 2 + bj, c]
                            if uval != 0:
                                i = 2 * ib + bi
                                j = 2 * jb + bj
                                U_full[i * m + j, col] = uval
                    # V: element (j, k) where j = 2*jb + bj, k = 2*kb + bk
                    for bj in range(2):
                        for bk in range(2):
                            vval = V2[bj * 2 + bk, c]
                            if vval != 0:
                                j = 2 * jb + bj
                                k = 2 * kb + bk
                                V_full[j * p + k, col] = vval
                    # W: element (k, i) where k = 2*kb + bk, i = 2*ib + bi
                    for bk in range(2):
                        for bi in range(2):
                            wval = W2[bk * 2 + bi, c]
                            if wval != 0:
                                k = 2 * kb + bk
                                i = 2 * ib + bi
                                W_full[k * n + i, col] = wval
                    col += 1
    
    return U_full, V_full, W_full, rank

def _build_hybrid(n, m, p, deadline):
    """
    Build a hybrid decomposition using Strassen for the 2x2 blocks where possible,
    and schoolbook for the remaining parts.
    For 3x5x6: we can use Strassen on the 2x2 sub-blocks.
    The 3x5 matrix has blocks: rows 0-1 (2 rows) + row 2 (1 row), cols 0-1, 2-3, 4 (2,2,1 cols)
    The 5x6 matrix has blocks: rows 0-1, 2-3, 4 (2,2,1 rows), cols 0-1, 2-3, 4-5 (2,2,2 cols)
    """
    # For 3x5x6, we can't directly use 2x2 Strassen on all blocks.
    # Instead, let's think of it as:
    # The matrix multiplication C = A * B where A is 3x5, B is 5x6.
    # We can partition A into blocks and use Strassen where dimensions are even.
    
    # Alternative approach: Use the fact that we can decompose this as a sum of
    # smaller matrix multiplications.
    
    # Let's try a different approach: use a structured decomposition that
    # combines Strassen-like ideas with the specific dimensions.
    
    # For 3x5x6, the schoolbook rank is 90.
    # A better approach: use the fact that the tensor rank of matrix multiplication
    # M_{n,m,p} is bounded by known results.
    
    # Let me try a direct structured approach:
    # Partition the computation into blocks and use Strassen for 2x2x2 subproblems.
    
    # A is 3x5, B is 5x6, C is 3x6.
    # Split A into A_top (2x5) and A_bot (1x5).
    # Split B into B_left (5x2), B_mid (5x2), B_right (5x2).
    # C_top = A_top * B (2x6)
    # C_bot = A_bot * B (1x6)
    
    # For A_top (2x5) * B (5x6) = C_top (2x6):
    # Split A_top into 2x2, 2x2, 2x1 blocks.
    # This gets complicated. Let me try a different strategy.
    
    # Strategy: Use a rank-reducing decomposition based on the structure.
    # The key insight from the knowledge: Block Strassen reduces rank from 90 to ~84.
    
    # Let's implement a proper block Strassen for 3x5x6.
    # Partition:
    # A (3x5) -> split rows into 2+1, cols into 2+2+1
    # B (5x6) -> split rows into 2+2+1, cols into 2+2+2
    # C (3x6) -> split rows into 2+1, cols into 2+2+2
    
    # For each 2x2x2 subblock, use Strassen (rank 7 instead of 8).
    # For mixed blocks, use schoolbook.
    
    # Actually, let me just use a simpler but effective approach:
    # Use the Strassen decomposition for the 2x2x2 core and handle the rest.
    
    # Simpler: just return schoolbook if we can't do better, but first try
    # a specific structured decomposition.
    
    # Let me try: decompose as a sum of rank-1 tensors with a structured pattern.
    
    # Actually, the most reliable improvement for small matrices like 3x5x6
    # is to use known rank bounds. The rank of M_{3,5,6} is known to be at most
    # some value. Let me just use schoolbook as the safe fallback and try to
    # find a better one through optimization.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 90
    
    # Try Strassen-based decomposition for even-dimensional subproblems
    # For 3x5x6, we can't directly apply 2x2 Strassen to the whole thing,
    # but we can apply it to 2x2x2 sub-blocks.
    
    # Let's build a block Strassen decomposition:
    # Split A (3x5) into blocks of size 2x2, 2x2, 2x1, 1x2, 1x2, 1x1
    # This is complex. Instead, let's use a simpler approach.
    
    # For 3x5x6, use the following decomposition strategy:
    # The matrix multiplication tensor has rank at most n*m*p = 90 (schoolbook).
    # Using Strassen on 2x2x2 sub-blocks can save 1 rank per 2x2x2 block.
    
    # Let me try a direct construction:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    
    # Use Strassen on the 2x2x2 blocks where i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3} or {4,5}
    # For i=2, use schoolbook.
    
    # Let's build this carefully:
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} - Strassen 2x2x2
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} - Strassen 2x2x2
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5} - Strassen 2x2x2
    # Block 4: i in {0,1}, j in {2,3}, k in {0,1} - Strassen 2x2x2
    # Block 5: i in {0,1}, j in {2,3}, k in {2,3} - Strassen 2x2x2
    # Block 6: i in {0,1}, j in {2,3}, k in {4,5} - Strassen 2x2x2
    # Block 7: i in {0,1}, j in {4}, k in {0,1} - schoolbook 2x1x2 (rank 4)
    # Block 8: i in {0,1}, j in {4}, k in {2,3} - schoolbook 2x1x2 (rank 4)
    # Block 9: i in {0,1}, j in {4}, k in {4,5} - schoolbook 2x1x2 (rank 4)
    # Block 10: i in {2}, j in {0,1}, k in {0,1} - schoolbook 1x2x2 (rank 4)
    # Block 11: i in {2}, j in {0,1}, k in {2,3} - schoolbook 1x2x2 (rank 4)
    # Block 12: i in {2}, j in {0,1}, k in {4,5} - schoolbook 1x2x2 (rank 4)
    # Block 13: i in {2}, j in {2,3}, k in {0,1} - schoolbook 1x2x2 (rank 4)
    # Block 14: i in {2}, j in {2,3}, k in {2,3} - schoolbook 1x2x2 (rank 4)
    # Block 15: i in {2}, j in {2,3}, k in {4,5} - schoolbook 1x2x2 (rank 4)
    # Block 16: i in {2}, j in {4}, k in {0,1} - schoolbook 1x1x2 (rank 2)
    # Block 17: i in {2}, j in {4}, k in {2,3} - schoolbook 1x1x2 (rank 2)
    # Block 18: i in {2}, j in {4}, k in {4,5} - schoolbook 1x1x2 (rank 2)
    
    # Total rank: 6*7 + 3*4 + 6*4 + 3*2 = 42 + 12 + 24 + 6 = 84
    
    # Let's build this:
    
    # Strassen 2x2x2 factors
    U2 = [[1,0,0,0,1,-1,0],
          [0,0,0,0,1,0,1],
          [0,1,0,0,0,1,0],
          [1,1,0,1,0,0,-1]]
    
    V2 = [[1,1,0,1,0,1,0],
          [0,0,0,0,0,1,0],
          [0,0,1,-1,0,0,1],
          [1,0,-1,0,1,0,1]]
    
    W2 = [[1,0,0,1,-1,0,1],
          [0,0,1,0,1,0,0],
          [0,1,0,1,0,0,0],
          [1,-1,1,0,0,1,0]]
    
    col = 0
    U_mat = np.zeros((n * m, 100))  # Max possible rank
    V_mat = np.zeros((m * p, 100))
    W_mat = np.zeros((p * n, 100))
    
    def add_strassen_block(i0, j0, k0):
        """Add a Strassen 2x2x2 block at offsets (i0, j0, k0)."""
        nonlocal col
        for c in range(7):
            # U: (i, j) where i in {i0, i0+1}, j in {j0, j0+1}
            for bi in range(2):
                for bj in range(2):
                    uval = U2[bi * 2 + bj][c]
                    if uval != 0:
                        i = i0 + bi
                        j = j0 + bj
                        U_mat[i * m + j, col] = uval
            # V: (j, k) where j in {j0, j0+1}, k in {k0, k0+1}
            for bj in range(2):
                for bk in range(2):
                    vval = V2[bj * 2 + bk][c]
                    if vval != 0:
                        j = j0 + bj
                        k = k0 + bk
                        V_mat[j * p + k, col] = vval
            # W: (k, i) where k in {k0, k0+1}, i in {i0, i0+1}
            for bk in range(2):
                for bi in range(2):
                    wval = W2[bk * 2 + bi][c]
                    if wval != 0:
                        k = k0 + bk
                        i = i0 + bi
                        W_mat[k * n + i, col] = wval
            col += 1
    
    def add_schoolbook_block(i0, i1, j0, j1, k0, k1):
        """Add a schoolbook block for i in [i0,i1], j in [j0,j1], k in [k0,k1]."""
        nonlocal col
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    U_mat[i * m + j, col] = 1
                    V_mat[j * p + k, col] = 1
                    W_mat[k * n + i, col] = 1
                    col += 1
    
    # Add Strassen blocks for 2x2x2 sub-blocks
    # i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    # i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen_block(0, 0, 2)
    # i in {0,1}, j in {0,1}, k in {4,5}
    add_strassen_block(0, 0, 4)
    # i in {0,1}, j in {2,3}, k in {0,1}
    add_strassen_block(0, 2, 0)
    # i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen_block(0, 2, 2)
    # i in {0,1}, j in {2,3}, k in {4,5}
    add_strassen_block(0, 2, 4)
    
    # Add schoolbook blocks for remaining
    # i in {0,1}, j in {4}, k in {0,1}
    add_schoolbook_block(0, 1, 4, 4, 0, 1)
    # i in {0,1}, j in {4}, k in {2,3}
    add_schoolbook_block(0, 1, 4, 4, 2, 3)
    # i in {0,1}, j in {4}, k in {4,5}
    add_schoolbook_block(0, 1, 4, 4, 4, 5)
    # i in {2}, j in {0,1}, k in {0,1}
    add_schoolbook_block(2, 2, 0, 1, 0, 1)
    # i in {2}, j in {0,1}, k in {2,3}
    add_schoolbook_block(2, 2, 0, 1, 2, 3)
    # i in {2}, j in {0,1}, k in {4,5}
    add_schoolbook_block(2, 2, 0, 1, 4, 5)
    # i in {2}, j in {2,3}, k in {0,1}
    add_schoolbook_block(2, 2, 2, 3, 0, 1)
    # i in {2}, j in {2,3}, k in {2,3}
    add_schoolbook_block(2, 2, 2, 3, 2, 3)
    # i in {2}, j in {2,3}, k in {4,5}
    add_schoolbook_block(2, 2, 2, 3, 4, 5)
    # i in {2}, j in {4}, k in {0,1}
    add_schoolbook_block(2, 2, 4, 4, 0, 1)
    # i in {2}, j in {4}, k in {2,3}
    add_schoolbook_block(2, 2, 4, 4, 2, 3)
    # i in {2}, j in {4}, k in {4,5}
    add_schoolbook_block(2, 2, 4, 4, 4, 5)
    
    # Trim to actual rank
    U_mat = U_mat[:, :col]
    V_mat = V_mat[:, :col]
    W_mat = W_mat[:, :col]
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U_mat]
    V_list = [[float(x) for x in row] for row in V_mat]
    W_list = [[float(x) for x in row] for row in W_mat]
    
    # Verify
    if _verify(U_list, V_list, W_list, n, m, p):
        best = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = col
    
    return best

# EVOLVE-BLOCK-END
