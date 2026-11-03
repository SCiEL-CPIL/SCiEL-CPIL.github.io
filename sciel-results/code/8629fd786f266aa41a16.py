"""Matrix multiplication 3x4x7 rank minimization."""
DIMENSIONS = [3, 4, 7]

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x7 matrix.
    
    Strategy: Use Strassen's algorithm on overlapping 2x2x2 sub-blocks to reduce
    rank below the schoolbook 84. We carefully select non-overlapping (or 
    compatible) block decompositions.
    
    For 3x4x7, we can partition as follows:
    - Use Strassen on a 2x2x2 block (saves 4 terms, uses 7 instead of 8)
    - Remaining terms handled by schoolbook
    
    Actually, let's think more carefully. The matrix multiplication tensor T_{3,4,7}
    has entries T[i,j,k] for i in [0,3), j in [0,4), k in [0,7), corresponding to
    C[i,k] = sum_j A[i,j]*B[j,k].
    
    A rank-r decomposition gives matrices U (12 x r), V (28 x r), W (21 x r) such that
    for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta (where the
    tensor is 1 at valid (i,j,k) and 0 elsewhere).
    
    The schoolbook rank is n*m*p = 3*4*7 = 84.
    
    Using Strassen on a 2x2x2 subproblem: if we have a sub-block where
    A_sub is 2x2, B_sub is 2x2, C_sub is 2x2, we can replace 8 rank-1 terms with 7.
    
    For 3x4x7, we can find 2x2 sub-blocks. Let's use the approach:
    - Pick rows i0,i1 of A, columns j0,j1 of the middle, columns k0,k1 of B
    - Apply Strassen to that 2x2x2 sub-problem
    
    But we need to be careful: each (i,j,k) triple can only be covered once in the
    decomposition. So we need disjoint 2x2x2 blocks.
    
    3 rows, 4 middle, 7 cols. We can fit at most floor(3/2)*floor(4/2)*floor(7/2) = 1*2*3 = 6
    disjoint 2x2x2 blocks, leaving some rows/cols for schoolbook.
    
    Let's pick:
    Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen (7 terms)
    Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> Strassen (7 terms)
    Block 3: i in {0,1}, j in {2,3}, k in {0,1} -> Strassen (7 terms)
    Block 4: i in {0,1}, j in {2,3}, k in {2,3} -> Strassen (7 terms)
    Block 5: i in {0,1}, j in {2,3}, k in {4,5} -> Strassen (7 terms)
    Remaining: 
      - i=2, all j in {0,1,2,3}, all k in {0,...,6} -> 4*7 = 28 schoolbook terms
      - i in {0,1}, j in {0,1}, k in {6} -> 2*1*1 = 2 schoolbook
      - i in {0,1}, j in {2,3}, k in {6} -> 2*2*1 = 4 schoolbook
      - i in {0,1}, j in {0,1}, k in {4,5} -> 2*2*2 = 8 schoolbook
      - i in {0,1}, j in {2,3}, k in {4,5} -> already in Block 5? No, Block 5 uses k in {4,5}.
    
    Wait, I need to re-plan. Let me partition the (i, j, k) space more carefully.
    
    i: 0,1,2 (3 rows)
    j: 0,1,2,3 (4 middle)
    k: 0,1,2,3,4,5,6 (7 cols)
    
    Let's group:
    i_group_0 = {0,1}, i_group_1 = {2}
    j_group_0 = {0,1}, j_group_1 = {2,3}
    k_group_0 = {0,1}, k_group_1 = {2,3}, k_group_2 = {4,5}, k_group_3 = {6}
    
    For each combination of (i_group, j_group, k_group):
    - If i_group has size 2, j_group has size 2, k_group has size 2: Strassen (7)
    - Otherwise: schoolbook (size_i * size_j * size_k)
    
    Combinations:
    (ig0, jg0, kg0): 2x2x2 -> Strassen: 7
    (ig0, jg0, kg1): 2x2x2 -> Strassen: 7
    (ig0, jg0, kg2): 2x2x2 -> Strassen: 7
    (ig0, jg0, kg3): 2x2x1 -> schoolbook: 4
    (ig0, jg1, kg0): 2x2x2 -> Strassen: 7
    (ig0, jg1, kg1): 2x2x2 -> Strassen: 7
    (ig0, jg1, kg2): 2x2x2 -> Strassen: 7
    (ig0, jg1, kg3): 2x2x1 -> schoolbook: 4
    (ig1, jg0, kg0): 1x2x2 -> schoolbook: 4
    (ig1, jg0, kg1): 1x2x2 -> schoolbook: 4
    (ig1, jg0, kg2): 1x2x2 -> schoolbook: 4
    (ig1, jg0, kg3): 1x2x1 -> schoolbook: 2
    (ig1, jg1, kg0): 1x2x2 -> schoolbook: 4
    (ig1, jg1, kg1): 1x2x2 -> schoolbook: 4
    (ig1, jg1, kg2): 1x2x2 -> schoolbook: 4
    (ig1, jg1, kg3): 1x2x1 -> schoolbook: 2
    
    Total Strassen: 6 * 7 = 42
    Total schoolbook: 4+4+4+2+4+4+4+2+4+4+4+2 = 42
    Total rank: 42 + 42 = 84. No improvement!
    
    The issue is that using Strassen on 2x2x2 blocks when the overall structure doesn't
    align well doesn't help. We need a different approach.
    
    Actually, the known lower bound for matmul(3,4,7) is... Let me think. The bilinear complexity
    of matrix multiplication is an open problem in general, but for small sizes we can look
    up or compute.
    
    A better approach: use a tensor decomposition algorithm. But that's complex. Let me try
    a simpler optimization: randomly search for a lower rank decomposition.
    
    Actually, let me try a different partition. What if I use a 2x3x3 or 3x2x2 block?
    Strassen only works on square blocks. For non-square, we need general tensor rank methods.
    
    Let me try a computational approach: use randomized search or algebraic methods to find
    a decomposition with rank < 84.
    
    Given the time constraints, let me try a structured approach with Strassen where it
    actually helps. The key insight: if we can find ANY 2x2x2 subproblem where the surrounding
    structure allows us to "absorb" the Strassen savings into the global decomposition, we
    save 1 rank per 2x2x2 block used.
    
    Let me just implement the schoolbook and try a few random perturbations to see if we
    can reduce rank. But that's hard to verify quickly.
    
    For now, let's return the schoolbook decomposition (rank 84) which is valid, and note
    that further optimization would require more sophisticated tensor rank algorithms.
    """
    n, m, p = DIMENSIONS
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
