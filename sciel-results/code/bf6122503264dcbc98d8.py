import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    
    We use a hybrid approach with Strassen 2x2x2 blocks where applicable:
    - Partition n=3 into n1=2, n2=1
    - Partition m=5 into m1=2, m2=3  
    - p=7 stays whole
    
    For the 2x2x2 sub-blocks (where both n and m have size 2), use Strassen (rank 7)
    For remaining blocks, use schoolbook.
    
    Block structure:
    A is 3x5: rows partitioned [2,1], cols partitioned [2,3]
    B is 5x7: rows partitioned [2,3], cols whole
    C is 3x7: rows partitioned [2,1], cols whole
    
    C = A*B where:
    C[0:2, :] = A[0:2, 0:2]*B[0:2, :] + A[0:2, 2:5]*B[2:5, :]
    C[2, :]    = A[2, 0:2]*B[0:2, :] + A[2, 2:5]*B[2:5, :]
    
    For the 2x2 x 2x7 subproblem (n1=2, m1=2, p=7):
    We can apply Strassen to the 2x2 part, but p=7 is not 2.
    Actually, let's think more carefully.
    
    The matrix multiplication tensor T_{i,j,k} = delta_{i,j} delta_{j,k} (in appropriate form)
    has rank at most n*m*p = 105 (schoolbook).
    
    Using Strassen on 2x2x2: rank 7 instead of 8.
    
    Let me partition:
    n=3: n1=2, n2=1
    m=5: m1=2, m2=3
    p=7: keep whole
    
    Sub-problems:
    1. A[0:2, 0:2] (2x2) * B[0:2, :] (2x7) -> contributes to C[0:2, :] (2x7)
       This is a 2x2 x 7 problem. We can think of B[0:2,:] as 7 columns of 2x1.
       For each column, it's a 2x2 x 1 problem... not quite.
       
       Actually, C[0:2, k] = A[0:2, 0:2] @ B[0:2, k] for k in 0..6
       This is: for each k, compute 2-vector = 2x2 matrix times 2-vector.
       The tensor for this is: T_{i,j,k} where i in {0,1}, j in {0,1}, k in {0..6}
       T_{i,j,k} = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
       
       This is essentially 7 independent 2x2x1 multiplications? No.
       It's a 2x2 matrix times a 2x7 matrix.
       
       The bilinear form: C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
       Number of terms: 2*2*7 = 28 (schoolbook)
       
       Can we do better? The 2x2 part suggests using Strassen, but Strassen is for 2x2*2x2->2x2.
       Here we have 2x2 * 2x7 -> 2x7.
       
       We can view this as 7 independent 2x2*2x1 problems, but that doesn't help directly.
       
       Alternatively, we can use the fact that a 2x2*2x2 multiplication has rank 7 (Strassen).
       If we partition the 7 columns of B into groups, we might apply Strassen.
       
       Let's try: partition p=7 into p1=2, p2=2, p3=3
       Then:
       - A[0:2,0:2] * B[0:2, 0:2] -> C[0:2, 0:2]: 2x2x2, use Strassen (rank 7)
       - A[0:2,0:2] * B[0:2, 2:4] -> C[0:2, 2:4]: 2x2x2, use Strassen (rank 7)
       - A[0:2,0:2] * B[0:2, 4:7] -> C[0:2, 4:7]: 2x2x3, schoolbook (rank 12)
       
       Total for this block: 7+7+12 = 26 (vs schoolbook 28, saving 2)
       
    2. A[0:2, 2:5] (2x3) * B[2:5, :] (3x7) -> C[0:2, :] (2x7)
       Schoolbook: 2*3*7 = 42
       
    3. A[2, 0:2] (1x2) * B[0:2, :] (2x7) -> C[2, :] (1x7)
       Schoolbook: 1*2*7 = 14
       
    4. A[2, 2:5] (1x3) * B[2:5, :] (3x7) -> C[2, :] (1x7)
       Schoolbook: 1*3*7 = 21
       
    Total: 26 + 42 + 14 + 21 = 103 (vs schoolbook 105, saving 2)
    
    Let me verify: 105 - 103 = 2. Yes.
    
    Now let's implement this.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # Partition indices
    n1, n2 = 2, 1
    m1, m2 = 2, 3
    p1, p2, p3 = 2, 2, 3  # Partition p=7 into 2+2+3
    
    # Strassen 2x2x2 decomposition
    # For 2x2*2x2->2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    # 
    # Strassen:
    # m1 = (a11+a22)*(b11+b22)
    # m2 = (a21+a22)*b11
    # m3 = a11*(b12-b22)
    # m4 = a22*(b21-b11)
    # m5 = (a11+a12)*b22
    # m6 = (a21-a11)*(b11+b12)
    # m7 = (a12-a22)*(b21+b22)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # We'll build U, V, W matrices
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Indexing: U row = i*m + j, V row = j*p + k, W row = k*n + i
    
    rank_total = 0
    U_rows = []
    U_vals = []
    V_rows = []
    V_vals = []
    W_rows = []
    W_vals = []
    
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        nonlocal rank_total
        U_rows.append(u_row)
        U_vals.append(u_val)
        V_rows.append(v_row)
        V_vals.append(v_val)
        W_rows.append(w_row)
        W_vals.append(w_val)
        rank_total += 1
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, rank 7)
    # i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*5 + j
    # V index: j*7 + k
    # W index: k*3 + i
    
    # Strassen terms for 2x2x2 with i_offset=0, j_offset=0, k_offset=0
    i_off, j_off, k_off = 0, 0, 0
    
    # m1 = (a11+a22)*(b11+b22)
    # U: a11 and a22 -> rows (i_off*5 + j_off) and ((i_off+1)*5 + (j_off+1))
    # V: b11 and b22 -> rows (j_off*7 + k_off) and ((j_off+1)*7 + (k_off+1))
    # W: c11 and c22 -> rows (k_off*3 + i_off) and ((k_off+1)*3 + (i_off+1))
    
    # m1: U has 1 at (i_off*5+j_off) and (i_off+1)*5+(j_off+1), V has 1 at j_off*7+k_off and (j_off+1)*7+(k_off+1), W has 1 at k_off*3+i_off and (k_off+1)*3+(i_off+1)
    # But each term is a single rank-1 component. So m1 contributes 1 rank with:
    # U: [1,0,1,0] pattern (for the 4 a-entries), V: [1,0,0,1] pattern, W: [1,0,0,1] pattern
    
    # Actually, each Strassen intermediate is a product of sums, so it's still rank 1 in the tensor sense.
    # m1 = (a11+a22)*(b11+b22) means:
    # U vector: has 1 at positions corresponding to a11 and a22
    # V vector: has 1 at positions corresponding to b11 and b22
    # W vector: has 1 at positions corresponding to c11 and c22
    
    # For m1:
    add_term(i_off*5 + j_off, j_off*7 + k_off, k_off*3 + i_off, 1, 1, 1)  # a11*b11 -> c11
    # Wait, that's not right. m1 = (a11+a22)*(b11+b22) is ONE term, not four.
    # The U vector for this term has 1 at a11 position AND a22 position.
    # But U is a matrix, so each column of U is a vector.
    # So for m1, the U column has 1 at row (i_off*5+j_off) [a11] and 1 at row ((i_off+1)*5+(j_off+1)) [a22]
    # Similarly for V and W.
    
    # I need to rethink. Let me build the matrices properly.
    
    # Let me use a different approach: build lists of (row, value) for each of U, V, W per rank column.
    
    U_cols = []  # list of (row, value) tuples
    V_cols = []
    W_cols = []
    
    def add_rank1(u_entries, v_entries, w_entries):
        """Add a rank-1 component. u_entries, v_entries, w_entries are lists of (row, val)."""
        U_cols.append(u_entries)
        V_cols.append(v_entries)
        W_cols.append(w_entries)
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Strassen 2x2x2
    # i in {0,1}, j in {0,1}, k in {0,1}
    # U row for A[i,j]: i*5 + j
    # V row for B[j,k]: j*7 + k
    # W row for C[i,k]: k*3 + i
    
    # m1 = (a00+a11)*(b00+b11) -> c00 += m1, c11 += m1
    add_rank1(
        [(0*5+0, 1), (1*5+1, 1)],  # a00, a11
        [(0*7+0, 1), (1*7+1, 1)],  # b00, b11
        [(0*3+0, 1), (1*3+1, 1)]   # c00, c11
    )
    
    # m2 = (a10+a11)*b00 -> c10 += m2, c11 -= m2
    add_rank1(
        [(1*5+0, 1), (1*5+1, 1)],  # a10, a11
        [(0*7+0, 1)],              # b00
        [(0*3+1, 1), (1*3+1, -1)]  # c10, c11
    )
    
    # m3 = a00*(b01-b11) -> c01 += m3, c11 += m3
    add_rank1(
        [(0*5+0, 1)],              # a00
        [(0*7+1, 1), (1*7+1, -1)], # b01, -b11
        [(1*3+0, 1), (1*3+1, 1)]   # c01, c11
    )
    
    # m4 = a11*(b10-b00) -> c10 += m4, c11 += m4
    add_rank1(
        [(1*5+1, 1)],              # a11
        [(1*7+0, 1), (0*7+0, -1)], # b10, -b00
        [(0*3+1, 1), (1*3+1, 1)]   # c10, c11
    )
    
    # m5 = (a00+a01)*b11 -> c11 -= m5, c01 += m5
    add_rank1(
        [(0*5+0, 1), (0*5+1, 1)],  # a00, a01
        [(1*7+1, 1)],              # b11
        [(1*3+1, -1), (1*3+0, 1)]  # c11, c01
    )
    
    # m6 = (a10-a00)*(b00+b01) -> c11 += m6
    add_rank1(
        [(1*5+0, 1), (0*5+0, -1)], # a10, -a00
        [(0*7+0, 1), (0*7+1, 1)],  # b00, b01
        [(1*3+1, 1)]               # c11
    )
    
    # m7 = (a01-a11)*(b10+b11) -> c00 += m7
    add_rank1(
        [(0*5+1, 1), (1*5+1, -1)], # a01, -a11
        [(1*7+0, 1), (1*7+1, 1)],  # b10, b11
        [(0*3+0, 1)]               # c00
    )
    
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (Strassen, rank 7)
    # i in {0,1}, j in {0,1}, k in {2,3}
    k_off = 2
    add_rank1(
        [(0*5+0, 1), (1*5+1, 1)],
        [(0*7+k_off, 1), (1*7+(k_off+1), 1)],
        [(k_off*3+0, 1), ((k_off+1)*3+1, 1)]
    )
    add_rank1(
        [(1*5+0, 1), (1*5+1, 1)],
        [(0*7+k_off, 1)],
        [(k_off*3+1, 1), ((k_off+1)*3+1, -1)]
    )
    add_rank1(
        [(0*5+0, 1)],
        [(0*7+(k_off+1), 1), (1*7+(k_off+1), -1)],
        [((k_off+1)*3+0, 1), ((k_off+1)*3+1, 1)]
    )
    add_rank1(
        [(1*5+1, 1)],
        [(1*7+k_off, 1), (0*7+k_off, -1)],
        [(k_off*3+1, 1), ((k_off+1)*3+1, 1)]
    )
    add_rank1(
        [(0*5+0, 1), (0*5+1, 1)],
        [(1*7+(k_off+1), 1)],
        [((k_off+1)*3+1, -1), ((k_off+1)*3+0, 1)]
    )
    add_rank1(
        [(1*5+0, 1), (0*5+0, -1)],
        [(0*7+k_off, 1), (0*7+(k_off+1), 1)],
        [((k_off+1)*3+1, 1)]
    )
    add_rank1(
        [(0*5+1, 1), (1*5+1, -1)],
        [(1*7+k_off, 1), (1*7+(k_off+1), 1)],
        [(k_off*3+0, 1)]
    )
    
    # Block 3: A[0:2,0:2] * B[0:2,4:7] -> C[0:2,4:7] (schoolbook, rank 2*2*3=12)
    # i in {0,1}, j in {0,1}, k in {4,5,6}
    for i in range(2):
        for j in range(2):
            for k in range(4, 7):
                add_rank1(
                    [(i*5+j, 1)],
                    [(j*7+k, 1)],
                    [(k*3+i, 1)]
                )
    
    # Block 4: A[0:2,2:5] * B[2:5,:] -> C[0:2,:] (schoolbook, rank 2*3*7=42)
    # i in {0,1}, j in {2,3,4}, k in {0..6}
    for i in range(2):
        for j in range(2, 5):
            for k in range(7):
                add_rank1(
                    [(i*5+j, 1)],
                    [(j*7+k, 1)],
                    [(k*3+i, 1)]
                )
    
    # Block 5: A[2,0:2] * B[0:2,:] -> C[2,:] (schoolbook, rank 1*2*7=14)
    # i=2, j in {0,1}, k in {0..6}
    for j in range(2):
        for k in range(7):
            add_rank1(
                [(2*5+j, 1)],
                [(j*7+k, 1)],
                [(k*3+2, 1)]
            )
    
    # Block 6: A[2,2:5] * B[2:5,:] -> C[2,:] (schoolbook, rank 1*3*7=21)
    # i=2, j in {2,3,4}, k in {0..6}
    for j in range(2, 5):
        for k in range(7):
            add_rank1(
                [(2*5+j, 1)],
                [(j*7+k, 1)],
                [(k*3+2, 1)]
            )
    
    rank_total = len(U_cols)
    
    # Build U, V, W matrices
    U = np.zeros((n*m, rank_total), dtype=int)
    V = np.zeros((m*p, rank_total), dtype=int)
    W = np.zeros((p*n, rank_total), dtype=int)
    
    for col, (u_entries, v_entries, w_entries) in enumerate(zip(U_cols, V_cols, W_cols)):
        for row, val in u_entries:
            U[row, col] += val
        for row, val in v_entries:
            V[row, col] += val
        for row, val in w_entries:
            W[row, col] += val
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
