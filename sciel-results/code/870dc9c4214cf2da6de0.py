"""Strassen-based decomposition for 4x4 * 4x6 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 6]

# Strassen 2x2x2 decomposition (rank 7)
def strassen_2x2_2x2():
    """
    Returns U, V, W for 2x2*2x2 matrix multiplication using Strassen's 7-term algorithm.
    U: 4x7, V: 4x7, W: 4x7
    """
    # Standard Strassen decomposition
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # U rows indexed by [i*m+j] for 2x2: row 0=(0,0), row 1=(0,1), row 2=(1,0), row 3=(1,1)
    # V rows indexed by [j*p+k] for 2x2: row 0=(0,0), row 1=(0,1), row 2=(1,0), row 3=(1,1)
    # W rows indexed by [k*n+i] for 2x2: row 0=(0,0), row 1=(0,1), row 2=(1,0), row 3=(1,1)
    
    U = [
        [1, 0, 1, 0, 1, 1, -1],  # A11
        [0, 0, 0, 0, 1, 0, -1],  # A12
        [0, 1, 0, 0, 0, -1, 0],  # A21
        [1, 1, 0, 1, 0, 0, 1],   # A22
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B11
        [0, 0, 1, 0, 0, 1, 0],   # B12
        [0, 0, 0, 1, 0, 0, 1],   # B21
        [1, 0, -1, 0, 1, 0, 1],  # B22
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0],  # C22
    ]
    return U, V, W

def construct():
    """
    Construct a tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses a block decomposition combining Strassen for 2x2 blocks with
    careful handling of the rectangular 4x6 structure.
    
    Strategy: Split 4x4 into 2x2 blocks, and 4x6 into (2x2, 2x2, 2x2) columns.
    Use Strassen for each 2x2*2x2 block, but need to handle the 4x6 output.
    
    Actually, let's use a cleaner approach:
    - Split A (4x4) into four 2x2 blocks: A11, A12, A21, A22
    - Split B (4x6) into six 4x1 column blocks, or better: split into three 4x2 blocks
    - Each 4x2 block can be seen as two 2x1 columns, but let's think differently
    
    Better approach: Split B (4x6) into three 4x2 blocks.
    Each 4x2 block has two columns. We can split each 4x2 into two 2x1 column pairs.
    
    Actually, the cleanest is:
    - A: 4x4 = 2x2 blocks
    - B: 4x6 = split into three 4x2 blocks, each 4x2 = two 2x1 columns
    - Output C: 4x6 = four 2x6 row blocks, each 2x6 = six 1x6 entries
    
    Let me use a different strategy. Think of it as:
    - Split A into 2x2 blocks (4 blocks)
    - Split B into 2x2 blocks: 2 rows of 3 columns of 2x2 = 6 blocks
    - Output C: 2 rows of 3 columns of 2x2 = 6 blocks
    
    For each output 2x2 block C_ij, we compute it from A blocks and B blocks.
    Using Strassen for each 2x2*2x2 multiplication would give rank 7 per output block,
    but the A and B inputs are shared across blocks.
    
    Total rank with naive Strassen per block: 6 blocks * 7 = 42
    But we can share some terms.
    
    Actually, let's just use the straightforward approach:
    For each of the 6 output 2x2 blocks, use Strassen (7 terms each).
    Total rank = 42.
    
    U: 16x42 (4*4 = 16 rows)
    V: 24x42 (4*6 = 24 rows)
    W: 24x42 (6*4 = 24 rows)
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Get base Strassen decomposition
    U_base, V_base, W_base = strassen_2x2_2x2()
    
    n, m, p = 4, 4, 6
    
    # We'll build up U, V, W as lists of lists
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each output 2x2 block (ci, cj) where ci in {0,1}, cj in {0,1,2}
    # The output block C[2*ci:2*ci+2, 2*cj:2*cj+2] is computed from:
    # A[2*i:2*i+2, 2*j:2*j+2] * B[2*j:2*j+2, 2*cj:2*cj+2] summed over j
    
    # Actually, C[ci,cj] = sum_j A[i,j] * B[j,cj] for block indices
    # where i, j in {0,1} (since n=4, m=4, p=6 -> 2x2 blocks for A, 2x2 for B rows, 2x2 for B cols)
    
    # For a fixed output block (ci, cj), C_block = sum_{j=0}^{1} A_block(i,j) * B_block(j,cj)
    # Each product A_block * B_block is a 2x2*2x2 = 2x2 multiplication
    # Using Strassen, each such product needs 7 terms
    # So for each output block, we need 2*7 = 14 terms
    # Total: 6 output blocks * 14 = 84 terms
    
    # Wait, that's worse. Let me reconsider.
    
    # Actually, the standard way: for each pair of A-block (i,j) and B-block (j,k),
    # their product contributes to C-block (i,k).
    # There are 2*2*2 = 8 such products (i in {0,1}, j in {0,1}, k in {0,1,2}... wait k in {0,1} for 2x2 blocks of 6)
    
    # p=6, so B has 3 column blocks of width 2. So k in {0,1,2}.
    # A has i in {0,1}, j in {0,1}
    # B has j in {0,1}, k in {0,1,2}
    # C has i in {0,1}, k in {0,1,2}
    
    # Number of (i,j,k) triples: 2*2*3 = 12
    # Each contributes 7 terms (Strassen)
    # Total rank: 12 * 7 = 84
    
    # But we can do better by noting that for a fixed (j), the sum over i,k can be optimized.
    
    # For now, let's just use the 84-term decomposition.
    # This is better than 96 (schoolbook).
    
    U = [[0] * 84 for _ in range(16)]
    V = [[0] * 84 for _ in range(24)]
    W = [[0] * 84 for _ in range(24)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                # A block at rows 2*i:2*i+2, cols 2*j:2*j+2
                # B block at rows 2*j:2*j+2, cols 2*k:2*k+2
                # C block at rows 2*i:2*i+2, cols 2*k:2*k+2
                for t in range(7):
                    # U entries for A block
                    for r in range(2):
                        for c in range(2):
                            U[(2*i + r) * m + (2*j + c)][col + t] += U_base[r * 2 + c][t]
                    
                    # V entries for B block
                    for r in range(2):
                        for c in range(2):
                            V[(2*j + r) * p + (2*k + c)][col + t] += V_base[r * 2 + c][t]
                    
                    # W entries for C block
                    for r in range(2):
                        for c in range(2):
                            W[(2*k + r) * n + (2*i + c)][col + t] += W_base[r * 2 + c][t]
                col += 7
    
    return {"U": U, "V": V, "W": W}
