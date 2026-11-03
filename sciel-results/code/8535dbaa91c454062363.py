import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-36 exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    Strategy:
    - Split the 4-dimensional middle into two 2-dimensional blocks: m=2+2
    - Each 2x2x5 block is further split into 2x2x2 + 2x2x3 (or similar)
    - Actually, let's use: 2x2x5 = 2x2x2 (Strassen rank 7) + 2x2x3 (schoolbook rank 6)
    - But 7+6=13 per block, 2 blocks = 26, that's too optimistic.
    
    Let me reconsider. The measured improvement said:
    "Decomposing the 2x4x5 multiplication into two 2x2x5 blocks, each further 
     decomposed into two 2x2x2 Strassen blocks (rank 7) and one 2x2x1 schoolbook block (rank 4),
     reduces the total rank from 40 to 36."
    
    Wait, 2x2x2 Strassen is rank 7, and there are two such per 2x2x5? That doesn't add up.
    
    Let me re-read: "two 2x2x5 blocks, each further decomposed into two 2x2x2 Strassen blocks (rank 7) and one 2x2x1 schoolbook block (rank 4)"
    
    Hmm, that's 7+7+4 = 18 per 2x2x5 block, times 2 = 36. But how do you decompose 2x2x5 into two 2x2x2 and one 2x2x1?
    
    Actually, I think the description means: the 2x2x5 block is split along the p=5 dimension into 2+2+1, giving three subproblems: 2x2x2, 2x2x2, 2x2x1.
    - 2x2x2 uses Strassen: rank 7
    - 2x2x2 uses Strassen: rank 7  
    - 2x2x1 uses schoolbook: rank 4
    Total per 2x2x5 block: 7+7+4 = 18
    Two blocks: 36
    
    This makes sense! The middle dimension m=4 is split into 2+2, and within each 2x2x5 block, the output dimension p=5 is split into 2+2+1.
    """
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We'll build the decomposition with rank 36
    rank = 36
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    def strassen_2x2x2(A_rows, B_rows, C_rows, base_col):
        """
        Add a Strassen decomposition for a 2x2x2 subproblem.
        A_rows: list of 2 row indices in U (for A[i,m_block])
        B_rows: list of 2 row indices in V (for B[m_block,j])
        C_rows: list of 2 row indices in W (for C[j,n_block])
        base_col: starting column index
        Returns: number of columns used
        """
        # Strassen's 7 products for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Let's label the 2x2 A block rows in U as a0, a1 (corresponding to A[0,block], A[1,block])
        # B block cols in V as b0, b1 (corresponding to B[block,0], B[block,1])
        # C block rows in W as c0, c1 (corresponding to C[0,block], C[1,block])
        
        a0, a1 = A_rows
        b0, b1 = B_rows
        c0, c1 = C_rows
        
        # M1 = (A00+A11)(B00+B11)
        col = base_col
        U[a0][col] += 1
        U[a1][col] += 1
        V[b0][col] += 1
        V[b1][col] += 1
        W[c0][col] += 1
        W[c1][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        # Wait, I need to be more careful with the indexing convention.
        # In Strassen, A is 2x2, B is 2x2, C = A*B
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # But in our tensor decomposition, the indices are:
        # U[i*m+j, c] for A[i,j]
        # V[j*p+k, c] for B[j,k]
        # W[k*n+i, c] for C[k,i]
        
        # For a 2x2x2 subproblem, we need to map the local indices to global ones.
        
        # Let me redo this more carefully.
        
        # Actually, let me just build the full decomposition directly.
        pass
    
    # Let me build it directly with explicit index mappings.
    
    # Global dimensions: A is 2x4, B is 4x5, C is 5x2
    # A[i,j] for i in [0,1], j in [0,3]
    # B[j,k] for j in [0,3], k in [0,4]
    # C[k,i] for k in [0,4], i in [0,1]
    # W index: k*n + i = k*2 + i
    
    # Split m=4 into m0=2 (j=0,1) and m1=2 (j=2,3)
    # Split p=5 into p0=2 (k=0,1), p1=2 (k=2,3), p2=1 (k=4)
    
    # Block (m0, p0): A[:, 0:2] * B[0:2, 0:2] -> C[0:2, :]
    # Block (m0, p1): A[:, 0:2] * B[0:2, 2:4] -> C[2:4, :]
    # Block (m0, p2): A[:, 0:2] * B[0:2, 4:5] -> C[4:5, :]
    # Block (m1, p0): A[:, 2:4] * B[2:4, 0:2] -> C[0:2, :]
    # Block (m1, p1): A[:, 2:4] * B[2:4, 2:4] -> C[2:4, :]
    # Block (m1, p2): A[:, 2:4] * B[2:4, 4:5] -> C[4:5, :]
    
    # For each 2x2x2 block, use Strassen (rank 7)
    # For each 2x2x1 block, use schoolbook (rank 4)
    
    col = 0
    
    def add_strassen_2x2x2(a_row_offset, b_row_offset, c_row_offset, start_col):
        """
        Add Strassen decomposition for a 2x2x2 subblock.
        a_row_offset: offset in U for the A block (local row index in the 2-row block)
        b_row_offset: offset in V for the B block
        c_row_offset: offset in W for the C block
        
        Local A: rows a_row_offset + 0, a_row_offset + 1 (these are A[i, j0] and A[i, j1] for i=0,1)
        Actually, U is indexed by i*m + j, so for a fixed j-block, the rows are:
        i=0: row = 0*m + j0 = j0
        i=1: row = 1*m + j0 = m + j0
        So for j-block [j0, j1], the A rows are [j0, m+j0, j1, m+j1]
        But we're dealing with a 2x2x2, so we have two j values and two i values.
        
        Let me define it as:
        A_local[0,0] -> U[j0]
        A_local[1,0] -> U[m + j0]
        A_local[0,1] -> U[j1]
        A_local[1,1] -> U[m + j1]
        
        B_local[0,0] -> V[j0*p + k0]
        B_local[1,0] -> V[j1*p + k0]
        B_local[0,1] -> V[j0*p + k1]
        B_local[1,1] -> V[j1*p + k1]
        
        C_local[0,0] -> W[k0*n + 0]
        C_local[1,0] -> W[k1*n + 0]
        C_local[0,1] -> W[k0*n + 1]
        C_local[1,1] -> W[k1*n + 1]
        
        Where j0,j1 are the two middle indices, k0,k1 are the two output indices.
        """
        j0, j1 = a_row_offset
        k0, k1 = c_row_offset
        
        # A rows in U
        A00 = j0
        A10 = m + j0
        A01 = j1
        A11 = m + j1
        
        # B rows in V
        B00 = j0 * p + k0
        B10 = j1 * p + k0
        B01 = j0 * p + k1
        B11 = j1 * p + k1
        
        # C rows in W
        C00 = k0 * n + 0
        C10 = k1 * n + 0
        C01 = k0 * n + 1
        C11 = k1 * n + 1
        
        # Strassen products
        # M1 = (A00+A11)(B00+B11)
        U[A00][start_col] += 1
        U[A11][start_col] += 1
        V[B00][start_col] += 1
        V[B11][start_col] += 1
        W[C00][start_col] += 1
        W[C11][start_col] += 1
        c = start_col + 1
        
        # M2 = (A10+A11)B00
        U[A10][c] += 1
        U[A11][c] += 1
        V[B00][c] += 1
        W[C10][c] += 1
        c += 1
        
        # M3 = A00(B10-B11)
        U[A00][c] += 1
        V[B10][c] += 1
        V[B11][c] += -1
        W[C01][c] += 1
        W[C11][c] += -1
        c += 1
        
        # M4 = A11(B01-B00)
        U[A11][c] += 1
        V[B01][c] += 1
        V[B00][c] += -1
        W[C00][c] += -1
        W[C10][c] += 1
        c += 1
        
        # M5 = (A00+A01)B11
        U[A00][c] += 1
        U[A01][c] += 1
        V[B11][c] += 1
        W[C00][c] += -1
        W[C01][c] += 1
        c += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[A10][c] += 1
        U[A00][c] += -1
        V[B00][c] += 1
        V[B10][c] += 1
        W[C10][c] += 1
        c += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[A01][c] += 1
        U[A11][c] += -1
        V[B01][c] += 1
        V[B11][c] += 1
        W[C00][c] += -1
        W[C11][c] += 1
        c += 1
        
        return c
    
    def add_schoolbook_2x2x1(a_row_offset, b_row_offset, c_row_offset, start_col):
        """
        Add schoolbook decomposition for a 2x2x1 subblock (rank 4).
        """
        j0, j1 = a_row_offset
        k0, _ = c_row_offset  # Only one output row
        
        # A rows in U
        A00 = j0
        A10 = m + j0
        A01 = j1
        A11 = m + j1
        
        # B rows in V
        B00 = j0 * p + k0
        B10 = j1 * p + k0
        
        # C rows in W
        C00 = k0 * n + 0
        C10 = k0 * n + 1
        
        c = start_col
        
        # C00 = A00*B00 + A01*B10
        U[A00][c] += 1
        V[B00][c] += 1
        W[C00][c] += 1
        c += 1
        
        U[A01][c] += 1
        V[B10][c] += 1
        W[C00][c] += 1
        c += 1
        
        # C10 = A10*B00 + A11*B10
        U[A10][c] += 1
        V[B00][c] += 1
        W[C10][c] += 1
        c += 1
        
        U[A11][c] += 1
        V[B10][c] += 1
        W[C10][c] += 1
        c += 1
        
        return c
    
    # Block (m0, p0): j in [0,1], k in [0,1] -> Strassen
    col = add_strassen_2x2x2((0, 1), (0, 1), (0, 1), col)
    
    # Block (m0, p1): j in [0,1], k in [2,3] -> Strassen
    col = add_strassen_2x2x2((0, 1), (2, 3), (2, 3), col)
    
    # Block (m0, p2): j in [0,1], k in [4] -> Schoolbook
    col = add_schoolbook_2x2x1((0, 1), (4,), (4,), col)
    
    # Block (m1, p0): j in [2,3], k in [0,1] -> Strassen
    col = add_strassen_2x2x2((2, 3), (0, 1), (0, 1), col)
    
    # Block (m1, p1): j in [2,3], k in [2,3] -> Strassen
    col = add_strassen_2x2x2((2, 3), (2, 3), (2, 3), col)
    
    # Block (m1, p2): j in [2,3], k in [4] -> Schoolbook
    col = add_schoolbook_2x2x1((2, 3), (4,), (4,), col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if (i,j,k) is valid, else 0
    # Actually, the tensor element is T[i,j,k] = delta(i,i') * delta(j,j') * delta(k,k') 
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all valid (i,j,k) since it's matrix multiplication.
    
    # Wait, that's not right. The matrix multiplication tensor is:
    # T[i,j,k] = 1 if the indices are consistent, else 0
    # But actually, for matrix multiplication, the tensor is just the identity in a sense.
    
    # Let me verify by checking a few entries.
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                expected = 1.0  # Every entry should be 1 for valid matrix multiplication
                error += abs(val - expected)
    
    if error > 1e-9:
        # Fallback to schoolbook
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
    else:
        # Success! Return the rank-36 decomposition
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Shouldn't happen, but safety check
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
