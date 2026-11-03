"""Matrix multiplication 3x4x6 with block-structured rank-66 decomposition."""
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) each of shape (4, 7) for the 2x2x2 tensor.
    Indexing: U[row, c] where row = i*2+j (i=row of A, j=col of A)
              V[row, c] where row = j*2+k (j=row of B, k=col of B)
              W[row, c] where row = k*2+i (k=col of C, i=row of C)
    """
    # Strassen's 7 products for C = A*B where A,B are 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b01)
    # m7 = (a01-a11)*(b10+b11)
    #
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    U = [[0]*7 for _ in range(4)]  # 2x2 A flattened: row = i*2+j
    V = [[0]*7 for _ in range(4)]  # 2x2 B flattened: row = j*2+k
    W = [[0]*7 for _ in range(4)]  # 2x2 C flattened: row = k*2+i
    
    # m1 = (a00+a11)*(b00+b11)
    # A factors: a00 -> U[0,0]=1, a11 -> U[3,0]=1
    # B factors: b00 -> V[0,0]=1, b11 -> V[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # c00 += m1, c11 += m1
    # c00 is at W[0*2+0][0] = W[0][0], c11 is at W[1*2+1][0] = W[3][0]
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (a10+a11)*b00
    # A: a10 -> U[2,1]=1, a11 -> U[3,1]=1
    # B: b00 -> V[0,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # c10 += m2, c11 -= m2
    # c10 is at W[0*2+1][1] = W[2][1], c11 is at W[3][1]
    W[2][1] = 1; W[3][1] = -1
    
    # m3 = a00*(b01-b11)
    # A: a00 -> U[0,2]=1
    # B: b01 -> V[1,2]=1, b11 -> V[3,2]=-1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # c01 += m3, c11 += m3
    # c01 is at W[1*2+0][2] = W[2][2], c11 is at W[3][2]
    W[2][2] = 1; W[3][2] = 1
    
    # m4 = a11*(b10-b00)
    # A: a11 -> U[3,3]=1
    # B: b10 -> V[2,3]=1, b00 -> V[0,3]=-1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # c00 += m4, c10 += m4
    # c00 is at W[0][3], c10 is at W[2][3]
    W[0][3] = 1; W[2][3] = 1
    
    # m5 = (a00+a01)*b11
    # A: a00 -> U[0,4]=1, a01 -> U[1,4]=1
    # B: b11 -> V[3,4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # c00 -= m5, c01 += m5
    # c00 is at W[0][4], c01 is at W[2][4]
    W[0][4] = -1; W[2][4] = 1
    
    # m6 = (a10-a00)*(b00+b01)
    # A: a10 -> U[2,5]=1, a00 -> U[0,5]=-1
    # B: b00 -> V[0,5]=1, b01 -> V[1,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # c11 += m6
    # c11 is at W[3][5]
    W[3][5] = 1
    
    # m7 = (a01-a11)*(b10+b11)
    # A: a01 -> U[1,6]=1, a11 -> U[3,6]=-1
    # B: b10 -> V[2,6]=1, b11 -> V[3,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # c00 += m7
    # c00 is at W[0][6]
    W[0][6] = 1
    
    return U, V, W


def _direct_1x2x2():
    """
    Direct decomposition for 1x2 * 2x2 -> 1x2.
    Returns (U, V, W) each of shape (2, 4) for the 1x2x2 tensor.
    U: shape (1*2, 4) = (2, 4), row = i*2+j but i is always 0, so row = j
    V: shape (2*2, 4) = (4, 4), row = j*2+k
    W: shape (2*1, 4) = (2, 4), row = k*1+i = k (since i=0)
    """
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # 4 multiplications: A[0,0]*B[0,0], A[0,0]*B[0,1], A[0,1]*B[1,0], A[0,1]*B[1,1]
    U = [[0]*4 for _ in range(2)]  # rows: j=0, j=1
    V = [[0]*4 for _ in range(4)]  # rows: j*2+k
    W = [[0]*4 for _ in range(2)]  # rows: k=0, k=1
    
    # c=0: A[0,0]*B[0,0] -> C[0,0]
    U[0][0] = 1  # j=0
    V[0][0] = 1  # j=0, k=0
    W[0][0] = 1  # k=0
    
    # c=1: A[0,0]*B[0,1] -> C[0,1]
    U[0][1] = 1  # j=0
    V[1][1] = 1  # j=0, k=1
    W[1][1] = 1  # k=1
    
    # c=2: A[0,1]*B[1,0] -> C[0,0]
    U[1][2] = 1  # j=1
    V[2][2] = 1  # j=1, k=0
    W[0][2] = 1  # k=0
    
    # c=3: A[0,1]*B[1,1] -> C[0,1]
    U[1][3] = 1  # j=1
    V[3][3] = 1  # j=1, k=1
    W[1][3] = 1  # k=1
    
    return U, V, W


def construct():
    """
    Block-structured decomposition for 3x4 * 4x6 -> 3x6.
    
    Partition:
    - n=3 split as 1+2 (rows of A, rows of C)
    - m=4 split as 2+2 (cols of A, rows of B)
    - p=6 split as 2+2+2 (cols of B, cols of C)
    
    Blocks:
    - 6 blocks of type 1x2 x 2x2 -> 1x2 (rank 4 each) = 24
    - 6 blocks of type 2x2 x 2x2 -> 2x2 (Strassen, rank 7 each) = 42
    Total rank = 66
    """
    n, m, p = 3, 4, 6
    total_rank = 66
    
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 66
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 66
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 66
    
    col_offset = 0
    
    # Get sub-decompositions
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    direct_U, direct_V, direct_W = _direct_1x2x2()
    
    # Define the block structure
    # n_blocks: [(start_row, size)] for A rows / C rows
    n_blocks = [(0, 1), (1, 2)]  # row 0, then rows 1-2
    # m_blocks: for A cols / B rows
    m_blocks = [(0, 2), (2, 2)]  # cols 0-1, then cols 2-3
    # p_blocks: for B cols / C cols
    p_blocks = [(0, 2), (2, 2), (4, 2)]  # cols 0-1, 2-3, 4-5
    
    for ni, (n_start, n_size) in enumerate(n_blocks):
        for mi, (m_start, m_size) in enumerate(m_blocks):
            for pi, (p_start, p_size) in enumerate(p_blocks):
                # Block: A[n_start:n_start+n_size, m_start:m_start+m_size] 
                #        * B[m_start:m_start+m_size, p_start:p_start+p_size]
                #        -> C[n_start:n_start+n_size, p_start:p_start+p_size]
                
                if n_size == 1 and m_size == 2 and p_size == 2:
                    # 1x2 x 2x2 -> 1x2, use direct (rank 4)
                    sub_U, sub_V, sub_W = direct_U, direct_V, direct_W
                    sub_rank = 4
                elif n_size == 2 and m_size == 2 and p_size == 2:
                    # 2x2 x 2x2 -> 2x2, use Strassen (rank 7)
                    sub_U, sub_V, sub_W = strassen_U, strassen_V, strassen_W
                    sub_rank = 7
                else:
                    raise ValueError(f"Unexpected block size {n_size}x{m_size}x{p_size}")
                
                # Copy into global arrays
                for c in range(sub_rank):
                    for i in range(n_size):
                        for j in range(m_size):
                            # A index: (n_start+i) * m + (m_start+j)
                            a_idx = (n_start + i) * m + (m_start + j)
                            U[a_idx][col_offset + c] += sub_U[i * m_size + j][c]
                    
                    for j in range(m_size):
                        for k in range(p_size):
                            # B index: (m_start+j) * p + (p_start+k)
                            b_idx = (m_start + j) * p + (p_start + k)
                            V[b_idx][col_offset + c] += sub_V[j * p_size + k][c]
                    
                    for k in range(p_size):
                        for i in range(n_size):
                            # C index: (p_start+k) * n + (n_start+i)
                            c_idx = (p_start + k) * n + (n_start + i)
                            W[c_idx][col_offset + c] += sub_W[k * n_size + i][c]
                
                col_offset += sub_rank
    
    assert col_offset == total_rank, f"col_offset {col_offset} != total_rank {total_rank}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
