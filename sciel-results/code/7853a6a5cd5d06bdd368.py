"""
Program to find rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy:
  Partition the 3x10 matrix B into five 3x2 blocks.
  For each 2x3 * 3x2 block, use a rank-11 decomposition:
    - Rank-7 Strassen decomposition for the 2x2 * 2x2 sub-block (columns 0,1 of A, rows 0,1 of B).
    - Rank-4 schoolbook decomposition for the remaining 2x1 * 1x2 part (column 2 of A, row 2 of B).
  Total rank: 5 blocks * 11 = 55.
  This is an improvement over the schoolbook rank of 60.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # Safety margin before 150s kill


def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen products:
    # M1 = (a00+a10)(b00+b10)
    U[0, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    # Note: W indexing is [k*n + i]. n=2.
    # C00 -> k=0, i=0 -> idx 0
    # C01 -> k=1, i=0 -> idx 2
    # C10 -> k=0, i=1 -> idx 1
    # C11 -> k=1, i=1 -> idx 3
    
    # Let's re-verify W indices for 2x2:
    # C[i, k]. W[k*2 + i].
    # C[0,0] -> W[0]
    # C[0,1] -> W[2]
    # C[1,0] -> W[1]
    # C[1,1] -> W[3]
    
    # M1 contributes to C00 and C11
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = a10*b00
    U[2, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M3 = a00*(b10+b11)
    U[0, 2] = 1
    V[2, 2] = 1
    V[3, 2] = 1
    W[2, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = (a00+a10)*(b11-b10)
    U[0, 3] = 1; U[2, 3] = 1
    V[3, 3] = 1; V[2, 3] = -1
    W[0, 3] = -1 # C00
    W[1, 3] = 1  # C10
    
    # M5 = (a00+a01)*(b00-b10)
    U[0, 4] = 1; U[1, 4] = 1
    V[0, 4] = 1; V[2, 4] = -1
    W[2, 4] = -1 # C01
    W[3, 4] = 1  # C11
    
    # M6 = (a10+a11)*(b00+b10)
    U[2, 5] = 1; U[3, 5] = 1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = -1 # C11
    
    # M7 = (a01-a10)*(b11+b10)
    U[1, 6] = 1; U[2, 6] = -1
    V[3, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1  # C00
    W[2, 6] = 1  # C01
    
    return U, V, W


def _schoolbook_2x1x2():
    """
    Returns U, V, W for 2x1 * 1x2 schoolbook decomposition.
    U: (2, 4), V: (2, 4), W: (4, 4)
    A is 2x1 (col 2 of original A), B is 1x2 (row 2 of original B).
    Output C is 2x2.
    U indices: [i*1 + 0] -> i in {0,1}. So U[0], U[1].
    V indices: [0*2 + k] -> k in {0,1}. So V[0], V[1].
    W indices: [k*2 + i].
    """
    rank = 4
    U = np.zeros((2, rank))
    V = np.zeros((2, rank))
    W = np.zeros((4, rank))
    
    # M8 = a02 * b20 -> C00
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # M9 = a02 * b21 -> C01
    U[0, 1] = 1
    V[1, 1] = 1
    W[2, 1] = 1
    
    # M10 = a12 * b20 -> C10
    U[1, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # M11 = a12 * b21 -> C11
    U[1, 3] = 1
    V[1, 3] = 1
    W[3, 3] = 1
    
    return U, V, W


def _rank11_decomp_2x3_3x2():
    """
    Combines Strassen (7) and Schoolbook (4) for 2x3 * 3x2.
    Returns U (6, 11), V (6, 11), W (4, 11).
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    U_sb, V_sb, W_sb = _schoolbook_2x1x2()
    
    U = np.zeros((6, 11))
    V = np.zeros((6, 11))
    W = np.zeros((4, 11))
    
    # Fill Strassen part (cols 0-6)
    # A indices for Strassen: 0,1 (row 0, col 0,1), 2,3 (row 1, col 0,1)
    # U_s is 4x7. Rows: 0:a00, 1:a01, 2:a10, 3:a11.
    # Our U is 6x11. Rows: 0:a00, 1:a01, 2:a02, 3:a10, 4:a11, 5:a12.
    # So U_s[0] -> U[0], U_s[1] -> U[1], U_s[2] -> U[3], U_s[3] -> U[4].
    U[0, 0:7] = U_s[0, :]
    U[1, 0:7] = U_s[1, :]
    U[3, 0:7] = U_s[2, :]
    U[4, 0:7] = U_s[3, :]
    
    # V_s is 4x7. Rows: 0:b00, 1:b01, 2:b10, 3:b11.
    # Our V is 6x11. Rows: 0:b00, 1:b01, 2:b10, 3:b11, 4:b20, 5:b21.
    # So V_s[0] -> V[0], V_s[1] -> V[1], V_s[2] -> V[2], V_s[3] -> V[3].
    V[0, 0:7] = V_s[0, :]
    V[1, 0:7] = V_s[1, :]
    V[2, 0:7] = V_s[2, :]
    V[3, 0:7] = V_s[3, :]
    
    # W_s is 4x7. Rows: 0:C00, 1:C10, 2:C01, 3:C11.
    # Our W is 4x11. Same indexing.
    W[:, 0:7] = W_s
    
    # Fill Schoolbook part (cols 7-10)
    # U_sb is 2x4. Rows: 0:a02, 1:a12.
    # Our U rows: 2:a02, 5:a12.
    U[2, 7:11] = U_sb[0, :]
    U[5, 7:11] = U_sb[1, :]
    
    # V_sb is 2x4. Rows: 0:b20, 1:b21.
    # Our V rows: 4:b20, 5:b21.
    V[4, 7:11] = V_sb[0, :]
    V[5, 7:11] = V_sb[1, :]
    
    # W_sb is 4x4. Same W indexing.
    W[:, 7:11] = W_sb
    
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    num_blocks = p // 2  # 5 blocks
    U_block, V_block, W_block = _rank11_decomp_2x3_3x2()
    rank_per_block = U_block.shape[1]  # 11
    
    total_rank = num_blocks * rank_per_block  # 55
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for block in range(num_blocks):
        offset = block * rank_per_block
        k_start = block * 2
        
        # U_block is the same for all blocks
        U[:, offset:offset + rank_per_block] = U_block
        
        # V_block: map 3x2 block to full 3x10
        # V_block rows: 0:b00, 1:b01, 2:b10, 3:b11, 4:b20, 5:b21
        # V rows: j*p + k
        for j in range(3):
            for k in range(2):
                V[j * p + (k_start + k), offset:offset + rank_per_block] = V_block[j * 2 + k, :]
        
        # W_block: map 2x2 output to full 2x10
        # W_block rows: 0:C00, 1:C10, 2:C01, 3:C11 (k*2+i)
        # W rows: k*n + i
        for k in range(2):
            for i in range(2):
                W[(k_start + k) * n + i, offset:offset + rank_per_block] = W_block[k * 2 + i, :]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    
    # Verify correctness
    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]
    
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]
        
        A_weighted = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_weighted[i, j] = u_r[i * m + j]
        
        B_weighted = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_weighted[j, k] = v_r[j * p + k]
        
        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)
        
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * w_r[k * n + i]
    
    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
