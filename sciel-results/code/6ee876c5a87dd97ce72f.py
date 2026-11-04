import time
import random
from itertools import product

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    def verify(rank, U, V, W):
        """Check T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
        U is 4x7 (for A indices (i,j)), V is 4x7 (for B indices (j,k)), W is 4x7 (for C indices (k,i)).
        Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Strassen 2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b11-b00)
        # M4 = a11(b10-b01)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3 = a00(b11-b00)
        U[0][2] = 1
        V[3][2] = 1; V[0][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = a11(b10-b01)
        U[3][3] = 1
        V[2][3] = 1; V[1][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[1][4] = 1
        
        # M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W

    def schoolbook_2x2x1():
        """Return (U, V, W) for 2x2 * 2x1 multiplication, rank 4.
        A is 2x2, B is 2x1, C is 2x1.
        U[i*2+j] for i in 0..1, j in 0..1 -> 4 entries
        V[j*1+k] for j in 0..1, k in 0..0 -> 2 entries  
        W[k*2+i] for k in 0..0, i in 0..1 -> 2 entries
        """
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        U[2][2] = 1; V[0][2] = 1; W[0][2] = 1
        U[3][3] = 1; V[1][3] = 1; W[0][3] = 1
        
        return U, V, W

    # Build the 6-block decomposition with rank 36
    # Partition:
    #   A rows: {0,1} (top) and {2,3} (bottom)
    #   B cols: {0,1} (left), {2,3} (mid), {4} (right)
    #   C cols: {0,1} (left), {2,3} (mid), {4} (right)
    #
    # Blocks:
    #   1. A_top x B_left  -> C_left:  2x2x2, rank 7 (Strassen)
    #   2. A_top x B_mid   -> C_mid:   2x2x2, rank 7 (Strassen)
    #   3. A_top x B_right -> C_right: 2x2x1, rank 4 (schoolbook)
    #   4. A_bot x B_left  -> C_left:  2x2x2, rank 7 (Strassen)
    #   5. A_bot x B_mid   -> C_mid:   2x2x2, rank 7 (Strassen)
    #   6. A_bot x B_right -> C_right: 2x2x1, rank 4 (schoolbook)
    # Total rank = 36
    
    total_rank = 36
    U = [[0]*total_rank for _ in range(n*m)]  # 16 x 36
    V = [[0]*total_rank for _ in range(m*p)]  # 20 x 36
    W = [[0]*total_rank for _ in range(p*n)]  # 20 x 36
    
    def embed_block(block_U, block_V, block_W, a_row_offset, b_col_offset, c_col_offset, base_rank):
        """Embed a local block into the global matrices.
        a_row_offset: starting row in A (0 or 2)
        b_col_offset: starting col in B/C (0, 2, or 4)
        c_col_offset: starting col in C (0, 2, or 4)
        base_rank: starting column in global rank dimension
        """
        r_block = len(block_U[0])
        
        # U: A sub-block, local (i,j) -> global (a_row_offset+i, j)
        for local_i in range(2):
            for local_j in range(2):
                gi = a_row_offset + local_i
                gj = local_j
                global_a_idx = gi * m + gj
                for c in range(r_block):
                    U[global_a_idx][base_rank + c] = block_U[local_i * 2 + local_j][c]
        
        # V: B sub-block, local (j,k) -> global (j, b_col_offset+k)
        num_b_cols = len(block_V) // 2
        for local_j in range(2):
            for local_k in range(num_b_cols):
                gj = local_j
                gk = b_col_offset + local_k
                global_b_idx = gj * p + gk
                for c in range(r_block):
                    V[global_b_idx][base_rank + c] = block_V[local_j * num_b_cols + local_k][c]
        
        # W: C sub-block, local (k,i) -> global (c_col_offset+k, a_row_offset+i)
        num_c_cols = len(block_W) // 2
        for local_k in range(num_c_cols):
            for local_i in range(2):
                gk = c_col_offset + local_k
                gi = a_row_offset + local_i
                global_c_idx = gk * n + gi
                for c in range(r_block):
                    W[global_c_idx][base_rank + c] = block_W[local_k * 2 + local_i][c]
    
    sU, sV, sW = strassen_2x2x2()
    bU, bV, bW = schoolbook_2x2x1()
    
    # Block 1: A rows 0-1, B cols 0-1, C cols 0-1
    embed_block(sU, sV, sW, 0, 0, 0, 0)
    # Block 2: A rows 0-1, B cols 2-3, C cols 2-3
    embed_block(sU, sV, sW, 0, 2, 2, 7)
    # Block 3: A rows 0-1, B col 4, C col 4
    embed_block(bU, bV, bW, 0, 4, 4, 14)
    # Block 4: A rows 2-3, B cols 0-1, C cols 0-1
    embed_block(sU, sV, sW, 2, 0, 0, 18)
    # Block 5: A rows 2-3, B cols 2-3, C cols 2-3
    embed_block(sU, sV, sW, 2, 2, 2, 25)
    # Block 6: A rows 2-3, B col 4, C col 4
    embed_block(bU, bV, bW, 2, 4, 4, 32)
    
    result = {"U": U, "V": V, "W": W}
    
    if verify(total_rank, U, V, W):
        return result
    
    # Fallback to schoolbook
    best = make_schoolbook()
    if verify(n*m*p, best['U'], best['V'], best['W']):
        return best
    
    return best

# EVOLVE-BLOCK-END
