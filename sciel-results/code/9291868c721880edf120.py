"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def get_strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
        U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
        Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # M1 = (A00 + A11)(B00 + B11) -> C00, C11
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10 + A11)B10 -> C10
        U[2][1] = 1; U[3][1] = 1
        V[2][1] = 1
        W[2][1] = 1
        
        # M3 = A00(B01 - B11) -> C01
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4 = A11(B10 - B00) -> C00, C10
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A00 + A01)B11 -> -C00, C01
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10 - A00)(B00 + B01) -> C11
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01 - A11)(B10 + B11) -> C00, C11
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W

    def add_schoolbook_block(U_full, V_full, W_full, col_offset, n_off, m_off, p_off, n_size, m_size, p_size):
        """
        Adds a schoolbook decomposition for a block of size n_size x m_size x p_size
        at offsets (n_off, m_off, p_off) to the full matrices.
        Returns the new column offset.
        """
        rank = n_size * m_size * p_size
        for i in range(n_size):
            for j in range(m_size):
                for k in range(p_size):
                    u_idx = (n_off + i) * m + (m_off + j)
                    v_idx = (m_off + j) * p + (p_off + k)
                    w_idx = (p_off + k) * n + (n_off + i)
                    
                    U_full[u_idx][col_offset] = 1
                    V_full[v_idx][col_offset] = 1
                    W_full[w_idx][col_offset] = 1
                    col_offset += 1
        return col_offset

    def add_strassen_block(U_full, V_full, W_full, col_offset, n_off, m_off, p_off):
        """
        Adds a Strassen decomposition for a 2x2x2 block
        at offsets (n_off, m_off, p_off) to the full matrices.
        Returns the new column offset.
        """
        U_s, V_s, W_s = get_strassen_2x2x2()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = (n_off + i) * m + (m_off + j)
                    U_full[u_idx][col_offset + r] += U_s[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    v_idx = (m_off + j) * p + (p_off + k)
                    V_full[v_idx][col_offset + r] += V_s[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    w_idx = (p_off + k) * n + (n_off + i)
                    W_full[w_idx][col_offset + r] += W_s[k * 2 + i][r]
        return col_offset + 7

    # Decomposition Strategy:
    # Split m=5 into [0:4] and [4:5]
    # Split p=6 into [0:4] and [4:6]
    # Blocks:
    # 1. n[0:4], m[0:4], p[0:4] -> 4x4x4 (Rank 28)
    # 2. n[0:4], m[4:5], p[0:4] -> 4x1x4 (Rank 16)
    # 3. n[0:4], m[0:4], p[4:6] -> 4x4x2 (Rank 28)
    # 4. n[0:4], m[4:5], p[4:6] -> 4x1x2 (Rank 8)
    # Total Rank = 28 + 16 + 28 + 8 = 80
    
    total_rank = 80
    U_full = [[0] * total_rank for _ in range(nm)]
    V_full = [[0] * total_rank for _ in range(mp)]
    W_full = [[0] * total_rank for _ in range(pn)]
    
    col = 0
    
    # Block 1: 4x4x4
    # Split into four 2x2x2 Strassen blocks
    col = add_strassen_block(U_full, V_full, W_full, col, 0, 0, 0)
    col = add_strassen_block(U_full, V_full, W_full, col, 0, 2, 0)
    col = add_strassen_block(U_full, V_full, W_full, col, 2, 0, 0)
    col = add_strassen_block(U_full, V_full, W_full, col, 2, 2, 0)
    
    # Block 2: 4x1x4 (Schoolbook)
    col = add_schoolbook_block(U_full, V_full, W_full, col, 0, 4, 0, 4, 1, 4)
    
    # Block 3: 4x4x2
    # Split into four 2x2x1 blocks? No, 2x2x2 Strassen requires p_size=2.
    # We have p_size=2.
    # So we can use Strassen for 2x2x2 blocks.
    # Split n=4 into 2+2, m=4 into 2+2.
    col = add_strassen_block(U_full, V_full, W_full, col, 0, 0, 4)
    col = add_strassen_block(U_full, V_full, W_full, col, 0, 2, 4)
    col = add_strassen_block(U_full, V_full, W_full, col, 2, 0, 4)
    col = add_strassen_block(U_full, V_full, W_full, col, 2, 2, 4)
    
    # Block 4: 4x1x2 (Schoolbook)
    col = add_schoolbook_block(U_full, V_full, W_full, col, 0, 4, 4, 4, 1, 2)
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U_full[i * m + j][r] * V_full[j * p + k][r] * W_full[k * n + i][r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
        if not valid:
            break
    
    if valid:
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        # Fallback to schoolbook if verification fails
        U_sb = [[0] * (n*m*p) for _ in range(nm)]
        V_sb = [[0] * (n*m*p) for _ in range(mp)]
        W_sb = [[0] * (n*m*p) for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
